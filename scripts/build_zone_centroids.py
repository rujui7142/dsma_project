"""Build taxi_zone_centroids.csv from the official TLC taxi zone shapefile.

Computes an area-weighted polygon centroid (correctly handling multi-part
zones and holes via the standard signed-area shoelace formula per ring) for
each of the 263 real taxi zones, in the shapefile's native projected CRS
(NAD83 State Plane New York Long Island, US survey feet -- confirmed via
taxi_zones.prj). That CRS is already a planar, foot-based, roughly
north/east-aligned coordinate system, so |x1-x2| + |y1-y2| on these centroids
is a direct, meaningful Manhattan-distance feature -- no further reprojection
needed, and no geopandas/pyproj/GDAL dependency required (just pyshp, a
pure-Python shapefile reader).

Zones 264 ("Unknown") and 265 ("Outside of NYC") have no real geometry and
are absent from the shapefile -- confirmed, not a bug -- so they get no
centroid; see features/domain.add_zone_geo_distance_features for how the
resulting NaNs are handled downstream.

This is a one-time, reproducible data-prep step (like scripts/download_data.py
and the macro_data/*.csv fetch) -- taxi_zone_centroids.csv is committed, not
regenerated on every run, so `pyshp` (this script's only non-stdlib dependency)
is deliberately NOT in requirements.txt -- the training/inference pipeline
only ever reads the committed CSV via pandas.

Usage:
    pip install pyshp
    python scripts/build_zone_centroids.py
"""

import csv
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import shapefile

ROOT_DIR = Path(__file__).parent.parent
SHAPEFILE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
OUTPUT_PATH = ROOT_DIR / "taxi_zone_centroids.csv"


def _ring_area_centroid(points):
    """Signed area + centroid of a single polygon ring (shoelace formula)."""
    n = len(points)
    area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area *= 0.5
    if area == 0:
        return 0.0, sum(p[0] for p in points) / n, sum(p[1] for p in points) / n
    return area, cx / (6 * area), cy / (6 * area)


def _polygon_centroid(shape):
    """Area-weighted centroid across all rings of a (possibly multi-part,
    possibly holed) polygon. Holes naturally subtract out because pyshp
    winds them in the opposite direction from exterior rings, giving them a
    negative signed area.
    """
    parts = list(shape.parts) + [len(shape.points)]
    total_area = sum_ax = sum_ay = 0.0
    for i in range(len(parts) - 1):
        ring_pts = shape.points[parts[i]:parts[i + 1]]
        if len(ring_pts) < 3:
            continue
        area, cx, cy = _ring_area_centroid(ring_pts)
        total_area += area
        sum_ax += area * cx
        sum_ay += area * cy
    if total_area == 0:
        return None
    return sum_ax / total_area, sum_ay / total_area


def main():
    # Manual mkdtemp + ignore_errors cleanup rather than TemporaryDirectory:
    # pyshp's Reader keeps the .dbf/.shp file handles open, and on Windows an
    # open file can't be deleted -- TemporaryDirectory's context-manager
    # cleanup would crash with PermissionError even after we're done reading.
    tmpdir = Path(tempfile.mkdtemp())
    try:
        zip_path = tmpdir / "taxi_zones.zip"
        print(f"Downloading {SHAPEFILE_URL} ...")
        urllib.request.urlretrieve(SHAPEFILE_URL, zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir)

        shp_path = next(tmpdir.rglob("taxi_zones.shp"))
        sf = shapefile.Reader(str(shp_path))

        rows = []
        for sr in sf.shapeRecords():
            loc_id = sr.record["LocationID"]
            centroid = _polygon_centroid(sr.shape)
            if centroid is None:
                print(f"  WARNING: zone {loc_id} ({sr.record['zone']}) has zero-area geometry, skipping")
                continue
            cx, cy = centroid
            rows.append((loc_id, sr.record["zone"], sr.record["borough"], cx, cy))
        sf.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    rows.sort(key=lambda r: r[0])
    with open(OUTPUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["LocationID", "zone", "borough", "centroid_x_ft", "centroid_y_ft"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} zone centroids to {OUTPUT_PATH}")
    missing = set(range(1, 266)) - {r[0] for r in rows}
    print(f"LocationIDs with no geometry (expected: 264 Unknown, 265 Outside of NYC): {sorted(missing)}")


if __name__ == "__main__":
    sys.exit(main())
