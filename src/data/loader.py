"""Load parquet files from training and test folders."""

import glob
from pathlib import Path
from typing import Optional

import pandas as pd


def load_parquet_files(
    folder: Path,
    n_per_file: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load all parquet files from *folder*, optionally sampling *n_per_file* rows each."""
    files = sorted(glob.glob(str(folder / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder}")

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        if n_per_file is not None:
            n = min(n_per_file, len(df))
            df = df.sample(n=n, random_state=random_state)
        frames.append(df)
        print(f"  {Path(f).name}: {len(df):,} rows")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(combined):,} rows from {len(files)} file(s)")
    return combined


def load_taxi_zones(path: Path) -> pd.DataFrame:
    """Load the TLC taxi zone lookup CSV, enriched with zone centroid
    coordinates (see scripts/build_zone_centroids.py) so downstream feature
    engineering can compute a real geographic distance between PU/DO zones.
    Zones 264 ("Unknown") and 265 ("Outside of NYC") have no real geometry,
    so they get NaN centroids -- handled explicitly in
    features/domain.add_zone_geo_distance_features, not silently here.
    """
    from src.config import DATA_PATHS  # local import: avoids a config<->loader cycle

    zones = pd.read_csv(path)
    zones.columns = [c.strip() for c in zones.columns]

    centroids_path = DATA_PATHS.get("taxi_zone_centroids")
    if centroids_path is not None and Path(centroids_path).exists():
        centroids = pd.read_csv(centroids_path)[["LocationID", "centroid_x_ft", "centroid_y_ft"]]
        zones = zones.merge(centroids, on="LocationID", how="left")
    return zones
