"""Download a minimal subset of NYC TLC yellow-taxi parquet files for CI.

All files are public on the TLC website. This script is called by the
GitHub Actions workflow before training so the runner has data to work with.
It is safe to run locally too — files that already exist are skipped.

Usage:
    python scripts/download_data.py
"""

import sys
import urllib.request
from pathlib import Path

TLC_BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"

# Months to download for training_set/ — spread across 2024-2025 to give the
# temporal CV folds enough coverage, plus the Nov-Dec 2025 val split months.
TRAINING_MONTHS = [
    ("2024", "01"), ("2024", "04"), ("2024", "07"), ("2024", "10"),
    ("2025", "01"), ("2025", "04"), ("2025", "07"), ("2025", "10"),
    ("2025", "11"), ("2025", "12"),
]

# 2026 test set — may not be available yet; failures are non-fatal.
TEST_MONTHS = [
    ("2026", "01"), ("2026", "02"),
]


def download_file(year: str, month: str, dest_dir: Path, required: bool = True) -> bool:
    fname = f"yellow_tripdata_{year}-{month}.parquet"
    dest = dest_dir / fname
    if dest.exists():
        print(f"  skip  {fname}  (already exists)")
        return True
    url = f"{TLC_BASE}/{fname}"
    print(f"  fetch {fname} …", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = dest.stat().st_size / 1_048_576
        print(f"done ({size_mb:.1f} MB)")
        return True
    except Exception as exc:
        print(f"FAILED — {exc}")
        if dest.exists():
            dest.unlink()
        if required:
            raise
        return False


def main() -> int:
    train_dir = Path("training_set")
    test_dir = Path("test_set")
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    print("=== Downloading training data ===")
    for year, month in TRAINING_MONTHS:
        download_file(year, month, train_dir, required=True)

    print("\n=== Downloading test data ===")
    ok = 0
    for year, month in TEST_MONTHS:
        if download_file(year, month, test_dir, required=False):
            ok += 1
    if ok == 0:
        print("  WARNING: no test data downloaded — evaluate.py will be skipped in CI.")

    print("\nData download complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
