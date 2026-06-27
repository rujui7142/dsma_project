"""Data cleaning and validation for NYC TLC trip records.

All cleaning decisions are grounded in the EDA (01_EDA_Main.ipynb) and
replicated here in a reproducible, modular form.
"""

from typing import List

import numpy as np
import pandas as pd

from src.config import CLEANING, TARGET_COL


# ---------------------------------------------------------------------------
# Step-wise transforms (each returns a new DataFrame copy)
# ---------------------------------------------------------------------------

def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive pickup time components needed for feature engineering."""
    df = df.copy()
    dt = pd.to_datetime(df["tpep_pickup_datetime"])
    df["pickup_hour"] = dt.dt.hour
    df["pickup_dayofweek"] = dt.dt.dayofweek  # 0=Monday … 6=Sunday
    df["pickup_month"] = dt.dt.month
    df["pickup_year"] = dt.dt.year
    return df


def compute_trip_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Add trip_duration_min column (used for cleaning validation only)."""
    df = df.copy()
    df["trip_duration_min"] = (
        pd.to_datetime(df["tpep_dropoff_datetime"])
        - pd.to_datetime(df["tpep_pickup_datetime"])
    ).dt.total_seconds() / 60
    return df


def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """Compute prediction target: total_amount − tip_amount."""
    df = df.copy()
    df[TARGET_COL] = df["total_amount"] - df["tip_amount"]
    return df


def filter_valid_trips(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove rows with physically impossible or corrupt values."""
    n_before = len(df)
    mask = (
        df["trip_distance"].between(CLEANING["trip_distance_min"], CLEANING["trip_distance_max"])
        & df["trip_duration_min"].between(CLEANING["trip_duration_min"], CLEANING["trip_duration_max"])
        & df[TARGET_COL].between(CLEANING["total_fare_min"], CLEANING["total_fare_max"])
        & (df["PULocationID"] > 0)
        & (df["DOLocationID"] > 0)
    )
    df = df[mask].copy()
    if verbose:
        removed = n_before - len(df)
        pct = removed / n_before * 100
        print(f"  filter_valid_trips: removed {removed:,} rows ({pct:.1f}%)")
    return df


def filter_outliers(df: pd.DataFrame, cols: List[str], upper_pct: float = 0.99, verbose: bool = True) -> pd.DataFrame:
    """Remove rows where any col exceeds its upper_pct quantile.

    Matches the EDA df_model step: rows with extreme trip_distance,
    trip_duration_min, or total_fare_amount are dropped, not clipped.
    """
    n_before = len(df)
    mask = pd.Series(True, index=df.index)
    for col in cols:
        upper = df[col].quantile(upper_pct)
        mask &= df[col] <= upper
    df = df[mask].copy()
    if verbose:
        removed = n_before - len(df)
        print(f"  filter_outliers (p{upper_pct:.0%}): removed {removed:,} rows ({removed/n_before*100:.1f}%)")
    return df


def drop_na_in_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing core input columns (PU/DO zone, distance, datetime)."""
    required = ["PULocationID", "DOLocationID", "trip_distance",
                "tpep_pickup_datetime", "tpep_dropoff_datetime"]
    before = len(df)
    df = df.dropna(subset=required)
    if len(df) < before:
        print(f"  drop_na_in_inputs: dropped {before - len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline for 2024–2025 training data."""
    df = drop_na_in_inputs(df)
    df = add_datetime_features(df)
    df = compute_trip_duration(df)
    df = compute_target(df)
    df = filter_valid_trips(df)
    df = filter_outliers(
        df,
        cols=["trip_distance", "trip_duration_min", TARGET_COL],
        upper_pct=CLEANING["outlier_percentile"],
    )
    return df.reset_index(drop=True)


def clean_test_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning pipeline for 2026 test data (same logic, no train-only assertions)."""
    df = drop_na_in_inputs(df)
    df = add_datetime_features(df)
    df = compute_trip_duration(df)
    df = compute_target(df)
    df = filter_valid_trips(df)
    return df.reset_index(drop=True)
