"""Data cleaning and validation for NYC TLC trip records.

All cleaning decisions are grounded in the EDA (01_EDA_Main.ipynb) and
replicated here in a reproducible, modular form.
"""

from typing import List

import numpy as np
import pandas as pd

from src.config import CLEANING, TARGET_COL, TLC_RULES

_AIRPORT_ZONES = {TLC_RULES["jfk_zone_id"], TLC_RULES["lga_zone_id"], TLC_RULES["ewr_zone_id"]}


# ---------------------------------------------------------------------------
# Step-wise transforms (each returns a new DataFrame copy)
# ---------------------------------------------------------------------------

def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive pickup time components needed for feature engineering."""
    df = df.copy()
    dt = pd.to_datetime(df["tpep_pickup_datetime"])
    df["pickup_hour"] = dt.dt.hour
    df["pickup_dayofweek"] = dt.dt.dayofweek  # 0=Monday … 6=Sunday
    df["pickup_day"] = dt.dt.day              # day of month (1-31) — needed for holiday matching
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


def robust_zscore(series: pd.Series) -> pd.Series:
    """Modified z-score via median absolute deviation (MAD) -- robust to
    heavy tails and to the very outliers it's trying to detect, unlike a
    mean/std z-score or a fixed percentile cutoff (which just moves with
    the data rather than flagging genuine anomalies).
    Iglewicz & Hoya (1993): |modified z| > 3.5 flags an outlier.
    """
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        mad = series.std() * 1.253314  # fallback if MAD degenerates (many ties)
    return 0.6745 * (series - median) / mad


def filter_fare_efficiency_outliers(
    df: pd.DataFrame, threshold: float = 3.5, verbose: bool = True
) -> pd.DataFrame:
    """Catch JOINT distance/duration/fare anomalies that filter_valid_trips
    and filter_outliers both miss because they only check each column
    MARGINALLY. E.g. a $92.94 fare for a 0.01-mile, 40-minute trip passes
    every existing bound individually (distance >= 0.01, duration <= 180,
    fare <= 300) but is physically absurd together -- almost certainly a
    GPS/meter recording glitch, not a real trip.

    Computed on fare-per-mile and fare-per-minute (log-transformed, since
    both ratios are heavily right-skewed), filtered via robust MAD z-score.

    Airport trips (JFK/LGA/EWR) are EXCLUDED from this check: they use flat
    or semi-flat fare structures that legitimately don't scale with metered
    distance/time, so the ratio is meaningless for them -- applying this
    filter without the exclusion would wrongly discard real airport trips
    (verified: rows sitting at the prior fare ceiling were 71% JFK/LGA).
    """
    n_before = len(df)
    is_airport = df["PULocationID"].isin(_AIRPORT_ZONES) | df["DOLocationID"].isin(_AIRPORT_ZONES)

    fare_per_mile = df[TARGET_COL] / df["trip_distance"].clip(lower=0.1)
    fare_per_min = df[TARGET_COL] / df["trip_duration_min"].clip(lower=0.5)
    z_fpm = robust_zscore(np.log1p(fare_per_mile))
    z_fpmin = robust_zscore(np.log1p(fare_per_min))

    is_anomaly = (z_fpm.abs() > threshold) | (z_fpmin.abs() > threshold)
    mask = ~(is_anomaly & ~is_airport)  # keep airport trips regardless
    df = df[mask].copy()
    if verbose:
        removed = n_before - len(df)
        print(f"  filter_fare_efficiency_outliers (|z|>{threshold}, non-airport only): "
              f"removed {removed:,} rows ({removed / n_before * 100:.2f}%)")
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


def filter_pickup_date_range(df: pd.DataFrame, min_year: int, verbose: bool = True) -> pd.DataFrame:
    """Drop rows with corrupt pickup years earlier than *min_year*.

    Requires pickup_year (added by add_datetime_features). A small fraction of
    TLC rows carry bad timestamps (e.g. 2007, 2008) that otherwise leak into
    the forward-chaining month buckets.
    """
    before = len(df)
    df = df[df["pickup_year"] >= min_year].copy()
    if verbose and len(df) < before:
        removed = before - len(df)
        print(f"  filter_pickup_date_range (>= {min_year}): removed {removed:,} rows "
              f"({removed / before * 100:.2f}%)")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Base cleaning pipeline shared by train and val/test alike.

    Deliberately does NOT include filter_outliers (the 99th-percentile
    distance/duration/fare trim): unlike the other steps here, which remove
    rows that are provably corrupt (missing inputs, impossible values, bad
    timestamps, physically-impossible fare/distance/duration combinations),
    the percentile filter trims a distributional TAIL that can contain
    genuinely valid rare trips. Applying it uniformly before any train/val
    split would (a) let the model be evaluated on an artificially easier,
    pre-trimmed validation distribution instead of the real tail it'll face
    in production, and (b) compute the percentile threshold from data spanning
    the whole dataset -- including, for early CV folds, months that are
    chronologically in the "future" relative to that fold's training window.
    Callers that do a train/val split (analyze.py, train.py, select_features.py,
    sweep.py) instead call filter_outliers on the TRAIN portion only, per
    split/fold, so the threshold is fit only on that fold's own training data
    and validation is scored against the untrimmed real distribution.
    """
    df = drop_na_in_inputs(df)
    df = add_datetime_features(df)
    df = filter_pickup_date_range(df, CLEANING["min_pickup_year"])
    df = compute_trip_duration(df)
    df = compute_target(df)
    df = filter_valid_trips(df)
    df = filter_fare_efficiency_outliers(df, threshold=CLEANING["fare_efficiency_zscore"])
    return df.reset_index(drop=True)


def clean_test_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning pipeline for the 2026 test set. Identical to clean_training_data:
    the test set is never split into a train/val pair, so there's no "train
    portion" to fit a percentile filter_outliers threshold on -- the real,
    untrimmed test distribution is exactly what we want to score against.
    """
    return clean_training_data(df)
