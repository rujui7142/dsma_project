"""Baseline training: raw input features only, no feature engineering.

Run once locally to establish the pre-EDA performance floor, then compare
against the full engineered pipeline in the W&B dashboard.

    python baseline.py [--sample N] [--no-wandb] [--tag baseline]

Uses only the 7 columns available at booking time (the same 7 our inference
interface accepts) with no target encoding, no zone lookup, no TLC rules.
Models: Ridge (linear) + RandomForest (tree) — minimal hyperparameter choices.
"""

import argparse
import sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_PATHS, SAMPLE_CONFIG, TARGET_COL, VAL_YEARS_MONTHS,
    WANDB_PROJECT, LOGS_DIR,
)
from src.data.cleaner import clean_training_data
from src.data.loader import load_parquet_files, load_taxi_zones
from src.models.trainer import compute_metrics
from src.tracking.wandb_tracker import WandbTracker

RAW_FEATURES = [
    "PULocationID", "DOLocationID", "trip_distance",
    "pickup_hour", "pickup_dayofweek", "pickup_month", "pickup_year",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=SAMPLE_CONFIG["n_per_month_train"])
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--tag", type=str, default="baseline")
    return p.parse_args()


def temporal_split(df: pd.DataFrame):
    is_val = pd.Series(False, index=df.index)
    for yr, mo in VAL_YEARS_MONTHS:
        is_val |= (df["pickup_year"] == yr) & (df["pickup_month"] == mo)
    return df[~is_val].copy(), df[is_val].copy()


def main():
    args = parse_args()
    tracker = WandbTracker(enabled=not args.no_wandb)

    print("\n=== Baseline: loading & cleaning data ===")
    raw_df = load_parquet_files(DATA_PATHS["training"], n_per_file=args.sample)
    clean_df = clean_training_data(raw_df)
    train_df, val_df = temporal_split(clean_df)

    X_train = train_df[RAW_FEATURES].fillna(0).astype(float)
    X_val = val_df[RAW_FEATURES].fillna(0).astype(float)
    y_train = train_df[TARGET_COL].reset_index(drop=True)
    y_val = val_df[TARGET_COL].reset_index(drop=True)
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Raw features: {len(RAW_FEATURES)}")

    results = {}

    print("\n[RIDGE-BASELINE] training ...")
    scaler = StandardScaler().fit(X_train)
    ridge = Ridge(alpha=1.0)
    ridge.fit(scaler.transform(X_train), y_train)
    y_pred = ridge.predict(scaler.transform(X_val))
    results["ridge_baseline"] = compute_metrics(y_val.values, y_pred)
    m = results["ridge_baseline"]
    print(f"  val RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R2={m['r2']:.4f}")

    print("\n[RF-BASELINE] training ...")
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=15, min_samples_leaf=10,
        max_samples=100_000, n_jobs=-1, random_state=42,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    results["rf_baseline"] = compute_metrics(y_val.values, y_pred)
    m = results["rf_baseline"]
    print(f"  val RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R2={m['r2']:.4f}")

    print("\n=== Baseline summary ===")
    summary_df = pd.DataFrame([{"model": k, **v} for k, v in results.items()])
    print(summary_df.to_string(index=False))
    out = LOGS_DIR / f"baseline_{args.tag}.csv"
    summary_df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    with tracker.init_run(
        name=f"baseline-{args.tag}",
        config={
            "features": RAW_FEATURES,
            "n_features": len(RAW_FEATURES),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "sample_per_month": args.sample,
            "note": "raw features only — no target encoding, no zone lookup",
        },
        tags=["baseline", args.tag],
    ):
        for name, m in results.items():
            tracker.log({f"{name}/{k}": v for k, v in m.items()})
        tracker.log_dataframe(summary_df, "baseline_comparison")

    return 0


if __name__ == "__main__":
    sys.exit(main())
