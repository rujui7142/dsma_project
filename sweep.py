"""Weights & Biases hyperparameter sweep for LightGBM (and optionally XGBoost).

Run:
    python sweep.py [--model lgbm|xgb] [--count 20] [--sample 50000] [--tag SWEEP_TAG]

Each agent run:
  1. Loads a subset of training data
  2. Applies cleaning + feature engineering (using saved engineer artifact if available,
     or fitting fresh from the sweep data subset)
  3. Trains the specified model with W&B-suggested hyperparameters
  4. Logs val_rmse, val_mae, val_r2 back to W&B
"""

import argparse
import sys
from typing import Optional

import pandas as pd

from src.config import (
    DATA_PATHS, SAMPLE_CONFIG, TARGET_COL, VAL_YEARS_MONTHS, WANDB_PROJECT,
)
from src.data.loader import load_parquet_files, load_taxi_zones
from src.data.cleaner import clean_training_data
from src.features.engineer import FeatureEngineer, get_raw_input_features
from src.models.trainer import train_model, compute_metrics, build_ridge_scaler
from src.tracking.wandb_tracker import WandbTracker, LGBM_SWEEP_CONFIG, XGB_SWEEP_CONFIG

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="lgbm", choices=["lgbm", "xgb"])
    p.add_argument("--count", type=int, default=20, help="Number of sweep trials")
    p.add_argument("--sample", type=int, default=50_000, help="Rows per month for sweep (smaller = faster)")
    p.add_argument("--tag", type=str, default="sweep")
    return p.parse_args()


# Module-level globals used by the sweep agent function
_X_TRAIN: Optional[pd.DataFrame] = None
_X_VAL: Optional[pd.DataFrame] = None
_Y_TRAIN: Optional[pd.Series] = None
_Y_VAL: Optional[pd.Series] = None
_MODEL_NAME: str = "lgbm"


def _sweep_agent_fn():
    """Function called by wandb.agent() for each trial."""
    if not _WANDB_AVAILABLE:
        raise RuntimeError("wandb not available")

    with wandb.init() as run:
        cfg = dict(wandb.config)
        model_name = _MODEL_NAME

        _, metrics = train_model(
            model_name=model_name,
            X_train=_X_TRAIN,
            y_train=_Y_TRAIN,
            X_val=_X_VAL,
            y_val=_Y_VAL,
            params=cfg,
        )

        wandb.log({
            "val_rmse": metrics["rmse"],
            "val_mae": metrics["mae"],
            "val_r2": metrics["r2"],
            "val_mape": metrics["mape"],
        })


def main():
    global _X_TRAIN, _X_VAL, _Y_TRAIN, _Y_VAL, _MODEL_NAME

    if not _WANDB_AVAILABLE:
        print("ERROR: wandb is not installed. Run: pip install wandb")
        return 1

    args = parse_args()
    _MODEL_NAME = args.model

    # ------------------------------------------------------------------
    # Prepare data (shared across all sweep trials)
    # ------------------------------------------------------------------
    print(f"\n=== Preparing sweep data ({args.sample} rows/month) ===")
    raw_df = load_parquet_files(DATA_PATHS["training"], n_per_file=args.sample)
    zones_df = load_taxi_zones(DATA_PATHS["taxi_zones"])
    clean_df = clean_training_data(raw_df)

    # Temporal split (same as train.py)
    is_val = pd.Series(False, index=clean_df.index)
    for yr, mo in VAL_YEARS_MONTHS:
        is_val |= (clean_df["pickup_year"] == yr) & (clean_df["pickup_month"] == mo)

    train_df = clean_df[~is_val].copy()
    val_df = clean_df[is_val].copy()

    X_train_raw = get_raw_input_features(train_df)
    X_val_raw = get_raw_input_features(val_df)
    y_train = train_df[TARGET_COL].reset_index(drop=True)
    y_val = val_df[TARGET_COL].reset_index(drop=True)

    engineer = FeatureEngineer(zones_df)
    engineer.fit(X_train_raw, y_train)

    X_train_eng = engineer.transform(X_train_raw)
    X_val_eng = engineer.transform(X_val_raw)

    _X_TRAIN = engineer.get_tree_features(X_train_eng)
    _X_VAL = engineer.get_tree_features(X_val_eng)
    _Y_TRAIN = y_train
    _Y_VAL = y_val

    print(f"  Train: {len(_X_TRAIN):,}  Val: {len(_X_VAL):,}  Features: {_X_TRAIN.shape[1]}")

    # ------------------------------------------------------------------
    # Create and run sweep
    # ------------------------------------------------------------------
    tracker = WandbTracker()
    sweep_cfg = LGBM_SWEEP_CONFIG if args.model == "lgbm" else XGB_SWEEP_CONFIG
    sweep_cfg["name"] = f"{args.model}-sweep-{args.tag}"

    sweep_id = tracker.create_sweep(sweep_cfg, count=args.count)
    print(f"\nStarting {args.count} sweep trials for {args.model.upper()} …")
    tracker.run_sweep_agent(sweep_id, _sweep_agent_fn, count=args.count)

    print("\nSweep complete. Best hyperparameters are visible in the W&B UI.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
