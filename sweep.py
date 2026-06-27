"""Weights & Biases hyperparameter sweep for all model types.

Run:
    python sweep.py --model lgbm   [--count 20]  [--sample 50000] [--tag SWEEP_TAG]
    python sweep.py --model xgb
    python sweep.py --model rf
    python sweep.py --model ridge
    python sweep.py --model all    # runs sweeps for all four sequentially

Each agent run:
  1. Shares a pre-loaded feature-engineered train/val split (built once)
  2. Trains the target model with W&B-suggested hyperparameters
  3. Logs val_rmse, val_mae, val_r2, val_mape back to W&B

After the sweep, visit the W&B UI to find the best hyperparameters, then
copy them into src/config.py MODEL_DEFAULTS and re-run train.py.
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
from src.models.trainer import train_model, build_ridge_scaler
from src.tracking.wandb_tracker import (
    WandbTracker,
    LGBM_SWEEP_CONFIG,
    XGB_SWEEP_CONFIG,
    RF_SWEEP_CONFIG,
    RIDGE_SWEEP_CONFIG,
)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

_SWEEP_CONFIGS = {
    "lgbm": LGBM_SWEEP_CONFIG,
    "xgb": XGB_SWEEP_CONFIG,
    "rf": RF_SWEEP_CONFIG,
    "ridge": RIDGE_SWEEP_CONFIG,
}

# Module-level globals shared across all sweep agent calls
_X_TRAIN: Optional[pd.DataFrame] = None
_X_VAL: Optional[pd.DataFrame] = None
_Y_TRAIN: Optional[pd.Series] = None
_Y_VAL: Optional[pd.Series] = None
_MODEL_NAME: str = "lgbm"
_SCALER = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", type=str, default="lgbm",
        choices=["lgbm", "xgb", "rf", "ridge", "all"],
        help="Model to sweep, or 'all' to run all four sequentially",
    )
    p.add_argument("--count", type=int, default=20, help="Trials per model")
    p.add_argument("--sample", type=int, default=50_000, help="Rows per month for sweep data")
    p.add_argument("--tag", type=str, default="sweep")
    return p.parse_args()


def _sweep_agent_fn():
    """Called by wandb.agent() for each trial."""
    if not _WANDB_AVAILABLE:
        raise RuntimeError("wandb not available")

    with wandb.init() as run:
        cfg = dict(wandb.config)
        scaler = _SCALER if _MODEL_NAME == "ridge" else None
        _, metrics = train_model(
            model_name=_MODEL_NAME,
            X_train=_X_TRAIN,
            y_train=_Y_TRAIN,
            X_val=_X_VAL,
            y_val=_Y_VAL,
            params=cfg,
            scaler=scaler,
        )
        wandb.log({
            "val_rmse": metrics["rmse"],
            "val_mae": metrics["mae"],
            "val_r2": metrics["r2"],
            "val_mape": metrics["mape"],
        })


def _run_sweep(tracker: WandbTracker, model_name: str, count: int, tag: str) -> None:
    global _MODEL_NAME
    _MODEL_NAME = model_name
    cfg = {**_SWEEP_CONFIGS[model_name], "name": f"{model_name}-sweep-{tag}"}
    n = count if model_name != "ridge" else min(count, len(_SWEEP_CONFIGS["ridge"]["parameters"]["alpha"]["values"]))
    sweep_id = tracker.create_sweep(cfg, count=n)
    print(f"\n[{model_name.upper()}] Starting {n} sweep trials …")
    tracker.run_sweep_agent(sweep_id, _sweep_agent_fn, count=n)
    print(f"[{model_name.upper()}] Sweep complete — view best params in W&B UI.")


def main():
    global _X_TRAIN, _X_VAL, _Y_TRAIN, _Y_VAL, _SCALER

    if not _WANDB_AVAILABLE:
        print("ERROR: wandb is not installed. Run: pip install wandb")
        return 1

    args = parse_args()

    # ------------------------------------------------------------------
    # Prepare data once — shared across all sweep trials and all models
    # ------------------------------------------------------------------
    print(f"\n=== Preparing sweep data ({args.sample:,} rows/month) ===")
    raw_df = load_parquet_files(DATA_PATHS["training"], n_per_file=args.sample)
    zones_df = load_taxi_zones(DATA_PATHS["taxi_zones"])
    clean_df = clean_training_data(raw_df)

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

    _X_TRAIN = engineer.get_tree_features(engineer.transform(X_train_raw))
    _X_VAL = engineer.get_tree_features(engineer.transform(X_val_raw))
    _Y_TRAIN = y_train
    _Y_VAL = y_val
    _SCALER = build_ridge_scaler(_X_TRAIN)

    print(f"  Train: {len(_X_TRAIN):,}  Val: {len(_X_VAL):,}  Features: {_X_TRAIN.shape[1]}")

    # ------------------------------------------------------------------
    # Run sweep(s)
    # ------------------------------------------------------------------
    tracker = WandbTracker()
    models = ["lgbm", "xgb", "rf", "ridge"] if args.model == "all" else [args.model]

    for model_name in models:
        _run_sweep(tracker, model_name, args.count, args.tag)

    print("\nAll sweeps complete. Best hyperparameters are visible in the W&B UI.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
