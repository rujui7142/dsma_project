"""Weights & Biases hyperparameter sweep for all model types.

Two-phase protocol (per lecture 3 best practice: random search for cheap
stochastic exploration, then Bayesian optimization "informed" by that
exploration's best region — not started cold):

  Phase 1 — RANDOM   : count_random trials over the full parameter space.
  Phase 2 — BAYESIAN : count_bayes  trials over a NARROWED space, centered on
                        phase 1's best result (read back via the W&B API).

Every model (lgbm, xgb, rf, ridge) goes through both phases; ridge's alpha is
a continuous log-uniform range (not a fixed grid) so it benefits the same way.

Run:
    python -m src.sweep --model lgbm   [--count-random 10] [--count-bayes 10]
    python -m src.sweep --model xgb
    python -m src.sweep --model rf
    python -m src.sweep --model ridge
    python -m src.sweep --model all    # runs both phases for all four sequentially

Each agent run:
  1. Shares a pre-loaded feature-engineered train/val split (built once)
  2. Trains the target model with W&B-suggested hyperparameters
  3. Logs val_rmse, val_mae, val_r2, val_mape back to W&B

After the sweep, visit the W&B UI to find the best hyperparameters, then
copy them into src/config.py MODEL_DEFAULTS and re-run train.py.
"""

import argparse
import copy
import math
import sys
from typing import Any, Dict, Optional

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

# Objective the sweep optimizes and reads back "best" by. Must match the
# "metric.name" set on each *_SWEEP_CONFIG above (both drive W&B's Bayesian
# search internally and the phase-1 readback in _best_run_config).
SWEEP_METRIC = "val_mae"

# How much of the original range the Bayesian phase's narrowed window keeps,
# as a fraction of the full [min, max] width, centered on phase 1's best value.
# 0.4 = a window 40% as wide as the original, clipped back to the original
# bounds so phase 2 can't wander outside what was ever a valid value.
NARROW_WINDOW_FRAC = 0.4

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
    p.add_argument("--count-random", type=int, default=10,
                    help="Phase 1 (random search) trial count")
    p.add_argument("--count-bayes", type=int, default=10,
                    help="Phase 2 (Bayesian, narrowed around phase-1 best) trial count")
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


def _best_run_config(tracker: WandbTracker, sweep_id: str) -> Optional[Dict[str, Any]]:
    """Read back the best trial's hyperparameters from a completed sweep."""
    try:
        api = wandb.Api()
        path = f"{tracker.entity}/{tracker.project}/{sweep_id}" if tracker.entity \
            else f"{tracker.project}/{sweep_id}"
        sweep = api.sweep(path)
        runs = [r for r in sweep.runs if r.summary.get(SWEEP_METRIC) is not None]
        if not runs:
            return None
        best = min(runs, key=lambda r: r.summary[SWEEP_METRIC])
        print(f"  Phase-1 best: {SWEEP_METRIC}={best.summary[SWEEP_METRIC]:.4f}  config={dict(best.config)}")
        return dict(best.config)
    except Exception as exc:
        print(f"  WARNING: could not read back phase-1 best run ({exc}); "
              f"phase 2 will use the full (un-narrowed) search space.")
        return None


def _narrow_param_space(
    parameters: Dict[str, Any],
    best_config: Optional[Dict[str, Any]],
    window_frac: float = NARROW_WINDOW_FRAC,
) -> Dict[str, Any]:
    """Center a tighter search window on phase-1's best value for each
    continuous (min/max) parameter, clipped to the original bounds.

    Discrete parameters (a fixed "values" list) are left unchanged — Bayesian
    search already re-weights categorical choices using trial history, and
    arbitrarily dropping options risks excluding the true optimum after only
    a handful of random-search samples.
    """
    if not best_config:
        return copy.deepcopy(parameters)

    narrowed = copy.deepcopy(parameters)
    for name, spec in narrowed.items():
        if "min" not in spec or "max" not in spec or name not in best_config:
            continue  # discrete "values" param, or not swept -> leave as-is

        lo, hi = spec["min"], spec["max"]
        best = best_config[name]
        is_log = spec.get("distribution") == "log_uniform_values"

        if is_log:
            # Narrow in LOG space — a linear window around `best` on a range
            # spanning multiple orders of magnitude (e.g. ridge alpha:
            # 0.001-1000) is wildly asymmetric and barely narrows the upper
            # bound at all. Guard against best/lo/hi <= 0 (shouldn't happen
            # for a log-uniform param, but be defensive).
            if best <= 0 or lo <= 0 or hi <= 0:
                continue
            log_lo, log_hi, log_best = math.log(lo), math.log(hi), math.log(best)
            half_width = (log_hi - log_lo) * window_frac / 2.0
            new_lo = math.exp(max(log_lo, log_best - half_width))
            new_hi = math.exp(min(log_hi, log_best + half_width))
        else:
            half_width = (hi - lo) * window_frac / 2.0
            new_lo = max(lo, best - half_width)
            new_hi = min(hi, best + half_width)

        if new_hi <= new_lo:  # degenerate guard (e.g. best sits exactly on a bound)
            new_lo, new_hi = lo, hi

        # Preserve the ORIGINAL bounds' type. W&B infers int_uniform vs
        # float_uniform from whether min/max are int or float; narrowing via
        # arithmetic silently turns an int-typed param (e.g. max_depth:
        # {"min": 4, "max": 12}) into floats, which W&B's validator then
        # rejects as "ambiguous" and crashes the whole sweep.
        if isinstance(lo, int) and isinstance(hi, int):
            new_lo, new_hi = int(round(new_lo)), int(round(new_hi))
            if new_hi <= new_lo:  # re-check after rounding could collapse the window
                new_lo, new_hi = lo, hi

        spec["min"], spec["max"] = new_lo, new_hi

    return narrowed


def _run_two_phase_sweep(
    tracker: WandbTracker, model_name: str, count_random: int, count_bayes: int, tag: str,
) -> None:
    global _MODEL_NAME
    _MODEL_NAME = model_name
    base_cfg = _SWEEP_CONFIGS[model_name]

    # ---- Phase 1: random search (cheap, stochastic exploration) ----
    phase1_cfg = {**base_cfg, "method": "random", "name": f"{model_name}-random-{tag}"}
    print(f"\n[{model_name.upper()}] Phase 1/2 -- random search, {count_random} trials ...")
    sweep_id_1 = tracker.create_sweep(phase1_cfg, count=count_random)
    tracker.run_sweep_agent(sweep_id_1, _sweep_agent_fn, count=count_random)

    # ---- Phase 2: Bayesian, informed by phase 1's best region ----
    best_config = _best_run_config(tracker, sweep_id_1)
    narrowed_params = _narrow_param_space(base_cfg["parameters"], best_config)
    phase2_cfg = {**base_cfg, "parameters": narrowed_params,
                  "method": "bayes", "name": f"{model_name}-bayes-{tag}"}
    print(f"\n[{model_name.upper()}] Phase 2/2 -- Bayesian search (narrowed around "
          f"phase-1 best), {count_bayes} trials ...")
    sweep_id_2 = tracker.create_sweep(phase2_cfg, count=count_bayes)
    tracker.run_sweep_agent(sweep_id_2, _sweep_agent_fn, count=count_bayes)

    final_best = _best_run_config(tracker, sweep_id_2) or best_config
    print(f"[{model_name.upper()}] Two-phase sweep complete. "
          f"Best config: {final_best}")


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
    # Run two-phase sweep(s)
    # ------------------------------------------------------------------
    tracker = WandbTracker()
    models = ["lgbm", "xgb", "rf", "ridge"] if args.model == "all" else [args.model]

    for model_name in models:
        _run_two_phase_sweep(tracker, model_name, args.count_random, args.count_bayes, args.tag)

    print("\nAll sweeps complete. Best hyperparameters are visible in the W&B UI.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
