"""Aggregate daily-fare forecasting with Prophet.

REFRAMED from an earlier per-trip attempt: Prophet's trend + seasonality +
LINEAR extra-regressor formulation badly underperformed a trivial constant-
mean baseline when treated as a per-trip fare regressor (val MAE 14.44 vs a
naive "always predict the training mean" baseline's 10.78, on ~99k trips with
8 of our strongest features as extra regressors). Not a tuning problem -- a
representational mismatch: per-trip fare is driven mostly by WHICH ZONE PAIR
(high-cardinality, highly nonlinear), which a handful of linear coefficients
on top of a smooth trend/seasonality curve can't represent, and forcing many
different fares onto the same smooth curve (many trips share similar
timestamps) actively hurt relative to just predicting the mean.

Reframed to what Prophet is actually built for: forecasting a smooth
AGGREGATE time series with trend + yearly/weekly seasonality + holiday
effects. Here: the mean fare per calendar day.

Pipeline:
  1. Aggregate cleaned training data to one row per calendar day (mean fare).
  2. Two-phase W&B sweep (random -> Bayesian) over Prophet's hyperparameters,
     validated on the held-out Nov-Dec 2025 daily aggregates (same
     VAL_YEARS_MONTHS convention as the rest of the project). Reuses
     sweep.py's _best_run_config / _narrow_param_space -- same two-phase
     narrowing logic, no need to duplicate it.
  3. Refit on the full 2024-2025 daily history with the tuned hyperparameters.
  4. Forecast the real, held-out 2026 test period and compare against the
     actual observed daily mean fare -- genuine test-set numbers.

Holidays are pulled from features.holidays' own calendar (federal + Christian/
Jewish/Muslim/other-cultural, computed via real calendar arithmetic) rather
than Prophet's built-in country holidays, so the two parts of this project
agree on what a "holiday" is.

Run:
    python -m src.prophet_forecast sweep [--count-random 10] [--count-bayes 10]
    python -m src.prophet_forecast test
"""

import argparse
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    DATA_PATHS, TARGET_COL, VAL_YEARS_MONTHS, SAMPLE_CONFIG, PROPHET_DEFAULTS,
)
from src.data.loader import load_parquet_files
from src.data.cleaner import clean_training_data, clean_test_data
from src.features.holidays import _HOLIDAY_SETS
from src.tracking.wandb_tracker import WandbTracker, PROPHET_SWEEP_CONFIG
from src.sweep import _best_run_config, _narrow_param_space

try:
    from prophet import Prophet
    _PROPHET_AVAILABLE = True
except ImportError:
    _PROPHET_AVAILABLE = False

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

_HOLIDAY_CATEGORIES = ("federal", "christian", "jewish", "muslim", "other_cultural")

# Module-level globals shared with the W&B sweep agent callback
_DAILY_TRAIN: Optional[pd.DataFrame] = None
_DAILY_VAL: Optional[pd.DataFrame] = None
_HOLIDAYS_DF: Optional[pd.DataFrame] = None


def build_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse cleaned trip-level data to one row per calendar day: mean
    fare (TARGET_COL) plus trip count for context. Requires pickup_year/
    month/day (added by cleaner.add_datetime_features).
    """
    dates = pd.to_datetime(
        {"year": df["pickup_year"], "month": df["pickup_month"], "day": df["pickup_day"]}
    )
    tmp = pd.DataFrame({"ds": dates, "y": df[TARGET_COL].values})
    daily = tmp.groupby("ds", as_index=False).agg(y=("y", "mean"), n_trips=("y", "size"))
    return daily.sort_values("ds").reset_index(drop=True)


def build_prophet_holidays() -> pd.DataFrame:
    """Prophet-format holidays dataframe (columns: holiday, ds), reusing our
    own holiday calendar (features.holidays) instead of Prophet's built-in
    country holidays -- so both parts of the project agree on what counts as
    a holiday. Each category gets its own name so Prophet can fit a distinct
    effect size per category (Christmas likely suppresses demand very
    differently than, say, a Muslim or Jewish observance).
    """
    rows = [
        {"holiday": category, "ds": pd.Timestamp(y, m, d)}
        for category in _HOLIDAY_CATEGORIES
        for (y, m, d) in _HOLIDAY_SETS[category]
    ]
    return pd.DataFrame(rows)


def split_train_val(daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Same VAL_YEARS_MONTHS convention as the rest of the project (Nov+Dec
    2025 held out), applied to the daily-aggregated series.
    """
    year_month = list(zip(daily["ds"].dt.year, daily["ds"].dt.month))
    is_val = pd.Series([ym in VAL_YEARS_MONTHS for ym in year_month], index=daily.index)
    return daily[~is_val].reset_index(drop=True), daily[is_val].reset_index(drop=True)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def fit_and_forecast(
    train_daily: pd.DataFrame,
    eval_daily: pd.DataFrame,
    params: Dict[str, Any],
    holidays_df: pd.DataFrame,
) -> Tuple["Prophet", Dict[str, float]]:
    """Fit Prophet on train_daily, forecast eval_daily's dates, return
    (fitted model, regression metrics against eval_daily's actual y).
    """
    model = Prophet(
        changepoint_prior_scale=params.get("changepoint_prior_scale", PROPHET_DEFAULTS["changepoint_prior_scale"]),
        seasonality_prior_scale=params.get("seasonality_prior_scale", PROPHET_DEFAULTS["seasonality_prior_scale"]),
        holidays_prior_scale=params.get("holidays_prior_scale", PROPHET_DEFAULTS["holidays_prior_scale"]),
        changepoint_range=params.get("changepoint_range", PROPHET_DEFAULTS["changepoint_range"]),
        seasonality_mode=params.get("seasonality_mode", PROPHET_DEFAULTS["seasonality_mode"]),
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,  # one row per day -- no sub-day granularity to model
        holidays=holidays_df,
    )
    model.fit(train_daily[["ds", "y"]])
    forecast = model.predict(eval_daily[["ds"]])
    metrics = _regression_metrics(eval_daily["y"].values, forecast["yhat"].values)
    return model, metrics


def _sweep_agent_fn():
    """Called by wandb.agent() for each trial."""
    if not _WANDB_AVAILABLE:
        raise RuntimeError("wandb not available")
    with wandb.init() as run:
        cfg = dict(wandb.config)
        _, metrics = fit_and_forecast(_DAILY_TRAIN, _DAILY_VAL, cfg, _HOLIDAYS_DF)
        wandb.log({
            "val_mae": metrics["mae"], "val_rmse": metrics["rmse"],
            "val_r2": metrics["r2"], "val_mape": metrics["mape"],
        })


def run_two_phase_sweep(tracker: WandbTracker, count_random: int, count_bayes: int, tag: str) -> Dict[str, Any]:
    base_cfg = PROPHET_SWEEP_CONFIG

    phase1_cfg = {**base_cfg, "method": "random", "name": f"prophet-random-{tag}"}
    print(f"\n[PROPHET] Phase 1/2 -- random search, {count_random} trials ...")
    sweep_id_1 = tracker.create_sweep(phase1_cfg, count=count_random)
    tracker.run_sweep_agent(sweep_id_1, _sweep_agent_fn, count=count_random)

    best_config = _best_run_config(tracker, sweep_id_1)
    narrowed_params = _narrow_param_space(base_cfg["parameters"], best_config)
    phase2_cfg = {**base_cfg, "parameters": narrowed_params, "method": "bayes", "name": f"prophet-bayes-{tag}"}
    print(f"\n[PROPHET] Phase 2/2 -- Bayesian search (narrowed around phase-1 best), {count_bayes} trials ...")
    sweep_id_2 = tracker.create_sweep(phase2_cfg, count=count_bayes)
    tracker.run_sweep_agent(sweep_id_2, _sweep_agent_fn, count=count_bayes)

    final_best = _best_run_config(tracker, sweep_id_2) or best_config
    print(f"\n[PROPHET] Two-phase sweep complete. Best config: {final_best}")
    return final_best


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["sweep", "test"])
    p.add_argument("--count-random", type=int, default=10)
    p.add_argument("--count-bayes", type=int, default=10)
    p.add_argument("--tag", type=str, default="prophet")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main():
    global _DAILY_TRAIN, _DAILY_VAL, _HOLIDAYS_DF

    if not _PROPHET_AVAILABLE:
        print("ERROR: prophet is not installed. Run: pip install prophet")
        return 1

    args = parse_args()

    print("\n=== Loading + cleaning training data (2024-2025) ===")
    raw_df = load_parquet_files(DATA_PATHS["training"], n_per_file=SAMPLE_CONFIG["n_per_month_train"])
    clean_df = clean_training_data(raw_df)
    daily = build_daily_series(clean_df)
    train_daily, val_daily = split_train_val(daily)
    print(f"  Daily series: {len(daily)} days total ({len(train_daily)} train, {len(val_daily)} val)")

    _HOLIDAYS_DF = build_prophet_holidays()

    if args.command == "sweep":
        _DAILY_TRAIN, _DAILY_VAL = train_daily, val_daily
        tracker = WandbTracker(enabled=not args.no_wandb)
        best_config = run_two_phase_sweep(tracker, args.count_random, args.count_bayes, args.tag)
        print(f"\nAdopt these into config.PROPHET_DEFAULTS:\n{best_config}")

    elif args.command == "test":
        print("\n=== Loading real 2026 test data ===")
        test_raw = load_parquet_files(DATA_PATHS["test"], n_per_file=SAMPLE_CONFIG["n_per_month_test"])
        test_clean = clean_test_data(test_raw)
        test_daily = build_daily_series(test_clean)
        print(f"  Test daily series: {len(test_daily)} days")

        print("\n=== Refitting Prophet on full 2024-2025 history with tuned hyperparameters ===")
        _, metrics = fit_and_forecast(daily, test_daily, PROPHET_DEFAULTS, _HOLIDAYS_DF)

        print("\n=== Real 2026 test results (daily mean-fare forecast) ===")
        print(f"  Test RMSE: {metrics['rmse']:.4f}")
        print(f"  Test MAE:  {metrics['mae']:.4f}")
        print(f"  Test R2:   {metrics['r2']:.4f}")
        print(f"  Test MAPE: {metrics['mape']:.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
