"""Model evaluation: metrics, error analysis, and feature importance."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from src.models.trainer import compute_metrics


# ---------------------------------------------------------------------------
# Segmented evaluation
# ---------------------------------------------------------------------------

def evaluate_by_segment(
    y_true: pd.Series,
    y_pred: np.ndarray,
    meta_df: pd.DataFrame,
    segment_col: str,
) -> pd.DataFrame:
    """Compute metrics broken down by a categorical segment column.

    Parameters
    ----------
    meta_df : original cleaned DataFrame aligned with y_true (same index).
    segment_col : column name in meta_df used to group rows.
    """
    results = []
    for segment_val, idx in meta_df.groupby(segment_col).groups.items():
        idx_aligned = meta_df.index.get_indexer(idx)
        y_t = y_true.iloc[idx_aligned]
        y_p = y_pred[idx_aligned]
        if len(y_t) < 10:
            continue
        m = compute_metrics(y_t.values, y_p)
        m[segment_col] = segment_val
        m["count"] = len(y_t)
        results.append(m)

    return (
        pd.DataFrame(results)
        .sort_values("rmse", ascending=True)
        .reset_index(drop=True)
    )


def error_analysis(
    y_true: pd.Series,
    y_pred: np.ndarray,
    meta_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Run segmented error analysis across multiple dimensions.

    Returns a dict mapping analysis_name → DataFrame of per-segment metrics.
    """
    meta = meta_df.copy().reset_index(drop=True)
    y_t = y_true.reset_index(drop=True)

    # distance bins
    meta["distance_bin"] = pd.cut(
        meta["trip_distance"],
        bins=[0, 1, 3, 5, 10, 20, 100],
        labels=["0-1mi", "1-3mi", "3-5mi", "5-10mi", "10-20mi", "20+mi"],
    )

    segment_cols = [
        "pu_borough", "pickup_hour", "distance_bin", "pickup_dayofweek",
        "is_airport_route", "is_hotspot_route", "is_west_village_route",
        "crosses_cbd", "fully_within_cbd", "is_major_holiday", "is_holiday",
    ]
    analyses = {}
    for col in segment_cols:
        if col in meta.columns:
            try:
                analyses[col] = evaluate_by_segment(y_t, y_pred, meta, col)
            except Exception:
                pass

    return analyses


# ---------------------------------------------------------------------------
# Cross-validation error aggregation (across forward-chaining folds)
# ---------------------------------------------------------------------------

def aggregate_segment_errors(
    fold_segment_dfs: List[pd.DataFrame],
    segment_col: str,
) -> pd.DataFrame:
    """Aggregate per-fold evaluate_by_segment() outputs into mean/std per segment.

    Parameters
    ----------
    fold_segment_dfs : list of DataFrames (one per fold) from evaluate_by_segment,
                       each containing the segment_col plus rmse / mae / count.

    Returns
    -------
    One row per segment value with rmse/mae mean & std across folds, sorted by
    mae_mean descending (worst segments first — these drive feature ideas).
    """
    valid = [d for d in fold_segment_dfs if d is not None and not d.empty]
    if not valid:
        return pd.DataFrame()

    combined = pd.concat(valid, ignore_index=True)
    agg = (
        combined.groupby(segment_col)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            avg_count=("count", "mean"),
            n_folds=("rmse", "size"),
        )
        .reset_index()
        .fillna({"rmse_std": 0.0, "mae_std": 0.0})
    )
    return agg.sort_values("mae_mean", ascending=False).reset_index(drop=True)


def plot_segment_error(
    agg_df: pd.DataFrame,
    segment_col: str,
    output_dir: str = "outputs/plots",
    filename: Optional[str] = None,
    title: Optional[str] = None,
) -> Any:
    """Bar chart of mean MAE per segment (error bars = std across folds)."""
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if agg_df.empty:
        return None

    labels = agg_df[segment_col].astype(str).tolist()
    fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(labels)), 4.5))
    ax.bar(labels, agg_df["mae_mean"], yerr=agg_df["mae_std"],
           capsize=3, color="#e15759", alpha=0.85)
    ax.set_xlabel(segment_col)
    ax.set_ylabel("MAE ($)  mean ± std across folds")
    ax.set_title(title or f"CV error analysis by {segment_col}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    fname = filename or f"cv_error_by_{segment_col}.png"
    out = Path(output_dir) / fname
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  Segment-error chart saved -> {out}")
    return fig


def plot_metric_over_folds(
    per_model_metric: Dict[str, List[float]],
    fold_labels: List[Any],
    metric_name: str = "MAE",
    output_dir: str = "outputs/plots",
    filename: str = "metric_over_folds.png",
    title: Optional[str] = None,
) -> Any:
    """Line chart of a metric per fold, one line per model (performance drift)."""
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    x = [str(f) for f in fold_labels]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for model_name, values in per_model_metric.items():
        ax.plot(x, values, marker="o", linewidth=2, label=model_name)
    ax.set_xlabel("Forward-chaining fold (time →)")
    ax.set_ylabel(f"{metric_name} ($)")
    ax.set_title(title or f"{metric_name} over forward-chaining folds")
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    out = Path(output_dir) / filename
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  {metric_name}-over-folds chart saved -> {out}")
    return fig


def plot_monthly_temporal_trend(
    monthly_df: pd.DataFrame,
    output_dir: str = "outputs/plots",
    filename: str = "monthly_temporal_trend.png",
    shift_month_label: Optional[str] = None,
) -> Any:
    """Two-panel month-by-month view: raw actual fare trend + champion MAE.

    Parameters
    ----------
    monthly_df : columns [month_label, actual_mean_fare, champion_mae], one row
                 per calendar month, sorted chronologically. actual_mean_fare
                 is a plain data statistic (no model); champion_mae comes from
                 an expanding walk-forward evaluation and may be NaN for the
                 initial training-only months.
    shift_month_label : month label to mark with a vertical line (e.g. the
                 first month affected by a known regime change, like the CBD
                 fee). Drawn on both panels if present in monthly_df.
    """
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    x = monthly_df["month_label"].astype(str).tolist()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax1.plot(x, monthly_df["actual_mean_fare"], marker="o", linewidth=2, color="#4e79a7")
    ax1.set_ylabel("Actual mean fare ($)")
    ax1.set_title("Raw fare trend by calendar month (no model)")

    ax2.plot(x, monthly_df["champion_mae"], marker="o", linewidth=2, color="#e15759")
    ax2.set_ylabel("Champion MAE ($)")
    ax2.set_xlabel("Month (time →)")
    ax2.set_title("Champion model MAE by calendar month (expanding walk-forward)")

    if shift_month_label is not None and shift_month_label in x:
        for ax in (ax1, ax2):
            ax.axvline(x.index(shift_month_label), ls="--", color="grey", alpha=0.7,
                       label=f"{shift_month_label} (regime change)")
        ax1.legend()

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out = Path(output_dir) / filename
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  Monthly temporal trend chart saved -> {out}")
    return fig


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(
    model: Any,
    feature_names: List[str],
    importance_type: str = "gain",
) -> pd.DataFrame:
    """Extract feature importance from tree-based models.

    Parameters
    ----------
    importance_type : 'gain' or 'split' for LightGBM; 'weight'/'gain'/'cover'
                      for XGBoost; 'impurity' for RF (sklearn).
    """
    fi: Optional[np.ndarray] = None

    if isinstance(model, lgb.LGBMRegressor):
        fi = model.booster_.feature_importance(importance_type=importance_type)

    elif isinstance(model, xgb.XGBRegressor):
        imp_map = model.get_booster().get_score(importance_type=importance_type)
        fi = np.array([imp_map.get(f, 0.0) for f in feature_names])

    elif hasattr(model, "feature_importances_"):
        fi = model.feature_importances_

    if fi is None:
        return pd.DataFrame({"feature": feature_names, "importance": [0.0] * len(feature_names)})

    df = pd.DataFrame({"feature": feature_names[: len(fi)], "importance": fi})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["importance_pct"] = df["importance"] / df["importance"].sum() * 100
    return df


# ---------------------------------------------------------------------------
# Residual summary
# ---------------------------------------------------------------------------

def residual_summary(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Return percentile summary of residuals (y_pred − y_true)."""
    residuals = y_pred - y_true.values
    pcts = np.percentile(residuals, [1, 5, 25, 50, 75, 95, 99])
    return {
        "p01": pcts[0], "p05": pcts[1], "p25": pcts[2], "median": pcts[3],
        "p75": pcts[4], "p95": pcts[5], "p99": pcts[6],
        "mean_residual": float(residuals.mean()),
        "std_residual": float(residuals.std()),
    }
