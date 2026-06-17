"""Model evaluation: metrics, error analysis, and feature importance."""

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

    analyses = {}
    for col in ["pu_borough", "pickup_hour", "distance_bin", "is_airport_route", "pickup_dayofweek"]:
        if col in meta.columns:
            try:
                analyses[col] = evaluate_by_segment(y_t, y_pred, meta, col)
            except Exception:
                pass

    return analyses


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
