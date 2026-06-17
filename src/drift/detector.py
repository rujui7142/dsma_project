"""Drift detection: data drift (PSI, KS test) and performance drift.

References:
  - Lecture 3: Hyperparameter tuning, error analysis and model drift
  - Lecture 4: Drift mitigation and automation
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Population Stability Index (PSI)
# ---------------------------------------------------------------------------

def compute_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
    epsilon: float = 1e-8,
) -> float:
    """Compute the Population Stability Index between two distributions.

    Interpretation:
        PSI < 0.10  → no significant shift
        0.10–0.25   → moderate shift
        > 0.25      → significant shift
    """
    combined = pd.concat([reference, current])
    breakpoints = np.linspace(combined.min(), combined.max(), bins + 1)
    breakpoints[0] -= 1e-9
    breakpoints[-1] += 1e-9

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = ref_counts / len(reference) + epsilon
    cur_pct = cur_counts / len(current) + epsilon

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


# ---------------------------------------------------------------------------
# KS test
# ---------------------------------------------------------------------------

def compute_ks_test(
    reference: pd.Series,
    current: pd.Series,
) -> Tuple[float, float]:
    """Two-sample KS test. Returns (statistic, p_value)."""
    stat, p_val = stats.ks_2samp(reference.dropna(), current.dropna())
    return float(stat), float(p_val)


# ---------------------------------------------------------------------------
# Feature drift report
# ---------------------------------------------------------------------------

def detect_feature_drift(
    df_ref: pd.DataFrame,
    df_curr: pd.DataFrame,
    feature_cols: List[str],
    psi_bins: int = 10,
) -> pd.DataFrame:
    """Generate a drift report for each feature column.

    Returns a DataFrame with columns:
      feature, psi, ks_stat, ks_pval, drift_level
    """
    records = []
    for col in feature_cols:
        if col not in df_ref.columns or col not in df_curr.columns:
            continue
        ref_col = pd.to_numeric(df_ref[col], errors="coerce").dropna()
        cur_col = pd.to_numeric(df_curr[col], errors="coerce").dropna()
        if len(ref_col) == 0 or len(cur_col) == 0:
            continue

        psi = compute_psi(ref_col, cur_col, bins=psi_bins)
        ks_stat, ks_pval = compute_ks_test(ref_col, cur_col)

        if psi < 0.10:
            level = "stable"
        elif psi < 0.25:
            level = "moderate"
        else:
            level = "significant"

        records.append({
            "feature": col,
            "psi": round(psi, 4),
            "ks_stat": round(ks_stat, 4),
            "ks_pval": round(ks_pval, 4),
            "drift_level": level,
        })

    return pd.DataFrame(records).sort_values("psi", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Performance drift over time
# ---------------------------------------------------------------------------

def detect_performance_drift(
    y_true: pd.Series,
    y_pred: np.ndarray,
    meta_df: pd.DataFrame,
    time_col: str = "pickup_month",
) -> pd.DataFrame:
    """Compute RMSE and MAE per time period to detect concept drift."""
    from src.models.trainer import compute_metrics

    meta = meta_df.copy().reset_index(drop=True)
    y_t = y_true.reset_index(drop=True)
    y_p = pd.Series(y_pred).reset_index(drop=True)

    records = []
    for period, idx in meta.groupby(time_col).groups.items():
        m = compute_metrics(y_t.iloc[idx].values, y_p.iloc[idx].values)
        m[time_col] = period
        m["count"] = len(idx)
        records.append(m)

    return pd.DataFrame(records).sort_values(time_col).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full drift report
# ---------------------------------------------------------------------------

def generate_drift_report(
    df_ref: pd.DataFrame,
    df_curr: pd.DataFrame,
    feature_cols: List[str],
    y_true_ref: Optional[pd.Series] = None,
    y_pred_ref: Optional[np.ndarray] = None,
    y_true_curr: Optional[pd.Series] = None,
    y_pred_curr: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Produce a comprehensive drift report comparing reference vs current data.

    Parameters
    ----------
    df_ref   : training data (e.g. last 2 months of 2025)
    df_curr  : test / production data (e.g. 2026 Jan-Feb)
    """
    from src.models.trainer import compute_metrics

    report: Dict[str, Any] = {}
    report["feature_drift"] = detect_feature_drift(df_ref, df_curr, feature_cols)

    n_significant = (report["feature_drift"]["drift_level"] == "significant").sum()
    report["summary"] = {
        "n_features_checked": len(report["feature_drift"]),
        "n_significant_drift": int(n_significant),
        "n_moderate_drift": int((report["feature_drift"]["drift_level"] == "moderate").sum()),
    }

    if y_true_ref is not None and y_pred_ref is not None:
        report["ref_metrics"] = compute_metrics(y_true_ref.values, y_pred_ref)

    if y_true_curr is not None and y_pred_curr is not None:
        report["curr_metrics"] = compute_metrics(y_true_curr.values, y_pred_curr)

    return report
