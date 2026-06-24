"""Drift detection using Evidently AI.

Covers: dataset drift, label (target) drift, and concept drift (regression quality).
"""

from pathlib import Path
from typing import Any, Dict

import pandas as pd

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, RegressionPreset
    from evidently import ColumnMapping
    _EVIDENTLY_AVAILABLE = True
except ImportError:
    _EVIDENTLY_AVAILABLE = False


def _require_evidently():
    if not _EVIDENTLY_AVAILABLE:
        raise ImportError("evidently is required: pip install 'evidently>=0.6.0'")


def run_evidently_drift_report(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    target_col: str = None,
):
    """Dataset + label drift report. ref_df and cur_df must include the target column."""
    _require_evidently()
    from src.config import TARGET_COL
    col_map = ColumnMapping(target=target_col or TARGET_COL)
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df, column_mapping=col_map)
    return report


def parse_drift_results(report) -> Dict[str, Any]:
    """Extract structured drift results from an Evidently DataDrift + TargetDrift report."""
    _require_evidently()
    data = report.as_dict()
    metrics = data.get("metrics", [])

    overall_drift = False
    n_drifted = 0
    share_drifted = 0.0
    drifted_features: list = []
    target_drift = False
    target_drift_score = 0.0

    for m in metrics:
        mid = str(m.get("metric", ""))
        res = m.get("result", {})

        if "DatasetDriftMetric" in mid:
            overall_drift = res.get("dataset_drift", False)
            n_drifted = res.get("number_of_drifted_columns", 0)
            n_total = res.get("number_of_columns", 1)
            share_drifted = n_drifted / max(n_total, 1)

        if "DataDriftTable" in mid:
            drift_by_col = res.get("drift_by_columns", {})
            drifted_features = [
                col for col, info in drift_by_col.items()
                if info.get("drift_detected", False)
            ]
            if not n_drifted:
                n_drifted = len(drifted_features)
                share_drifted = n_drifted / max(len(drift_by_col), 1)

        if "ColumnDriftMetric" in mid or "TargetDrift" in mid:
            target_drift = res.get("drift_detected", False)
            target_drift_score = res.get("drift_score", 0.0)

    return {
        "overall_drift": overall_drift,
        "n_drifted": n_drifted,
        "share_drifted": share_drifted,
        "drifted_features": drifted_features,
        "target_drift": target_drift,
        "target_drift_score": target_drift_score,
    }


def run_evidently_concept_drift_report(
    ref_perf_df: pd.DataFrame,
    cur_perf_df: pd.DataFrame,
    target_col: str = None,
):
    """Concept drift report using RegressionPreset.

    Both DataFrames must contain the target column and a 'prediction' column.
    """
    _require_evidently()
    from src.config import TARGET_COL
    col_map = ColumnMapping(target=target_col or TARGET_COL, prediction="prediction")
    report = Report(metrics=[RegressionPreset()])
    report.run(reference_data=ref_perf_df, current_data=cur_perf_df, column_mapping=col_map)
    return report


def parse_concept_drift_results(
    report,
    threshold: float = 0.10,
) -> Dict[str, Any]:
    """Extract MAE-based concept drift result from an Evidently RegressionPreset report."""
    _require_evidently()
    data = report.as_dict()
    metrics = data.get("metrics", [])

    ref_mae = 0.0
    cur_mae = 0.0

    for m in metrics:
        res = m.get("result", {})
        cur = res.get("current", {})
        ref = res.get("reference", {})
        if cur.get("mean_abs_error") is not None:
            cur_mae = float(cur["mean_abs_error"])
            ref_mae = float(ref.get("mean_abs_error", ref_mae))
            break

    mae_pct_increase = (cur_mae - ref_mae) / max(ref_mae, 1e-8)
    concept_drift = bool(mae_pct_increase > threshold)

    return {
        "concept_drift_detected": concept_drift,
        "ref_mae": ref_mae,
        "cur_mae": cur_mae,
        "mae_pct_increase": mae_pct_increase,
    }


def select_mitigation_strategy(
    drift_results: Dict[str, Any],
    concept_drift_results: Dict[str, Any],
) -> str:
    """Choose a mitigation strategy from drift detection results.

    Logic:
        - No concept drift  → none
        - Concept drift + specific feature drift (< 20 % MAE increase) → drop_features
        - Concept drift otherwise → reweight_retrain
    """
    if not concept_drift_results["concept_drift_detected"]:
        return "none"

    n_drifted = len(drift_results.get("drifted_features", []))
    pct = concept_drift_results["mae_pct_increase"]

    strategy = "drop_features" if (n_drifted > 0 and pct < 0.20) else "reweight_retrain"
    print(
        f"  Concept drift detected: MAE increased by {pct:.1%}"
        f" -- strategy: {strategy}"
    )
    return strategy
