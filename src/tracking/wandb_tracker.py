"""Weights & Biases experiment tracking utilities.

All W&B interactions are centralised here so the rest of the codebase
stays clean if W&B is disabled or unavailable.

Usage:
    tracker = WandbTracker(project="nyc-tlc-fare-prediction")
    with tracker.init_run(name="lgbm-baseline", config={...}):
        tracker.log({"val_rmse": 4.32})
        tracker.log_dataframe(feature_importance_df, "feature_importance")
        tracker.log_model(model_path, "lgbm-v1")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import WANDB_PROJECT, WANDB_ENTITY

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# W&B Sweep configuration for LightGBM hyperparameter search
# ---------------------------------------------------------------------------

LGBM_SWEEP_CONFIG: Dict[str, Any] = {
    "method": "bayes",
    "metric": {"name": "val_rmse", "goal": "minimize"},
    "parameters": {
        "num_leaves": {"min": 31, "max": 255},
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 0.01,
            "max": 0.3,
        },
        "max_depth": {"values": [-1, 5, 7, 9, 12]},
        "min_child_samples": {"min": 20, "max": 200},
        "subsample": {"min": 0.5, "max": 1.0},
        "colsample_bytree": {"min": 0.5, "max": 1.0},
        "reg_alpha": {"min": 0.0, "max": 2.0},
        "reg_lambda": {"min": 0.0, "max": 2.0},
        "n_estimators": {"values": [500, 750, 1000, 1500]},
    },
}

XGB_SWEEP_CONFIG: Dict[str, Any] = {
    "method": "bayes",
    "metric": {"name": "val_rmse", "goal": "minimize"},
    "parameters": {
        "max_depth": {"min": 4, "max": 12},
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 0.01,
            "max": 0.3,
        },
        "subsample": {"min": 0.5, "max": 1.0},
        "colsample_bytree": {"min": 0.5, "max": 1.0},
        "reg_alpha": {"min": 0.0, "max": 2.0},
        "reg_lambda": {"min": 0.0, "max": 5.0},
        "n_estimators": {"values": [500, 750, 1000]},
    },
}


# ---------------------------------------------------------------------------
# Tracker class
# ---------------------------------------------------------------------------

class WandbTracker:
    """Thin wrapper around the wandb SDK.

    If wandb is not installed or the user is not logged in, all methods
    are no-ops so training still runs correctly.
    """

    def __init__(
        self,
        project: str = WANDB_PROJECT,
        entity: Optional[str] = WANDB_ENTITY,
        enabled: bool = True,
    ):
        self.project = project
        self.entity = entity
        self.enabled = enabled and _WANDB_AVAILABLE
        if enabled and not _WANDB_AVAILABLE:
            print("[WandbTracker] wandb not installed – tracking disabled.")

    # ------------------------------------------------------------------

    def init_run(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> Any:
        """Start a new W&B run and return the run object (or a no-op context)."""
        if not self.enabled:
            return _NoopRun()
        return wandb.init(
            project=self.project,
            entity=self.entity,
            name=name,
            config=config or {},
            tags=tags or [],
            notes=notes,
            reinit=True,
        )

    def log(self, metrics: Dict[str, Any]) -> None:
        if self.enabled:
            wandb.log(metrics)

    def log_dataframe(self, df: pd.DataFrame, table_name: str) -> None:
        """Log a pandas DataFrame as a W&B Table."""
        if self.enabled:
            wandb.log({table_name: wandb.Table(dataframe=df)})

    def log_model(self, model_path: Path, artifact_name: str, metadata: Optional[Dict] = None) -> None:
        """Log a saved model file as a W&B artifact."""
        if not self.enabled:
            return
        artifact = wandb.Artifact(artifact_name, type="model", metadata=metadata or {})
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)

    def log_all_models(
        self,
        model_paths: Dict[str, Path],
        run_metrics: Dict[str, Dict[str, float]],
    ) -> None:
        """Log all model artifacts and a comparison table."""
        if not self.enabled:
            return
        rows = []
        for name, metrics in run_metrics.items():
            row = {"model": name, **metrics}
            rows.append(row)
            if name in model_paths and model_paths[name].exists():
                artifact = wandb.Artifact(f"model-{name}", type="model",
                                          metadata={"metrics": metrics})
                artifact.add_file(str(model_paths[name]))
                wandb.log_artifact(artifact)

        comparison_df = pd.DataFrame(rows)
        wandb.log({"model_comparison": wandb.Table(dataframe=comparison_df)})

    def log_drift_report(self, drift_report: Dict[str, Any]) -> None:
        """Log drift detection results."""
        if not self.enabled:
            return
        summary = drift_report.get("summary", {})
        wandb.log({f"drift/{k}": v for k, v in summary.items()})

        feat_drift = drift_report.get("feature_drift")
        if feat_drift is not None:
            wandb.log({"drift/feature_drift": wandb.Table(dataframe=feat_drift)})

        for split in ("ref_metrics", "curr_metrics"):
            if split in drift_report:
                wandb.log({f"drift/{split}/{k}": v for k, v in drift_report[split].items()})

    def finish(self) -> None:
        if self.enabled:
            wandb.finish()

    # ------------------------------------------------------------------
    # Sweep helpers
    # ------------------------------------------------------------------

    def create_sweep(self, sweep_config: Dict, count: int = 20) -> str:
        """Create a W&B sweep and return the sweep_id."""
        if not self.enabled:
            raise RuntimeError("W&B is not enabled.")
        sweep_id = wandb.sweep(
            sweep_config,
            project=self.project,
            entity=self.entity,
        )
        print(f"Sweep created: {sweep_id}  (run with: wandb agent {sweep_id})")
        return sweep_id

    def run_sweep_agent(self, sweep_id: str, train_fn: Any, count: int = 20) -> None:
        if not self.enabled:
            raise RuntimeError("W&B is not enabled.")
        wandb.agent(sweep_id, function=train_fn, count=count)


# ---------------------------------------------------------------------------
# No-op context manager (when W&B is disabled)
# ---------------------------------------------------------------------------

class _NoopRun:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
