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


# ---------------------------------------------------------------------------
# ExperimentTracker — eager init style (matches Lecture 4 API)
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """W&B run that starts immediately on construction (no context manager needed).

    Mirrors the Lecture 4 API:
        tracker = ExperimentTracker(project, run_name, tags, config)
        tracker.log_summary({...})
        url = tracker.finish()
    """

    def __init__(
        self,
        project: str = WANDB_PROJECT,
        run_name: str = "run",
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled and _WANDB_AVAILABLE
        self._run = None
        if self.enabled:
            self._run = wandb.init(
                project=project,
                name=run_name,
                tags=tags or [],
                config=config or {},
                reinit=True,
            )

    def log_summary(self, metrics: Dict[str, Any]) -> None:
        if self.enabled and self._run is not None:
            for k, v in metrics.items():
                self._run.summary[k] = v

    def log(self, metrics: Dict[str, Any]) -> None:
        if self.enabled:
            wandb.log(metrics)

    def log_table(self, df: pd.DataFrame, table_name: str) -> None:
        if self.enabled:
            wandb.log({table_name: wandb.Table(dataframe=df.reset_index(drop=True))})

    def log_plot(self, fig: Any, name: str) -> None:
        if self.enabled:
            wandb.log({name: fig})

    def log_image_file(self, path: Path, name: str) -> None:
        if self.enabled and Path(path).exists():
            wandb.log({name: wandb.Image(str(path))})

    def log_artifact(
        self,
        path: Path,
        artifact_name: str,
        artifact_type: str = "dataset",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        art = wandb.Artifact(artifact_name, type=artifact_type, metadata=metadata or {})
        art.add_file(str(path))
        wandb.log_artifact(art)

    def log_code(self) -> None:
        if self.enabled and self._run is not None:
            self._run.log_code()

    def alert(self, title: str, text: str) -> None:
        if self.enabled and self._run is not None:
            self._run.alert(title=title, text=text)

    def finish(self) -> str:
        if self.enabled and self._run is not None:
            url = self._run.get_url() or ""
            wandb.finish()
            return url
        return ""


# ---------------------------------------------------------------------------
# W&B artifact versioning helpers (standalone functions)
# ---------------------------------------------------------------------------

def log_data_artifact(
    tracker: ExperimentTracker,
    path: Path,
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a data file (parquet, csv) as a versioned W&B artifact."""
    tracker.log_artifact(path, artifact_name=name, artifact_type="dataset", metadata=metadata)


def log_model_artifact(
    tracker: ExperimentTracker,
    path: Path,
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a serialised model file as a versioned W&B artifact."""
    tracker.log_artifact(path, artifact_name=name, artifact_type="model", metadata=metadata)


def log_feature_artifact(
    tracker: ExperimentTracker,
    path: Path,
    active_feature_steps: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a feature pipeline artifact (scaler/engineer pkl) to W&B."""
    meta = dict(metadata or {})
    if active_feature_steps is not None:
        meta["feature_columns"] = active_feature_steps
    tracker.log_artifact(path, artifact_name="feature-pipeline", artifact_type="pipeline", metadata=meta)


# ---------------------------------------------------------------------------
# Per-month drift run logger
# ---------------------------------------------------------------------------

def log_monthly_drift_run(
    month_label: str,
    month_num: int,
    mae: float,
    drift_report: Dict[str, Any],
    project: str = WANDB_PROJECT,
    mae_delta: float = 0.0,
    n_trips: int = 0,
    label_drift: Optional[Dict[str, Any]] = None,
) -> None:
    """Create a W&B run for a single month's drift metrics (enables x=month_num line charts)."""
    if not _WANDB_AVAILABLE:
        return

    tracker = ExperimentTracker(
        project=project,
        run_name=f"drift-{month_label}",
        tags=["drift-monthly", month_label],
        config={"month": month_label, "month_num": month_num, "n_trips": n_trips},
    )
    summary: Dict[str, Any] = {
        "month_num": month_num,
        "mae": mae,
        "mae_delta": mae_delta,
    }
    if label_drift:
        for k, v in label_drift.items():
            summary[f"label_{k}"] = v

    feat_drift = drift_report.get("feature_drift")
    if feat_drift is not None and not feat_drift.empty:
        summary["n_significant_feature_drift"] = int(
            (feat_drift["drift_level"] == "significant").sum()
        )
        tracker.log_table(feat_drift, "feature_drift")

    tracker.log_summary(summary)
    tracker.finish()
