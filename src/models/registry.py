"""Save and load trained models and artifacts."""

from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from src.config import MODEL_DIR


# ---------------------------------------------------------------------------
# Local save / load
# ---------------------------------------------------------------------------

def save_artifact(obj: Any, name: str, subdir: Optional[str] = None) -> Path:
    """Persist any Python object via joblib."""
    folder = MODEL_DIR / subdir if subdir else MODEL_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.joblib"
    joblib.dump(obj, path)
    print(f"  Saved: {path}")
    return path


def load_artifact(name: str, subdir: Optional[str] = None) -> Any:
    """Load a previously saved joblib artifact."""
    folder = MODEL_DIR / subdir if subdir else MODEL_DIR
    path = folder / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)


def save_run_artifacts(
    feature_engineer: Any,
    scaler: Any,
    models: Dict[str, Any],
    best_model_name: str,
    run_tag: str = "latest",
) -> None:
    """Save feature engineer, scaler, all models, and mark the best."""
    save_artifact(feature_engineer, "feature_engineer", subdir=run_tag)
    save_artifact(scaler, "scaler", subdir=run_tag)
    for name, model in models.items():
        save_artifact(model, name, subdir=run_tag)
    # Convenience symlink: copy best model as 'best_model'
    best_model = models[best_model_name]
    save_artifact(best_model, "best_model", subdir=run_tag)
    print(f"  Best model ({best_model_name}) also saved as 'best_model'.")


def load_inference_artifacts(run_tag: str = "latest"):
    """Load everything needed for inference at serving time."""
    engineer = load_artifact("feature_engineer", subdir=run_tag)
    scaler = load_artifact("scaler", subdir=run_tag)
    model = load_artifact("best_model", subdir=run_tag)
    return engineer, scaler, model


# ---------------------------------------------------------------------------
# W&B artifact helpers (called from wandb_tracker to avoid circular imports)
# ---------------------------------------------------------------------------

def get_artifact_paths(run_tag: str = "latest") -> Dict[str, Path]:
    folder = MODEL_DIR / run_tag
    return {
        "feature_engineer": folder / "feature_engineer.joblib",
        "scaler": folder / "scaler.joblib",
        "best_model": folder / "best_model.joblib",
        "lgbm": folder / "lgbm.joblib",
        "xgb": folder / "xgb.joblib",
        "rf": folder / "rf.joblib",
        "ridge": folder / "ridge.joblib",
    }
