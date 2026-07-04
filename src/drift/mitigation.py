"""Drift mitigation strategies.

Three strategies:
  reweight_retrain  — combine old + recent data, upweight recent rows, retrain
  drop_features     — remove drifted features, retrain on combined data
  recalibrate       — post-hoc bias correction (no retraining)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

RECENT_WEIGHT = 3.0


def _simple_fit(model_name: str, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> Any:
    """Fit a model on combined old+recent data for retraining.

    For the boosted models we early-stop against a chronological tail of the
    combined data (the most-recent rows — which is exactly the period we want
    the retrained model to generalise to). Without this the retrain runs the
    full 1000 boosting rounds and overfits, so mitigation can end up *worse*
    than the frozen model.
    """
    from src.models.trainer import get_model, cap_rf_max_samples
    import lightgbm as lgb
    import xgboost as xgb

    model = get_model(model_name)

    # Chronological tail validation split (combined data is old rows followed by
    # recent rows, so the tail is the freshest period).
    n = len(X)
    use_es = n > 1000
    if use_es:
        n_val = max(200, int(n * 0.15))
        X_tr, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
        y_tr, y_val = y.iloc[:-n_val], y.iloc[-n_val:]
        w_tr = sample_weight[:-n_val] if sample_weight is not None else None
    else:
        X_tr, y_tr, w_tr = X, y, sample_weight

    if isinstance(model, lgb.LGBMRegressor):
        cat_cols = [c for c in ["PULocationID", "DOLocationID"] if c in X.columns]
        fit_kw: Dict[str, Any] = {"categorical_feature": cat_cols or "auto"}
        if w_tr is not None:
            fit_kw["sample_weight"] = w_tr
        if use_es:
            fit_kw["eval_set"] = [(X_val, y_val)]
            fit_kw["callbacks"] = [
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ]
        model.fit(X_tr, y_tr, **fit_kw)

    elif isinstance(model, xgb.XGBRegressor):
        if use_es:
            model.set_params(early_stopping_rounds=50)
            model.fit(X_tr, y_tr, sample_weight=w_tr,
                      eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_tr, y_tr, sample_weight=w_tr)

    else:
        cap_rf_max_samples(model, len(X))  # no-op for Ridge; guards RandomForest
        kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
        model.fit(X, y, **kw)

    return model


def mitigate(
    strategy: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_recent: pd.DataFrame,
    y_recent: pd.Series,
    model_name: str,
    model_dir: str,
    base_model: Any = None,
    drifted_features: Optional[List[str]] = None,
) -> Tuple[Any, Optional[List[str]]]:
    """Apply a mitigation strategy and save the resulting model.

    Returns
    -------
    (mitigated_model, dropped_features_or_None)
    """
    import joblib

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    if strategy == "none":
        return base_model, None

    if strategy == "reweight_retrain":
        n_old, n_new = len(X_train), len(X_recent)
        print(f"  Combining {n_old:,} old rows (w=1.0) + {n_new:,} recent rows (w={RECENT_WEIGHT})")
        X_comb = pd.concat([X_train, X_recent], ignore_index=True)
        y_comb = pd.concat([y_train, y_recent], ignore_index=True)
        weights = np.concatenate([np.ones(n_old), np.full(n_new, RECENT_WEIGHT)])
        model = _simple_fit(model_name, X_comb, y_comb, sample_weight=weights)
        out = Path(model_dir) / f"{model_name}_reweighted.pkl"
        joblib.dump(model, out)
        print(f"  Saved -> {out}")
        return model, None

    if strategy == "drop_features":
        drop = [f for f in (drifted_features or []) if f in X_train.columns]
        if not drop:
            print("  No drifted features to drop -- falling back to reweight_retrain")
            return mitigate(
                "reweight_retrain", X_train, y_train, X_recent, y_recent,
                model_name, model_dir, base_model, drifted_features,
            )
        print(f"  Dropping drifted features: {drop}")
        X_comb = pd.concat(
            [X_train.drop(columns=drop), X_recent.drop(columns=drop, errors="ignore")],
            ignore_index=True,
        )
        y_comb = pd.concat([y_train, y_recent], ignore_index=True)
        model = _simple_fit(model_name, X_comb, y_comb)
        out = Path(model_dir) / f"{model_name}_drop_features.pkl"
        joblib.dump(model, out)
        print(f"  Saved -> {out}")
        return model, drop

    if strategy == "recalibrate":
        if base_model is None:
            raise ValueError("recalibrate requires base_model")
        y_pred_recent = base_model.predict(X_recent)
        bias = float(np.mean(y_pred_recent - y_recent.values))
        print(f"  Bias correction: {bias:+.4f}")

        class _Recalibrated:
            def __init__(self, m, b):
                self._m, self._b = m, b

            def predict(self, X):
                return self._m.predict(X) - self._b

        return _Recalibrated(base_model, bias), None

    raise ValueError(f"Unknown mitigation strategy: {strategy!r}")


def plot_mitigation_comparison(
    error_dict: Dict[str, np.ndarray],
    output_dir: str = "outputs/plots",
) -> Any:
    """Box plot of absolute errors comparing strategies; also prints a comparison table."""
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    labels = list(error_dict.keys())
    errors = [error_dict[k] for k in labels]

    first_mae = float(np.mean(errors[0]))
    print(f"\n  {'Strategy':<45} {'MAE':>7}  {'vs first':>10}")
    print("  " + "-" * 65)
    for label, err in zip(labels, errors):
        mae = float(np.mean(err))
        ref_str = "(reference)" if label == labels[0] else f"{(mae - first_mae) / first_mae * 100:+.1f}%"
        print(f"  {label:<45} {mae:>7.2f}  {ref_str:>10}")

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(errors, labels=labels, showfliers=False, patch_artist=True)
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]
    for patch, color in zip(bp["boxes"], colors[: len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Absolute Error ($)")
    ax.set_title("Drift Mitigation: Absolute Error Comparison")
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()

    out = Path(output_dir) / "mitigation_comparison.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  Plot saved -> {out}")
    return fig
