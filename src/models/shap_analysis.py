"""SHAP (Shapley additive explanations) analysis for tree models.

We use the models' *native* TreeSHAP implementations rather than the standalone
`shap` package:

  - LightGBM : booster.predict(X, pred_contrib=True)
  - XGBoost  : booster.predict(DMatrix(X), pred_contribs=True)

Both return exact TreeSHAP values (one column per feature plus a trailing base
value), identical to shap.TreeExplainer but with no extra dependency. For models
without native SHAP support (RandomForest, Ridge) we fall back to their built-in
importance / coefficients so the analysis degrades gracefully.

The key deliverable is `shap_importance_over_folds`, which tracks how each
feature's mean |SHAP| contribution evolves across the forward-chaining folds
(i.e. over time) — answering "which features are most predictive over time".
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb


# ---------------------------------------------------------------------------
# Per-feature SHAP importance for a single fitted model
# ---------------------------------------------------------------------------

def tree_shap_values(model: Any, X: pd.DataFrame) -> Optional[np.ndarray]:
    """Return the (n_samples, n_features) TreeSHAP matrix for a tree model.

    The trailing base-value column returned by the native APIs is dropped.
    Returns None for models without native TreeSHAP support.
    """
    if isinstance(model, lgb.LGBMRegressor):
        contrib = model.booster_.predict(X, pred_contrib=True)
        return np.asarray(contrib)[:, :-1]  # drop base value

    if isinstance(model, xgb.XGBRegressor):
        booster = model.get_booster()
        dmat = xgb.DMatrix(X, feature_names=list(X.columns))
        contrib = booster.predict(dmat, pred_contribs=True)
        return np.asarray(contrib)[:, :-1]  # drop base value

    return None


def shap_importance(
    model: Any,
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Mean |SHAP| per feature (global importance).

    Falls back to feature_importances_ (RF) or |coef_| (Ridge) when native
    TreeSHAP is unavailable, tagging the source in the 'method' column so the
    caller knows the values are not strictly Shapley.
    """
    names = list(feature_names) if feature_names is not None else list(X.columns)

    shap_mat = tree_shap_values(model, X)
    if shap_mat is not None:
        mean_abs = np.abs(shap_mat).mean(axis=0)
        method = "treeshap"
    elif hasattr(model, "feature_importances_"):
        mean_abs = np.asarray(model.feature_importances_, dtype=float)
        method = "impurity"
    elif hasattr(model, "coef_"):
        mean_abs = np.abs(np.asarray(model.coef_, dtype=float))
        method = "abs_coef"
    else:
        mean_abs = np.zeros(len(names))
        method = "none"

    n = min(len(names), len(mean_abs))
    df = pd.DataFrame({"feature": names[:n], "mean_abs_shap": mean_abs[:n]})
    total = df["mean_abs_shap"].sum()
    df["shap_pct"] = df["mean_abs_shap"] / total * 100 if total > 0 else 0.0
    df["method"] = method
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# SHAP importance across forward-chaining folds (predictive value over time)
# ---------------------------------------------------------------------------

def shap_importance_over_folds(
    fold_importances: Dict[Any, pd.DataFrame],
) -> pd.DataFrame:
    """Combine per-fold shap_importance() outputs into a wide time-series table.

    Parameters
    ----------
    fold_importances : {fold_label -> shap_importance DataFrame}

    Returns
    -------
    DataFrame indexed by feature, one column per fold holding shap_pct, plus
    'mean_pct' and 'std_pct' summary columns, sorted by mean importance.
    """
    series = {}
    for fold_label, imp_df in fold_importances.items():
        series[fold_label] = imp_df.set_index("feature")["shap_pct"]

    wide = pd.DataFrame(series).fillna(0.0)
    wide["mean_pct"] = wide.mean(axis=1)
    wide["std_pct"] = wide[list(series.keys())].std(axis=1)
    return wide.sort_values("mean_pct", ascending=False)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_shap_bar(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    output_dir: str = "outputs/plots",
    filename: str = "shap_importance_bar.png",
    title: str = "SHAP Feature Importance (mean |contribution|)",
) -> Any:
    """Horizontal bar chart of the top-N features by mean |SHAP|."""
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    top = importance_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(top))))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#4e79a7")
    ax.set_xlabel("mean |SHAP| contribution ($)")
    ax.set_title(title)
    plt.tight_layout()

    out = Path(output_dir) / filename
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  SHAP bar chart saved -> {out}")
    return fig


def plot_shap_over_time(
    wide_importance: pd.DataFrame,
    fold_labels: List[Any],
    top_n: int = 8,
    output_dir: str = "outputs/plots",
    filename: str = "shap_importance_over_time.png",
) -> Any:
    """Line chart: how the top-N features' SHAP share evolves across folds."""
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    top = wide_importance.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 5))
    for feature, row in top.iterrows():
        ax.plot(
            [str(f) for f in fold_labels],
            [row[f] for f in fold_labels],
            marker="o", linewidth=2, label=feature,
        )
    ax.set_xlabel("Forward-chaining fold (time →)")
    ax.set_ylabel("SHAP importance share (%)")
    ax.set_title("Feature predictive value over time (TreeSHAP)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    out = Path(output_dir) / filename
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  SHAP-over-time chart saved -> {out}")
    return fig
