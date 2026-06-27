"""Model training utilities.

Supports four model types:
  - lgbm   : LightGBM (primary model, categorical-aware)
  - xgb    : XGBoost
  - rf     : Random Forest
  - ridge  : Ridge Regression (linear baseline)

Each model is trained via train_model(); train_all_models() iterates all four
and returns a comparison dict.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

from src.config import MODEL_DEFAULTS, TARGET_COL
from src.features.engineer import LGBM_CAT_FEATURES, TREE_FEATURES


# ---------------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_model(model_name: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """Return an untrained model instance for the given *model_name*."""
    defaults = MODEL_DEFAULTS[model_name].copy()
    if params:
        defaults.update(params)
    if model_name == "lgbm":
        return lgb.LGBMRegressor(**defaults)
    if model_name == "xgb":
        return xgb.XGBRegressor(**defaults)
    if model_name == "rf":
        return RandomForestRegressor(**defaults)
    if model_name == "ridge":
        return Ridge(**defaults)
    raise ValueError(f"Unknown model_name: {model_name!r}")


# ---------------------------------------------------------------------------
# LightGBM fit helper (handles v2.x vs v3+ API difference)
# ---------------------------------------------------------------------------

def _fit_lgbm(
    model: lgb.LGBMRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> lgb.LGBMRegressor:
    cat_cols = [c for c in LGBM_CAT_FEATURES if c in X_train.columns]
    lgb_major = int(lgb.__version__.split(".")[0])

    fit_kwargs: Dict[str, Any] = {
        "categorical_feature": cat_cols,
        "eval_set": [(X_val, y_val)],
    }

    if lgb_major >= 3:
        fit_kwargs["callbacks"] = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ]
    else:
        fit_kwargs["early_stopping_rounds"] = 50
        fit_kwargs["verbose"] = 100

    model.fit(X_train, y_train, **fit_kwargs)
    return model


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[Any, Dict[str, float]]:
    """Train a single model and evaluate on the validation split.

    Parameters
    ----------
    model_name : one of 'lgbm', 'xgb', 'rf', 'ridge'.
    X_train / X_val : feature DataFrames (tree features, pre-encoded).
    y_train / y_val : target Series.
    params : optional hyperparameter overrides.
    scaler : pre-fitted StandardScaler for linear models.

    Returns
    -------
    (fitted_model, val_metrics_dict)
    """
    model = get_model(model_name, params)
    print(f"\n[{model_name.upper()}] training ...")

    if model_name == "lgbm":
        model = _fit_lgbm(model, X_train, y_train, X_val, y_val)

    elif model_name == "xgb":
        # In XGBoost 2.x, early_stopping_rounds moved to the constructor.
        # Re-instantiate with it so we support both old and new API.
        xgb_params = MODEL_DEFAULTS["xgb"].copy()
        if params:
            xgb_params.update(params)
        xgb_params["early_stopping_rounds"] = 50
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    elif model_name == "rf":
        model.fit(X_train, y_train)

    elif model_name == "ridge":
        if scaler is None:
            raise ValueError("Ridge requires a fitted StandardScaler.")
        X_tr_sc = scaler.transform(X_train)
        X_vl_sc = scaler.transform(X_val)
        model.fit(X_tr_sc, y_train)
        y_pred = model.predict(X_vl_sc)
        metrics = compute_metrics(y_val.values, y_pred)
        print(f"  val RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R2={metrics['r2']:.4f}")
        return model, metrics

    y_pred = model.predict(X_val)
    metrics = compute_metrics(y_val.values, y_pred)
    print(f"  val RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R2={metrics['r2']:.4f}")
    return model, metrics


# ---------------------------------------------------------------------------
# Train all models at once
# ---------------------------------------------------------------------------

def build_ridge_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on training features (for Ridge only)."""
    return StandardScaler().fit(X_train.values)


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_names: tuple = ("lgbm", "xgb", "rf", "ridge"),
) -> Dict[str, Tuple[Any, Dict[str, float]]]:
    """Train the requested model types and return their results.

    Parameters
    ----------
    model_names : subset of ('lgbm', 'xgb', 'rf', 'ridge') to train.
                  Pass a smaller tuple to resume after a partial crash.
    """
    scaler = build_ridge_scaler(X_train)
    results: Dict[str, Tuple[Any, Dict[str, float]]] = {}

    for name in model_names:
        model, metrics = train_model(
            name, X_train, y_train, X_val, y_val, scaler=scaler
        )
        results[name] = (model, metrics)

    return results, scaler


def select_best_model(
    results: Dict[str, Tuple[Any, Dict[str, float]]],
    metric: str = "rmse",
    higher_is_better: bool = False,
) -> str:
    """Return the name of the best-performing model."""
    scores = {name: m[1][metric] for name, m in results.items()}
    if higher_is_better:
        return max(scores, key=scores.__getitem__)
    return min(scores, key=scores.__getitem__)
