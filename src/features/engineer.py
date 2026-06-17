"""Feature engineering pipeline.

FeatureEngineer is a scikit-learn compatible transformer that:
  1. Applies all unsupervised transforms (time, distance, zone, domain rules).
  2. On fit(), learns zone-level mean-fare target encoding from training labels.
  3. On transform(), applies learned encoding (unseen zones → global mean).

Usage
-----
engineer = FeatureEngineer(zones_df)
engineer.fit(X_train_raw, y_train)          # learn target encoding
X_train_feat = engineer.transform(X_train_raw)
X_test_feat  = engineer.transform(X_test_raw)
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import RAW_INPUT_COLS, TARGET_COL
from src.features.domain import (
    add_zone_features,
    add_airport_features,
    add_congestion_surcharge,
    add_cbd_fee,
    add_time_surcharges,
    add_estimated_charges_total,
)


# ---------------------------------------------------------------------------
# Feature column lists (ordered for readability)
# ---------------------------------------------------------------------------

# Numeric features fed to all model types
NUMERIC_FEATURES: List[str] = [
    # --- distance ---
    "trip_distance",
    "log_distance",
    "distance_sq",
    # --- time (raw) ---
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    # --- time (cyclic) ---
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    # --- time (binary) ---
    "is_weekend",
    "is_rush_hour",
    "is_overnight",
    # --- airport ---
    "is_jfk_pu",
    "is_lga_pu",
    "is_jfk_do",
    "is_lga_do",
    "is_airport_pickup",
    "is_airport_route",
    "airport_fee_est",
    # --- zone type ---
    "is_yellow_zone_pu",
    "is_yellow_zone_do",
    "is_manhattan_pu",
    "is_manhattan_do",
    "is_cross_borough",
    # --- surcharge estimates ---
    "congestion_surcharge_est",
    "cbd_fee_est",
    "is_post_cbd",
    "extra_est",
    "mta_tax_est",
    "improvement_surcharge_est",
    "estimated_surcharges",
    # --- interactions ---
    "distance_x_airport",
    "distance_x_rush",
    # --- target encoding ---
    "pu_zone_mean_fare",
    "do_zone_mean_fare",
    # --- encoded booleans from zone lookup ---
    "pu_borough_enc",
    "do_borough_enc",
    "pu_service_zone_enc",
    "do_service_zone_enc",
]

# Integer zone IDs treated as categoricals by tree models
CATEGORICAL_ID_FEATURES: List[str] = [
    "PULocationID",
    "DOLocationID",
]

# All features used for tree-based models (LightGBM, XGBoost, RF)
TREE_FEATURES: List[str] = CATEGORICAL_ID_FEATURES + NUMERIC_FEATURES

# Categorical feature names to pass to LightGBM's categorical_feature param
LGBM_CAT_FEATURES: List[str] = CATEGORICAL_ID_FEATURES + [
    "pu_borough_enc",
    "do_borough_enc",
    "pu_service_zone_enc",
    "do_service_zone_enc",
]


# ---------------------------------------------------------------------------
# Pure functions for unsupervised transforms
# ---------------------------------------------------------------------------

def _add_cyclic_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["pickup_dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["pickup_dayofweek"] / 7)
    df["is_weekend"] = (df["pickup_dayofweek"] >= 5).astype(np.int8)
    return df


def _add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_distance"] = np.log1p(df["trip_distance"])
    df["distance_sq"] = df["trip_distance"] ** 2
    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distance_x_airport"] = df["trip_distance"] * df["is_airport_route"]
    df["distance_x_rush"] = df["trip_distance"] * df["is_rush_hour"]
    return df


# ---------------------------------------------------------------------------
# FeatureEngineer (sklearn-compatible)
# ---------------------------------------------------------------------------

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """End-to-end feature engineering transformer.

    Parameters
    ----------
    zones_df : pd.DataFrame
        Taxi zone lookup table (LocationID, Borough, Zone, service_zone).
    """

    def __init__(self, zones_df: pd.DataFrame):
        self.zones_df = zones_df
        # Fitted attributes (set during fit)
        self._pu_means: Optional[pd.Series] = None
        self._do_means: Optional[pd.Series] = None
        self._global_mean: float = 0.0

    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """Fit target encoding on training data.

        Parameters
        ----------
        X : raw input DataFrame (must contain PULocationID, DOLocationID).
        y : training target (total_fare_amount). If None, encoding defaults to 0.
        """
        if y is not None:
            tmp = X[["PULocationID", "DOLocationID"]].copy()
            tmp["_y"] = y.values if hasattr(y, "values") else y
            self._pu_means = tmp.groupby("PULocationID")["_y"].mean()
            self._do_means = tmp.groupby("DOLocationID")["_y"].mean()
            self._global_mean = float(tmp["_y"].mean())
        return self

    # ------------------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps and return the enriched DataFrame."""
        df = X.copy()

        # 1. Unsupervised transforms
        df = _add_cyclic_time(df)
        df = _add_distance_features(df)
        df = add_zone_features(df, self.zones_df)
        df = add_airport_features(df)
        df = add_congestion_surcharge(df)
        df = add_cbd_fee(df)
        df = add_time_surcharges(df)
        df = add_estimated_charges_total(df)
        df = _add_interaction_features(df)

        # 2. Target encoding (zone-level mean fare)
        gm = self._global_mean
        if self._pu_means is not None:
            df["pu_zone_mean_fare"] = df["PULocationID"].map(self._pu_means).fillna(gm)
            df["do_zone_mean_fare"] = df["DOLocationID"].map(self._do_means).fillna(gm)
        else:
            df["pu_zone_mean_fare"] = gm
            df["do_zone_mean_fare"] = gm

        return df

    def get_tree_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and order columns for tree-based models."""
        cols = [c for c in TREE_FEATURES if c in df.columns]
        return df[cols]

    def get_feature_names(self) -> List[str]:
        return list(TREE_FEATURES)


def get_raw_input_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the inference-time available columns from a cleaned DataFrame."""
    cols = [c for c in RAW_INPUT_COLS if c in df.columns]
    return df[cols].copy()
