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

import src.config as _config
from src.config import (
    RAW_INPUT_COLS, TARGET_COL, N_TOP_ZONES_ONEHOT, ONEHOT_ZONE_PREFIX,
    ROUTE_TE_SMOOTHING,
)
from src.features.domain import (
    add_zone_features,
    add_airport_features,
    add_hotspot_features,
    add_cbd_crossing,
    add_borough_flags,
    add_metered_fare_estimate,
    add_congestion_surcharge,
    add_cbd_fee,
    add_time_surcharges,
    add_estimated_charges_total,
    add_trip_shape,
    add_extra_time_flags,
    learn_zone_popularity,
    add_zone_popularity,
    add_top_zone_onehot,
    learn_route_stats,
    add_route_features,
    learn_zone_fare_std,
    add_zone_fare_std,
)
from src.features.holidays import add_holiday_features


# ---------------------------------------------------------------------------
# Feature column lists (ordered for readability)
# ---------------------------------------------------------------------------

# Numeric features fed to all model types
NUMERIC_FEATURES: List[str] = [
    # --- distance ---
    "trip_distance",
    "log_distance",
    "distance_sq",
    "sqrt_distance",
    "is_short_trip",
    "is_long_trip",
    "is_same_zone",
    # --- time (raw) ---
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_day",
    "pickup_month",
    # --- time (cyclic) ---
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    # --- time (binary) ---
    "is_weekend",
    "is_rush_hour",
    "is_overnight",
    "is_late_night",
    "is_morning_rush",
    "is_evening_rush",
    # --- airport ---
    "is_jfk_pu",
    "is_lga_pu",
    "is_jfk_do",
    "is_lga_do",
    "is_airport_pickup",
    "is_airport_route",
    "airport_fee_est",
    # --- hotspot / high-demand zones ---
    "is_west_village_pu",
    "is_west_village_do",
    "is_west_village_route",
    "is_hotspot_pu",
    "is_hotspot_do",
    "is_hotspot_route",
    "pu_zone_popularity",
    "do_zone_popularity",
    "route_popularity",
    # --- zone type ---
    "is_yellow_zone_pu",
    "is_yellow_zone_do",
    "is_manhattan_pu",
    "is_manhattan_do",
    "is_cross_borough",
    "crosses_cbd",
    "fully_within_cbd",
    "enters_cbd",
    "exits_cbd",
    # --- borough-specific (weak-segment handles) ---
    "is_brooklyn_pu",
    "is_brooklyn_do",
    "is_queens_pu",
    "is_queens_do",
    "is_bronx_pu",
    "is_bronx_do",
    "is_outer_borough_pu",
    "is_outer_borough_do",
    # --- holidays ---
    "is_holiday",
    "is_major_holiday",
    "is_federal_holiday",
    "is_christian_holiday",
    "is_jewish_holiday",
    "is_muslim_holiday",
    "is_other_cultural_holiday",
    "days_to_nearest_holiday",
    "jewish_holiday_x_brooklyn",
    "muslim_holiday_x_queens",
    "cultural_holiday_x_queens",
    # --- surcharge estimates ---
    "congestion_surcharge_est",
    "cbd_fee_est",
    "is_post_cbd",
    "extra_est",
    "mta_tax_est",
    "improvement_surcharge_est",
    "estimated_surcharges",
    "est_metered_fare",
    # --- interactions ---
    "distance_x_airport",
    "distance_x_rush",
    "distance_x_manhattan",
    "distance_x_hotspot",
    "distance_x_cross_borough",
    "distance_x_post_cbd",
    "distance_x_cbd_cross",
    "hour_x_distance",
    "cbd_active_cross",
    "overnight_x_distance",
    "night_x_airport",
    "longtrip_x_airport",
    # --- target encoding ---
    "pu_zone_mean_fare",
    "do_zone_mean_fare",
    "pu_zone_std_fare",
    "do_zone_std_fare",
    "route_mean_fare",
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
    df["distance_x_manhattan"] = df["trip_distance"] * df["is_manhattan_do"]
    df["distance_x_hotspot"] = df["trip_distance"] * df["is_hotspot_route"]
    df["distance_x_cross_borough"] = df["trip_distance"] * df["is_cross_borough"]
    df["distance_x_post_cbd"] = df["trip_distance"] * df["is_post_cbd"]
    df["distance_x_cbd_cross"] = df["trip_distance"] * df["crosses_cbd"]
    df["hour_x_distance"] = df["pickup_hour"] * df["trip_distance"]
    # Targeted at the highest-error segments (CV error analysis):
    #   crosses_cbd=1, nighttime, and long trips.
    df["cbd_active_cross"] = df["crosses_cbd"] * df["is_post_cbd"]   # $9 fee only post-2025
    df["overnight_x_distance"] = df["is_overnight"] * df["trip_distance"]
    df["night_x_airport"] = df["is_overnight"] * df["is_airport_route"]
    df["longtrip_x_airport"] = df["is_long_trip"] * df["is_airport_route"]
    # Borough x religious/cultural-holiday interactions — domain priors on
    # which NYC communities are concentrated where (not measured demographic
    # data): Brooklyn has the city's largest Orthodox/Hasidic Jewish
    # population (Borough Park, Williamsburg, Crown Heights); Queens has large
    # South Asian/Middle Eastern Muslim communities (Jackson Heights, Astoria,
    # Richmond Hill) and the Flushing Chinese community relevant to Diwali/
    # Lunar New Year. Left for SHAP/feature-selection to validate or reject.
    df["jewish_holiday_x_brooklyn"] = df["is_jewish_holiday"] * (df["is_brooklyn_pu"] | df["is_brooklyn_do"])
    df["muslim_holiday_x_queens"] = df["is_muslim_holiday"] * (df["is_queens_pu"] | df["is_queens_do"])
    df["cultural_holiday_x_queens"] = df["is_other_cultural_holiday"] * (df["is_queens_pu"] | df["is_queens_do"])
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
        # Learned zone-popularity (unsupervised frequency) + top-zone one-hot
        self._pu_freq: Optional[pd.Series] = None
        self._do_freq: Optional[pd.Series] = None
        self._top_pu_zones: List[int] = []
        self._onehot_cols: List[str] = []
        # Learned route-level target encoding + zone fare dispersion
        self._route_te: Optional[pd.Series] = None
        self._route_freq: Optional[pd.Series] = None
        self._pu_std: Optional[pd.Series] = None
        self._do_std: Optional[pd.Series] = None

    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """Fit all supervised encodings on the training fold (leak-free).

        Parameters
        ----------
        X : raw input DataFrame (must contain PULocationID, DOLocationID).
        y : training target (total_fare_amount). If None, encodings default to 0.
        """
        if y is not None:
            tmp = X[["PULocationID", "DOLocationID"]].copy()
            tmp["_y"] = y.values if hasattr(y, "values") else y
            self._pu_means = tmp.groupby("PULocationID")["_y"].mean()
            self._do_means = tmp.groupby("DOLocationID")["_y"].mean()
            self._global_mean = float(tmp["_y"].mean())
            # route-level (PU,DO) smoothed target encoding + zone fare std
            self._route_te, self._route_freq, self._global_mean = learn_route_stats(
                X, y, ROUTE_TE_SMOOTHING
            )
            self._pu_std, self._do_std = learn_zone_fare_std(X, y)

        # Learn zone popularity from training-data frequency only (no target,
        # so this is leak-free). Unseen zones at transform time map to 0.
        (
            self._pu_freq,
            self._do_freq,
            self._top_pu_zones,
            self._onehot_cols,
        ) = learn_zone_popularity(X, N_TOP_ZONES_ONEHOT, ONEHOT_ZONE_PREFIX)
        return self

    # ------------------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps and return the enriched DataFrame."""
        df = X.copy()

        # 1. Unsupervised transforms
        df = _add_cyclic_time(df)
        df = _add_distance_features(df)
        df = add_trip_shape(df)
        df = add_extra_time_flags(df)
        df = add_zone_features(df, self.zones_df)
        df = add_borough_flags(df)           # needs pu_borough / do_borough
        df = add_airport_features(df)
        df = add_hotspot_features(df)
        df = add_cbd_crossing(df)            # needs is_yellow_zone_* from zone features
        df = add_congestion_surcharge(df)
        df = add_cbd_fee(df)                 # sets is_post_cbd
        df = add_time_surcharges(df)
        df = add_estimated_charges_total(df)
        df = add_metered_fare_estimate(df)   # needs estimated_surcharges
        df = add_holiday_features(df)        # needs pickup_year/month/day
        df = _add_interaction_features(df)   # needs zone/time/cbd/holiday flags

        # 2. Learned zone popularity + route stats + zone fare std + one-hot
        df = add_zone_popularity(df, self._pu_freq, self._do_freq)
        df = add_route_features(df, self._route_te, self._route_freq, self._global_mean)
        df = add_zone_fare_std(df, self._pu_std, self._do_std)
        df = add_top_zone_onehot(df, self._top_pu_zones, self._onehot_cols)

        # 3. Target encoding (zone-level mean fare)
        gm = self._global_mean
        if self._pu_means is not None:
            df["pu_zone_mean_fare"] = df["PULocationID"].map(self._pu_means).fillna(gm)
            df["do_zone_mean_fare"] = df["DOLocationID"].map(self._do_means).fillna(gm)
        else:
            df["pu_zone_mean_fare"] = gm
            df["do_zone_mean_fare"] = gm

        return df

    def get_tree_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and order columns for tree-based models.

        Includes the learned top-zone one-hot columns (set during fit). If
        config.SELECTED_FEATURES is set (by select_features.py), the output is
        restricted to that learned "most predictive" subset.
        """
        cols = self.get_feature_names()
        cols = [c for c in cols if c in df.columns]
        return df[cols]

    def get_feature_names(self) -> List[str]:
        """Full candidate feature list, or the selected subset when configured."""
        all_feats = list(TREE_FEATURES) + list(self._onehot_cols)
        selected = getattr(_config, "SELECTED_FEATURES", None)
        if selected:
            return [c for c in all_feats if c in selected]
        return all_feats


def get_raw_input_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the inference-time available columns from a cleaned DataFrame."""
    cols = [c for c in RAW_INPUT_COLS if c in df.columns]
    return df[cols].copy()
