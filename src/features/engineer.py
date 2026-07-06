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
    ROUTE_TE_SMOOTHING, BOROUGH_HOLIDAY_NAMES,
)
from src.features.domain import (
    add_raw_metadata_features,
    add_zone_features,
    add_zone_geo_distance_features,
    add_airport_features,
    add_jfk_manhattan_flat_route,
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
    add_route_duration_feature,
    learn_zone_fare_std,
    add_zone_fare_std,
)
from src.features.holidays import add_holiday_features
from src.features.macro import add_macro_features


# ---------------------------------------------------------------------------
# Feature column lists (ordered for readability)
# ---------------------------------------------------------------------------

# Numeric features fed to all model types
NUMERIC_FEATURES: List[str] = [
    # --- raw trip metadata ---
    "VendorID",
    "passenger_count",
    "store_and_fwd_flag_enc",
    # --- distance ---
    "trip_distance",
    "log_distance",
    "distance_sq",
    "sqrt_distance",
    "zone_manhattan_distance",
    "zone_euclidean_distance",
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
    "is_lga_route",
    "lga_surcharge_est",
    "ewr_surcharge_est",
    "is_jfk_manhattan_flat_route",
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
    "is_cbd_pu",
    "is_cbd_do",
    "is_cross_borough",
    "crosses_cbd",
    "fully_within_cbd",
    "enters_cbd",
    "exits_cbd",
    "is_outside_nyc_pu",
    "is_outside_nyc_do",
    "is_outside_nyc_route",
    # --- borough-specific (weak-segment handles) ---
    "is_brooklyn_pu",
    "is_brooklyn_do",
    "is_queens_pu",
    "is_queens_do",
    "is_bronx_pu",
    "is_bronx_do",
    "is_staten_island_pu",
    "is_staten_island_do",
    "is_outer_borough_pu",
    "is_outer_borough_do",
    # --- holidays ---
    "is_holiday",
    "is_major_holiday",
    "is_federal_holiday",
    "is_legal_holiday",
    "is_christian_holiday",
    "is_jewish_holiday",
    "is_muslim_holiday",
    "is_other_cultural_holiday",
    "days_to_nearest_holiday",
    # --- macro-financial (wealth-effect / ability-to-pay proxy) ---
    "dxy_level",
    "sp500_level",
    "dxy_change_1m",
    "sp500_return_1m",
    # --- borough x holiday-religion interactions (full unbiased cross product) ---
    "christian_holiday_x_manhattan",
    "jewish_holiday_x_manhattan",
    "muslim_holiday_x_manhattan",
    "other_cultural_holiday_x_manhattan",
    "christian_holiday_x_brooklyn",
    "jewish_holiday_x_brooklyn",
    "muslim_holiday_x_brooklyn",
    "other_cultural_holiday_x_brooklyn",
    "christian_holiday_x_queens",
    "jewish_holiday_x_queens",
    "muslim_holiday_x_queens",
    "other_cultural_holiday_x_queens",
    "christian_holiday_x_bronx",
    "jewish_holiday_x_bronx",
    "muslim_holiday_x_bronx",
    "other_cultural_holiday_x_bronx",
    "christian_holiday_x_staten_island",
    "jewish_holiday_x_staten_island",
    "muslim_holiday_x_staten_island",
    "other_cultural_holiday_x_staten_island",
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
    "distance_x_jfk_flat",
    # --- target encoding ---
    "pu_zone_mean_fare",
    "do_zone_mean_fare",
    "pu_zone_std_fare",
    "do_zone_std_fare",
    "route_mean_fare",
    "route_mean_duration_min",
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
    # Residual distance signal within the JFK<->Manhattan flat-rate regime
    # (real fares there aren't PERFECTLY flat -- mean $72, std $11 in our
    # 2026 test data -- so let the tree use distance to explain that residual
    # instead of forcing every flat-route trip to an identical prediction).
    df["distance_x_jfk_flat"] = df["trip_distance"] * df["is_jfk_manhattan_flat_route"]
    # Borough x religious/cultural-holiday interactions — full, unbiased cross
    # product of all 5 boroughs x all 4 holiday-religion categories. No prior
    # assumption about which borough is "affected" by which holiday (e.g.
    # Brooklyn/Jewish, Queens/Muslim) is baked in here; every combination is
    # generated identically and left for SHAP/feature-selection to validate
    # or reject on its own merit.
    for borough in BOROUGH_HOLIDAY_NAMES:
        touches_borough = df[f"is_{borough}_pu"] | df[f"is_{borough}_do"]
        for religion in ("christian", "jewish", "muslim", "other_cultural"):
            df[f"{religion}_holiday_x_{borough}"] = df[f"is_{religion}_holiday"] * touches_borough
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
        # Learned route-level MEAN DURATION (leak-free proxy for time-based
        # fare component -- see add_route_duration_feature)
        self._route_duration_te: Optional[pd.Series] = None
        self._global_mean_duration: float = 0.0

    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        duration: Optional[pd.Series] = None,
    ) -> "FeatureEngineer":
        """Fit all supervised encodings on the training fold (leak-free).

        Parameters
        ----------
        X : raw input DataFrame (must contain PULocationID, DOLocationID).
        y : training target (total_fare_amount). If None, encodings default to 0.
        duration : this fold's trip_duration_min (same row order as X), used
            ONLY to fit a route-level mean-duration lookup (never as a
            per-trip input -- see add_route_duration_feature). If None,
            route_mean_duration_min falls back to a constant 0.0.
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

        if duration is not None:
            self._route_duration_te, _, self._global_mean_duration = learn_route_stats(
                X, duration, ROUTE_TE_SMOOTHING
            )

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
        df = add_raw_metadata_features(df)
        df = _add_cyclic_time(df)
        df = _add_distance_features(df)
        df = add_trip_shape(df)
        df = add_extra_time_flags(df)
        # Moved before add_time_surcharges (needs pickup_year/month/day only,
        # which are raw inputs available from the start) -- add_time_surcharges
        # needs is_legal_holiday to gate the rush-hour surcharge correctly.
        df = add_holiday_features(df)
        df = add_zone_features(df, self.zones_df)
        df = add_zone_geo_distance_features(df, self.zones_df)  # needs trip_distance for the NaN fallback
        df = add_borough_flags(df)           # needs pu_borough / do_borough
        df = add_airport_features(df)
        df = add_jfk_manhattan_flat_route(df)  # needs pu/do_borough + is_jfk_pu/do
        df = add_hotspot_features(df)
        df = add_cbd_crossing(df)            # needs is_yellow_zone_* from zone features
        df = add_congestion_surcharge(df)
        df = add_cbd_fee(df)                 # sets is_post_cbd
        df = add_time_surcharges(df)         # needs is_jfk_manhattan_flat_route + is_legal_holiday
        df = add_estimated_charges_total(df)
        df = add_metered_fare_estimate(df)   # needs estimated_surcharges + is_jfk_manhattan_flat_route
        df = add_macro_features(df)          # needs pickup_year/month/day
        df = _add_interaction_features(df)   # needs zone/time/cbd/holiday/jfk-flat flags

        # 2. Learned zone popularity + route stats + zone fare std + one-hot
        df = add_zone_popularity(df, self._pu_freq, self._do_freq)
        df = add_route_features(df, self._route_te, self._route_freq, self._global_mean)
        df = add_route_duration_feature(df, self._route_duration_te, self._global_mean_duration)
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
