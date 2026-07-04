"""Centralised project configuration: paths, constants, and model defaults."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent

DATA_PATHS = {
    "training": ROOT_DIR / "training_set",
    "test": ROOT_DIR / "test_set",
    "taxi_zones": ROOT_DIR / "taxi_zone_lookup.csv",
}

MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
TARGET_COL = "total_fare_amount"

# Columns available at inference time (booking moment)
RAW_INPUT_COLS = [
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "pickup_year",
]

# ---------------------------------------------------------------------------
# Data cleaning thresholds  (mirrored from EDA findings)
# ---------------------------------------------------------------------------
CLEANING = {
    "trip_distance_min": 0.01,
    "trip_distance_max": 100.0,
    "trip_duration_min": 1.0,
    "trip_duration_max": 180.0,
    "total_fare_min": 0.01,
    "total_fare_max": 300.0,
    "outlier_percentile": 0.99,
}

# ---------------------------------------------------------------------------
# NYC TLC domain constants
# (source: https://www.nyc.gov/site/tlc/passengers/taxi-fare.page)
# ---------------------------------------------------------------------------
TLC_RULES = {
    # Fixed per-trip surcharges
    "mta_tax": 0.50,
    "improvement_surcharge": 1.00,
    "congestion_surcharge": 2.50,   # for trips in/to Manhattan (Yellow Zone)
    "airport_fee": 1.75,            # pickup at JFK or LGA only
    "cbd_congestion_fee": 9.00,     # Congestion Relief Zone, from 2025-01-05
    # Time extras (approximate)
    "extra_rush_hour": 1.00,        # weekdays 16:00-20:00
    "extra_overnight": 0.50,        # 20:00-06:00
    # Zone IDs
    "jfk_zone_id": 132,
    "lga_zone_id": 138,
    "ewr_zone_id": 1,
    # Temporal rules
    "cbd_start_year": 2025,
    "rush_hour_start": 16,
    "rush_hour_end": 20,
    "overnight_start": 20,
    "overnight_end": 6,
}

# Stable categorical value mappings (must stay consistent across train/val/test)
BOROUGH_MAP = {
    "Bronx": 0,
    "Brooklyn": 1,
    "EWR": 2,
    "Manhattan": 3,
    "Queens": 4,
    "Staten Island": 5,
    "Unknown": 6,
}

SERVICE_ZONE_MAP = {
    "Airports": 0,
    "Boro Zone": 1,
    "EWR": 2,
    "Yellow Zone": 3,
    "Unknown": 4,
}

# ---------------------------------------------------------------------------
# High-demand "hotspot" zones (domain knowledge from EDA)
# ---------------------------------------------------------------------------
# West Village was flagged in the EDA as an especially popular / informative
# pickup area (weekend + nightlife demand). LocationID 249 = "West Village",
# 158 = "Meatpacking/West Village West".
WEST_VILLAGE_ZONES = [249, 158]

# Broader nightlife / high-traffic Manhattan cluster. Used for a coarse
# is_hotspot flag; the per-zone popularity signal is learned from data
# (see FeatureEngineer), this list only encodes stable domain priors.
HOTSPOT_ZONES = [
    249, 158,              # West Village / Meatpacking
    113, 114,              # Greenwich Village North / South
    79, 148,               # East Village, Lower East Side
    234, 90, 107,          # Union Sq, Flatiron, Gramercy
    230,                   # Times Sq / Theatre District
    211, 144,              # SoHo, Little Italy / NoLiTa
    161, 162, 163, 164,    # Midtown Center / East / North / South
]

# Number of most-frequent pickup zones to one-hot encode. Learned at fit()
# time from training-data frequency (unsupervised → no target leakage).
# Kept small deliberately (lean encoding: trees use ordinal/target encoding;
# one-hot mainly helps the linear Ridge model).
N_TOP_ZONES_ONEHOT = 12

# Column-name prefix for the learned top-zone one-hot features
# (e.g. "pu_top_zone_161"). Kept in config so any code that needs to detect
# these columns can reference a single source of truth.
ONEHOT_ZONE_PREFIX = "pu_top_zone_"

# ---------------------------------------------------------------------------
# Metered-fare skeleton (rough NYC yellow-cab meter, used as an engineered
# candidate feature — a strong distance-based fare prior).
# Source: TLC standard metered rate (initial charge + per-mile).
# ---------------------------------------------------------------------------
METERED_FARE = {
    "base": 3.00,       # initial charge
    "per_mile": 3.50,   # ~ $0.70 per 1/5 mile
}

# Smoothing weight for learned (PU, DO) route-level target encoding:
# route_mean = (sum_fare + m * global_mean) / (count + m). Higher m = more
# shrinkage toward the global mean for rare routes.
ROUTE_TE_SMOOTHING = 20.0

# ---------------------------------------------------------------------------
# Feature selection (step 3): the wide candidate net is always computed, but
# training can optionally be restricted to a learned "most predictive" subset
# chosen by select_features.py (forward-chaining CV). When SELECTED_FEATURES is
# None the full ~98-feature candidate set is used; otherwise get_tree_features()
# filters to the given list.
#
# NOTE: a 30k-sample forward-chaining sweep found a 12-feature subset with
# slightly lower CV RMSE than the full set, but at production scale (150k/month)
# this pruning reduced performance widely — the smaller sample under-sampled
# rare zones/routes, making high-cardinality features (PU/DOLocationID-derived
# encodings, zone one-hots) look like noise when they are not at full scale.
# Reverted to the full candidate net for training. Re-validate any future
# pruning at the actual training sample size before trusting it.
# ---------------------------------------------------------------------------
SELECTED_FEATURES = None

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
SAMPLE_CONFIG = {
    "n_per_month_train": 150_000,
    "n_per_month_test": None,       # None = load full test set
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# Validation split: last N months of training data held out
# ---------------------------------------------------------------------------
VAL_YEARS_MONTHS = [(2025, 11), (2025, 12)]  # Nov + Dec 2025

# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------
WANDB_PROJECT = "nyc-tlc-fare-prediction"
WANDB_ENTITY = None  # defaults to logged-in W&B user

# ---------------------------------------------------------------------------
# Model hyperparameter defaults
# ---------------------------------------------------------------------------
MODEL_DEFAULTS = {
    "lgbm": {
        # n_estimators is a ceiling, not a target: early stopping (see
        # trainer._fit_lgbm) determines the actual number of trees from
        # validation performance. Set high so the cap is never the binding
        # constraint — 1000 was truncating convergence on the full feature set.
        "n_estimators": 3000,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
    "xgb": {
        # Same rationale as lgbm above — ceiling, early stopping picks the count.
        "n_estimators": 3000,
        "learning_rate": 0.05,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
    },
    "rf": {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_leaf": 10,
        "max_samples": 200_000,   # cap bootstrap size per tree — keeps RF fast on large datasets
        "random_state": 42,
        "n_jobs": -1,
    },
    "ridge": {
        "alpha": 1.0,
    },
}
