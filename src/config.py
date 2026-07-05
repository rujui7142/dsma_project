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
    "pickup_day",
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
    # A small fraction of TLC rows carry corrupt pickup timestamps (years like
    # 2007, 2008 ...). They pollute the forward-chaining month buckets (e.g. a
    # fold labelled "2007-12..2024-02"). Drop anything before Jan 2014.
    "min_pickup_year": 2014,
}

# ---------------------------------------------------------------------------
# NYC TLC domain constants
# (source: https://www.nyc.gov/site/tlc/passengers/taxi-fare.page)
# ---------------------------------------------------------------------------
TLC_RULES = {
    # Fixed per-trip surcharges
    "mta_tax": 0.50,
    "improvement_surcharge": 1.00,
    "congestion_surcharge": 2.50,   # NYS congestion surcharge, Manhattan south of 96th St (Yellow Zone)
    "airport_fee": 1.75,            # pickup at JFK or LGA only
    # CRZ per-trip charge, Manhattan south of 60th St, from 2025-01-05.
    # CORRECTED 9.00 -> 0.75: the $9 figure is the base congestion toll for
    # PRIVATE passenger vehicles. Yellow taxis are exempt from that toll and
    # instead pass a flat $0.75 per-trip charge to the passenger (high-volume
    # FHVs like Uber pay $1.50). This dataset is yellow-taxi trips, so $0.75
    # is what actually appears in total_amount. The old $9.00 inflated
    # cbd_fee_est / estimated_surcharges / est_metered_fare by ~$8.25 on every
    # post-2025 Manhattan trip -- the main driver of the fold-3 overprediction.
    # NOTE: the CRZ boundary is 60th St, but we proxy it with the Yellow Zone
    # (96th St) service-zone flag, so a band between 60th-96th St is
    # over-attributed the $0.75; small given the corrected magnitude.
    "cbd_congestion_fee": 0.75,
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
# Features excluded from PSI/drift MONITORING (not from the model itself).
#
# In forward-chaining CV each fold validates on a LATER calendar window than
# the fixed reference, so pure time-index features shift by construction and
# always score high PSI — but that is not actionable "the data broke" drift,
# it just restates "time moved forward". Reporting them buried the ~13 genuine
# CBD-regime signals under uninformative red rows.
#
# NOTE: PSI measures the distribution shift of a feature between reference and
# current data — NOT any "actual vs estimated" accuracy. pickup_month scoring
# high PSI does not mean the month was computed wrong; it means the validation
# fold covers different months than the reference (expected in temporal CV).
#
# is_post_cbd / cbd_* are deliberately NOT excluded: their shift reflects a
# real external regime change (the 2025 congestion fee) with modelling impact.
# ---------------------------------------------------------------------------
DRIFT_EXCLUDE_FEATURES = ["pickup_month", "pickup_year", "month_sin", "month_cos"]

# ---------------------------------------------------------------------------
# Monotonic constraints for tree models (LightGBM / XGBoost).
#
# Root cause fixed: fold-3 investigation found the Jan-2025 CBD $9 fee pushes
# estimated_surcharges / cbd_fee_est to values that never existed for SHORT
# trips during training (previously, only long/expensive airport trips had
# comparably high combined surcharges). Unconstrained trees route these
# out-of-range values to whatever leaf covers that boundary — which was
# calibrated on rare high-surcharge/long-distance rows — causing systematic
# OVERprediction (~+$7-8 bias) on CBD trips post-shift.
#
# A monotone_constraints=+1 on these fare-additive features forces the model
# to extrapolate them as a smooth non-decreasing effect on predicted fare
# (the true domain relationship) instead of an arbitrary leaf lookup, which
# fixes exactly this failure mode for values beyond the training range.
# ---------------------------------------------------------------------------
MONOTONIC_INCREASING_FEATURES = [
    "trip_distance", "log_distance", "distance_sq", "sqrt_distance",
    "est_metered_fare",
    "cbd_fee_est", "airport_fee_est", "congestion_surcharge_est", "extra_est",
    "estimated_surcharges",
    "route_mean_fare", "pu_zone_mean_fare", "do_zone_mean_fare",
]

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
SAMPLE_CONFIG = {
    "n_per_month_train": 150_000,
    # Capped, not None ("load full month"): a full unsampled TLC month is
    # 2.5-3.5M rows, and evaluate.py also builds Evidently drift reports over
    # the ~98-feature engineered set -- combined, that OOMs a standard
    # 7GB GitHub Actions runner. 200k/month is still a large, statistically
    # robust evaluation sample. Override via evaluate.py's --sample flag.
    "n_per_month_test": 200_000,
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
# Tuned via the 20-trial W&B sweep (tag "hp-retune", 30k sample/month,
# lgbm/xgb=bayes, rf=random, ridge=grid — the pre-two-phase sweep protocol).
# Best val_rmse: xgb=8.346, lgbm=8.375, ridge=8.452, rf=8.467 (tightly
# clustered on the full feature set). n_estimators is deliberately NOT taken
# from the sweep for lgbm/xgb: it's a ceiling in our design (see
# trainer.EARLY_STOPPING_ROUNDS), and the sweep tuned it as one swept value
# among others (500-1500) rather than as "as high as possible" — adopting a
# low sweep value here would reintroduce the truncated-convergence bug we
# fixed by raising it to 3000. Re-run with the new two-phase protocol
# (random -> narrowed bayes) to refine further.
MODEL_DEFAULTS = {
    "lgbm": {
        # n_estimators is a ceiling, not a target: early stopping (see
        # trainer._fit_lgbm) determines the actual number of trees from
        # validation performance. Set high so the cap is never the binding
        # constraint — 1000 was truncating convergence on the full feature set.
        "n_estimators": 3000,
        "learning_rate": 0.1185205798862516,
        "num_leaves": 45,
        "max_depth": 12,
        "min_child_samples": 110,
        "subsample": 0.9306560055940204,
        "colsample_bytree": 0.5662342412045633,
        "reg_alpha": 1.2312824586076985,
        "reg_lambda": 1.521244034412757,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
    "xgb": {
        # Same rationale as lgbm above — ceiling, early stopping picks the count.
        "n_estimators": 3000,
        "learning_rate": 0.1504414371946877,
        "max_depth": 12,
        "subsample": 0.5361097024847514,
        "colsample_bytree": 0.5715637330073469,
        "reg_alpha": 0.0792290051076241,
        "reg_lambda": 1.3483715452148335,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
    },
    "rf": {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_leaf": 5,
        "max_features": "log2",
        "max_samples": 150_000,   # cap bootstrap size per tree — keeps RF fast on large datasets
        "random_state": 42,
        "n_jobs": -1,
    },
    "ridge": {
        # Sweep's best (0.001) sat at the search grid's lower boundary —
        # re-run with the new two-phase protocol (continuous log-uniform
        # range) to check whether even less regularization helps further.
        "alpha": 0.001,
    },
}
