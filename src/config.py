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
    # Zone centroids (NAD83 State Plane NY Long Island, US survey feet --
    # same CRS as the official TLC taxi_zones shapefile they're derived from)
    # used to compute a genuine geographic distance between PU/DO zones,
    # independent of the metered trip_distance. See features/domain.py
    # add_zone_geo_distance_features and scripts/build_zone_centroids.py.
    "taxi_zone_centroids": ROOT_DIR / "taxi_zone_centroids.csv",
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
    "VendorID",
    "passenger_count",
    "store_and_fwd_flag",
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
    "min_pickup_year": 2024,
    # Robust (MAD-based) modified-z-score threshold for the fare-efficiency
    # (fare-per-mile / fare-per-minute) joint-anomaly filter -- catches rows
    # like a $92.94 fare for a 0.01-mile, 40-minute trip, which pass every
    # existing MARGINAL bound individually. 3.5 is the standard Iglewicz &
    # Hoya (1993) convention for this method. Non-airport trips only (see
    # cleaner.filter_fare_efficiency_outliers).
    "fare_efficiency_zscore": 3.5,
}

# ---------------------------------------------------------------------------
# NYC TLC domain constants
# (source: https://www.nyc.gov/site/tlc/passengers/taxi-fare.page)
# ---------------------------------------------------------------------------
TLC_RULES = {
    # Fixed per-trip surcharges
    "mta_tax": 0.50,
    "improvement_surcharge": 1.00,
    "congestion_surcharge": 2.50,  # NYS congestion surcharge, Manhattan south of 96th St (Yellow Zone)
    # Airport Access Fee: pickup at JFK or LGA only. CORRECTED 1.75 -> 2.00
    # against the current official rate card (nyc.gov/site/tlc/passengers/
    # taxi-fare.page) -- was stale by one prior rate-increase cycle, same
    # issue as extra_rush_hour/extra_overnight below.
    "airport_fee": 2.00,
    # LGA-specific: "Trips to and from LaGuardia Airport are charged the
    # standard metered fare, plus a $5.00 surcharge" -- applies for LGA on
    # EITHER end (pickup or dropoff), distinct from and additive with the
    # $2.00 Airport Access Fee above (which is pickup-only).
    "lga_surcharge": 5.00,
    # Newark Surcharge: "Trips to Newark Airport (EWR): Standard metered
    # fare, plus $20.00 Newark Surcharge" -- dropoff at EWR only (the rate
    # card only documents this direction).
    "ewr_surcharge": 20.00,
    # JFK<->Manhattan flat rate ("Rate #2 - JFK Airport"), either direction:
    # $70 REPLACES the distance/time-metered base fare entirely (surcharges
    # below still apply additively on top). Trips between JFK and any OTHER
    # NYC destination are standard metered fare (not flat) -- see
    # domain.add_jfk_manhattan_flat_route for the exact route condition.
    "jfk_manhattan_flat_fare": 70.00,
    # The JFK flat-rate page lists its own rush-hour surcharge ($5, not the
    # standard $2.50) and no overnight surcharge at all -- see
    # domain.add_time_surcharges.
    "jfk_manhattan_rush_surcharge": 5.00,
    # CRZ per-trip charge, Manhattan at/south of 60th St (config.CBD_ZONES,
    # is_cbd_pu/do), from 2025-01-05. CORRECTED 9.00 -> 0.75: the $9 figure is
    # the base congestion toll for PRIVATE passenger vehicles. Yellow taxis
    # are exempt from that toll and instead pass a flat $0.75 per-trip charge
    # to the passenger (high-volume FHVs like Uber pay $1.50). This dataset
    # is yellow-taxi trips, so $0.75 is what actually appears in total_amount.
    "cbd_congestion_fee": 0.75,
    # Time extras. CORRECTED via web search against the current official TLC
    # rate card (nyc.gov/site/tlc/passengers/taxi-fare.page): both were stale
    # by roughly one prior rate-increase cycle. These have been wrong for the
    # ENTIRE 2024-2025 dataset (not just post-2025), so unlike the CBD fee bug
    # this is a uniform bias across all folds, not a temporal-drift driver.
    "extra_rush_hour": 2.50,  # weekdays 16:00-20:00, excluding legal holidays (was 1.00)
    "extra_overnight": 1.00,  # 20:00-06:00 (was 0.50)
    # Zone IDs
    "jfk_zone_id": 132,
    "lga_zone_id": 138,
    "ewr_zone_id": 1,
    # "Outside of NYC" catch-all zone (taxi_zone_lookup.csv id 265, Borough
    # "N/A") -- conflates two distinct real-world rate regimes we can't tell
    # apart from zone ID alone: Rate #04 (Westchester/Nassau, metered fare
    # DOUBLED for the portion beyond city limits -- but we only have a single
    # aggregate trip_distance/duration, not a within-city/beyond-city split)
    # and Rate #05 (further counties, a fully negotiated flat fare with no
    # formula at all). Modeled as a single flag (domain.is_outside_nyc_*) and
    # left for the model to learn an empirical average adjustment from
    # historical fares, rather than hard-coding a distance-doubling formula
    # we cannot correctly bound from the data we have.
    "outside_nyc_zone_id": 265,
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
# NYC Congestion Relief Zone (CRZ) — the true $0.75 CBD toll boundary.
#
# The MTA's official rule: all local Manhattan streets/avenues AT OR SOUTH OF
# 60th Street (source: mta.info/agency/bridges-and-tunnels/congestion-relief-zone).
# This is a TIGHTER boundary than the "Yellow Zone" service_zone label already
# used elsewhere (that one extends to 96th St, for the separate $2.50 NYS
# congestion surcharge). Built by classifying every Manhattan taxi zone
# against the 60th-St line (verified against the MTA's public boundary
# description), not a coarse proxy.
# ---------------------------------------------------------------------------
CBD_ZONES = [
    4, 12, 13, 45, 48, 50, 68, 79, 87, 88, 90, 100, 107, 113, 114, 125, 137,
    144, 148, 158, 161, 162, 163, 164, 170,           # Midtown/Downtown core
    186, 209, 211, 224, 229, 230, 231, 232, 233, 234, # Flatiron/Union Sq/SoHo/
    246, 249, 261,                                    # TriBeCa/West Village/WTC
]

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
    249,
    158,  # West Village / Meatpacking
    113,
    114,  # Greenwich Village North / South
    79,
    148,  # East Village, Lower East Side
    234,
    90,
    107,  # Union Sq, Flatiron, Gramercy
    230,  # Times Sq / Theatre District
    211,
    144,  # SoHo, Little Italy / NoLiTa
    161,
    162,
    163,
    164,  # Midtown Center / East / North / South
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
    "base": 3.00,  # initial charge
    "per_mile": 3.50,  # ~ $0.70 per 1/5 mile
}

# Smoothing weight for learned (PU, DO) route-level target encoding:
# route_mean = (sum_fare + m * global_mean) / (count + m). Higher m = more
# shrinkage toward the global mean for rare routes.
ROUTE_TE_SMOOTHING = 20.0

# ---------------------------------------------------------------------------
# Feature selection (step 3): the wide candidate net is always computed, but
# training can optionally be restricted to a learned "most predictive" subset.
# When SELECTED_FEATURES is None the full ~147-feature candidate set is used;
# otherwise get_tree_features() filters to the given list.
#
# HISTORICAL NOTE: an earlier 30k-sample forward-chaining sweep found a
# 12-feature subset with slightly lower CV RMSE than the full set, but at
# production scale (150k/month) that pruning reduced performance widely --
# the smaller sample under-sampled rare zones/routes, making high-cardinality
# features (PU/DOLocationID-derived encodings, zone one-hots) look like noise
# when they are not at full scale. Reverted to the full candidate net at the
# time.
#
# CURRENT SELECTION (src/feature_selection_temporal.py, tag trial4-protected):
# an incremental K-sweep -- rank by mean SHAP importance, sweep K from small
# to full, keep the smallest K where BOTH CV MAE and Val MAE hold within
# tolerance of their best observed values -- with the TLC fare-rule features
# (JFK/LGA/EWR fees, CBD fee, holiday-gated rush hour, outside-NYC flag)
# protected from removal regardless of aggregate SHAP rank, since they were
# purpose-built to fix a specific worst-performing SEGMENT (20+mi/JFK bucket)
# that an aggregate-MAE-only ranking can't see the value of. Initially run at
# --sample 30000 (best K=39, CV MAE 2.3525 vs 2.4108 full, Val MAE 3.3911 vs
# 3.4514 full) -- then explicitly RE-VALIDATED at production scale (150k/month,
# src/validate_feature_selection.py) to guard against repeating the exact
# historical mistake above. It held, and by a WIDER margin at full scale:
# CV MAE 2.2903 vs 2.3492 full, Val MAE 3.3620 vs 3.3973 full, and the
# selected set beat the full set on every one of the 5 individual CV folds
# too -- not a small-sample artifact.
#
# ADDED SEPARATELY (src/trial_raw_metadata_features.py): VendorID,
# passenger_count, store_and_fwd_flag_enc. None has a documented causal link
# to fare, but each was validated the same way as everything else here --
# forward-chaining CV + Val ablation -- and improved MAE on every one of the
# 5 folds plus Val (CV MAE 2.2985 vs 2.3209, Val MAE 3.3459 vs 3.3698 without
# them), most likely proxying for other structure (vendor fleet/service-area
# differences, connectivity patterns correlating with trip type) rather than
# a direct pricing effect.
# ---------------------------------------------------------------------------
SELECTED_FEATURES = [
    "PULocationID", "DOLocationID", "trip_distance", "log_distance", "distance_sq",
    "zone_manhattan_distance", "zone_euclidean_distance", "pickup_hour",
    "pickup_dayofweek", "pickup_month", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_jfk_pu", "is_lga_pu", "is_jfk_do", "is_lga_do", "is_airport_pickup",
    "is_airport_route", "airport_fee_est", "is_lga_route", "lga_surcharge_est",
    "ewr_surcharge_est", "is_jfk_manhattan_flat_route", "is_outside_nyc_pu",
    "is_outside_nyc_do", "is_outside_nyc_route", "is_legal_holiday",
    "congestion_surcharge_est", "cbd_fee_est", "is_post_cbd", "est_metered_fare",
    "distance_x_manhattan", "distance_x_cross_borough", "hour_x_distance",
    "distance_x_jfk_flat", "route_mean_fare", "route_mean_duration_min",
    "VendorID", "passenger_count", "store_and_fwd_flag_enc",
]

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
#
# days_to_nearest_holiday is excluded for the same reason as the calendar
# features above: it's seasonally cyclical (peaks/troughs recur every year
# around the same holidays), so a fold validating on a different slice of the
# calendar than the reference will show it "drifted" purely because time
# moved forward, not because the holiday calendar itself changed.
#
# Same logic extends to every holiday flag EXCEPT ones that land on a fixed
# Gregorian calendar date every year. Movable/lunar holidays (Jewish, Muslim,
# Diwali/Lunar New Year, Easter/Good Friday) and "nth weekday of month" rule
# holidays (MLK Day, Thanksgiving, Labor Day, ...) fall on a DIFFERENT
# (month, day) each year, so a fold spanning different years than the
# reference will show spurious PSI purely from the date shifting — not a
# genuine change in the holiday itself. is_holiday / is_major_holiday /
# is_federal_holiday / is_christian_holiday are themselves MIXES of fixed and
# movable dates (e.g. is_federal_holiday includes both Jul 4 [fixed] and MLK
# Day [movable]), so they inherit the same noise and are excluded too. All 20
# borough x religion interactions involve at least one movable category.
# ---------------------------------------------------------------------------
BOROUGH_HOLIDAY_NAMES = ["manhattan", "brooklyn", "queens", "bronx", "staten_island"]
_HOLIDAY_RELIGION_NAMES = ["christian", "jewish", "muslim", "other_cultural"]

# dxy_level / sp500_level are raw index levels -- they trend across our
# window essentially by definition, so any two non-overlapping time windows
# will show elevated PSI purely because "the index moved since then", same
# issue as pickup_month. dxy_change_1m / sp500_return_1m are NOT excluded:
# a real 1-month swing (e.g. a market shock) is exactly the kind of genuine,
# actionable external event PSI should flag -- same rationale as keeping
# is_post_cbd / cbd_* in monitoring.
DRIFT_EXCLUDE_FEATURES = [
    "pickup_month", "pickup_year", "month_sin", "month_cos", "days_to_nearest_holiday",
    "is_holiday", "is_major_holiday", "is_federal_holiday", "is_legal_holiday",
    "is_christian_holiday", "is_jewish_holiday", "is_muslim_holiday", "is_other_cultural_holiday",
    "dxy_level", "sp500_level",
] + [
    f"{religion}_holiday_x_{borough}"
    for borough in BOROUGH_HOLIDAY_NAMES
    for religion in _HOLIDAY_RELIGION_NAMES
]

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
    "trip_distance",
    "log_distance",
    "distance_sq",
    "sqrt_distance",
    "zone_manhattan_distance",
    "zone_euclidean_distance",
    "est_metered_fare",
    "cbd_fee_est",
    "airport_fee_est",
    "lga_surcharge_est",
    "ewr_surcharge_est",
    "congestion_surcharge_est",
    "extra_est",
    "estimated_surcharges",
    "route_mean_fare",
    "route_mean_duration_min",
    "pu_zone_mean_fare",
    "do_zone_mean_fare",
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
        # constraint -- deliberately NOT adopted from the sweep below, which
        # tunes n_estimators as an ordinary value rather than a hard ceiling.
        "n_estimators": 3000,
        # Rest tuned via two-phase sweep, tag hp-metadata -- run against the
        # 42-feature SELECTED_FEATURES set. Phase 1 improved on phase 2 this
        # time: val_mae 3.3439 < 3.3476, so phase 1's config was kept.
        "learning_rate": 0.03332603109540011,
        "num_leaves": 126,
        "max_depth": -1,
        "min_child_samples": 175,
        "subsample": 0.5800447981178494,
        "colsample_bytree": 0.7607324900934898,
        "reg_alpha": 0.3770438460638095,
        "reg_lambda": 0.3896066185681699,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
    "xgb": {
        # Same rationale as lgbm above — ceiling, early stopping picks the count.
        "n_estimators": 3000,
        # Rest tuned via two-phase sweep, tag hp-metadata (phase 2 improved on
        # phase 1: val_mae 3.4615 < 3.4655).
        "learning_rate": 0.08248568829014966,
        "max_depth": 11,
        "subsample": 0.5787897977299619,
        "colsample_bytree": 0.5948077970388683,
        "reg_alpha": 0.3205295668334495,
        "reg_lambda": 2.0762685926188773,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
    },
    "rf": {
        # Tuned via two-phase sweep, tag hp-metadata. Phase 2 improved on
        # phase 1: val_mae 3.5081 < 3.5491.
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_leaf": 5,
        "max_features": 0.7,
        "max_samples": 200_000,  # cap bootstrap size per tree — keeps RF fast on large datasets
        "random_state": 42,
        "n_jobs": -1,
    },
    "ridge": {
        # Tuned via two-phase sweep, tag hp-metadata (continuous log-uniform
        # range); phase 1/phase 2 val_mae tied at 3.8232, took phase 2's alpha.
        "alpha": 101.35552899295628,
    },
}
