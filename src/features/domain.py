"""NYC TLC domain-based feature engineering.

Each function adds features that can be computed exclusively from inference-time
inputs (PULocationID, DOLocationID, trip_distance, pickup time components).
Surcharge estimates are derived from publicly available TLC fare rules:
https://www.nyc.gov/site/tlc/passengers/taxi-fare.page
"""

from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

from src.config import (
    TLC_RULES, BOROUGH_MAP, SERVICE_ZONE_MAP,
    WEST_VILLAGE_ZONES, HOTSPOT_ZONES, METERED_FARE, CBD_ZONES,
)

_JFK = TLC_RULES["jfk_zone_id"]
_LGA = TLC_RULES["lga_zone_id"]
_EWR = TLC_RULES["ewr_zone_id"]
_CBD_YEAR = TLC_RULES["cbd_start_year"]
_WEST_VILLAGE = set(WEST_VILLAGE_ZONES)
_HOTSPOTS = set(HOTSPOT_ZONES)
_CBD_ZONES = set(CBD_ZONES)


# ---------------------------------------------------------------------------
# Zone-level lookups
# ---------------------------------------------------------------------------

def add_zone_features(df: pd.DataFrame, zones_df: pd.DataFrame) -> pd.DataFrame:
    """Merge borough and service-zone attributes for pickup and dropoff zones."""
    df = df.copy()
    borough_map = zones_df.set_index("LocationID")["Borough"]
    svc_map = zones_df.set_index("LocationID")["service_zone"]

    df["pu_borough"] = df["PULocationID"].map(borough_map).fillna("Unknown")
    df["do_borough"] = df["DOLocationID"].map(borough_map).fillna("Unknown")
    df["pu_service_zone"] = df["PULocationID"].map(svc_map).fillna("Unknown")
    df["do_service_zone"] = df["DOLocationID"].map(svc_map).fillna("Unknown")

    df["is_manhattan_pu"] = (df["pu_borough"] == "Manhattan").astype(np.int8)
    df["is_manhattan_do"] = (df["do_borough"] == "Manhattan").astype(np.int8)
    df["is_yellow_zone_pu"] = (df["pu_service_zone"] == "Yellow Zone").astype(np.int8)
    df["is_yellow_zone_do"] = (df["do_service_zone"] == "Yellow Zone").astype(np.int8)
    # Precise Congestion Relief Zone (CRZ) membership — Manhattan at/south of
    # 60th St (see config.CBD_ZONES). Tighter than the Yellow Zone label
    # above, which is used for the separate $2.50 NYS surcharge (96th St).
    df["is_cbd_pu"] = df["PULocationID"].isin(_CBD_ZONES).astype(np.int8)
    df["is_cbd_do"] = df["DOLocationID"].isin(_CBD_ZONES).astype(np.int8)
    df["is_cross_borough"] = (df["pu_borough"] != df["do_borough"]).astype(np.int8)

    # Encode string categoricals to stable integers (same mapping at inference)
    df["pu_borough_enc"] = df["pu_borough"].map(BOROUGH_MAP).fillna(BOROUGH_MAP["Unknown"]).astype(np.int8)
    df["do_borough_enc"] = df["do_borough"].map(BOROUGH_MAP).fillna(BOROUGH_MAP["Unknown"]).astype(np.int8)
    df["pu_service_zone_enc"] = df["pu_service_zone"].map(SERVICE_ZONE_MAP).fillna(SERVICE_ZONE_MAP["Unknown"]).astype(np.int8)
    df["do_service_zone_enc"] = df["do_service_zone"].map(SERVICE_ZONE_MAP).fillna(SERVICE_ZONE_MAP["Unknown"]).astype(np.int8)

    return df


# ---------------------------------------------------------------------------
# Airport features
# ---------------------------------------------------------------------------

def add_airport_features(df: pd.DataFrame) -> pd.DataFrame:
    """Flag airport trips and estimate the $1.75 airport pickup fee.

    - airport_fee ($1.75) applies only when picked up at JFK (132) or LGA (138).
    - JFK flat-rate routes (rate code 2) are identified by zone 132 on either end.
    """
    df = df.copy()
    df["is_jfk_pu"] = (df["PULocationID"] == _JFK).astype(np.int8)
    df["is_lga_pu"] = (df["PULocationID"] == _LGA).astype(np.int8)
    df["is_jfk_do"] = (df["DOLocationID"] == _JFK).astype(np.int8)
    df["is_lga_do"] = (df["DOLocationID"] == _LGA).astype(np.int8)
    df["is_ewr_do"] = (df["DOLocationID"] == _EWR).astype(np.int8)

    df["is_airport_pickup"] = (df["is_jfk_pu"] | df["is_lga_pu"]).astype(np.int8)
    df["is_airport_route"] = (
        df["is_jfk_pu"] | df["is_lga_pu"] | df["is_jfk_do"] | df["is_lga_do"]
    ).astype(np.int8)

    df["airport_fee_est"] = df["is_airport_pickup"].astype(float) * TLC_RULES["airport_fee"]
    return df


# ---------------------------------------------------------------------------
# Hotspot / high-demand zone features (domain priors from EDA)
# ---------------------------------------------------------------------------

def add_hotspot_features(df: pd.DataFrame) -> pd.DataFrame:
    """Flag trips touching well-known high-demand Manhattan zones.

    West Village (249 / 158) was called out in the EDA as a strongly
    informative, high-volume nightlife/weekend area. HOTSPOT_ZONES is a
    broader curated cluster (Village, Union Sq, Times Sq, SoHo, Midtown...).
    These are stable domain priors; the *learned* per-zone popularity signal
    lives in FeatureEngineer.
    """
    df = df.copy()
    df["is_west_village_pu"] = df["PULocationID"].isin(_WEST_VILLAGE).astype(np.int8)
    df["is_west_village_do"] = df["DOLocationID"].isin(_WEST_VILLAGE).astype(np.int8)
    df["is_west_village_route"] = (
        df["is_west_village_pu"] | df["is_west_village_do"]
    ).astype(np.int8)

    df["is_hotspot_pu"] = df["PULocationID"].isin(_HOTSPOTS).astype(np.int8)
    df["is_hotspot_do"] = df["DOLocationID"].isin(_HOTSPOTS).astype(np.int8)
    df["is_hotspot_route"] = (df["is_hotspot_pu"] | df["is_hotspot_do"]).astype(np.int8)
    return df


# ---------------------------------------------------------------------------
# Borough-specific flags (candidate features — Brooklyn / outer boroughs were
# the weakest error segments, so give the model explicit handles on them).
# Requires pu_borough / do_borough from add_zone_features.
# ---------------------------------------------------------------------------

def add_borough_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Explicit per-borough and outer-borough indicators for PU and DO."""
    df = df.copy()
    for side, col in (("pu", "pu_borough"), ("do", "do_borough")):
        b = df[col]
        df[f"is_brooklyn_{side}"] = (b == "Brooklyn").astype(np.int8)
        df[f"is_queens_{side}"] = (b == "Queens").astype(np.int8)
        df[f"is_bronx_{side}"] = (b == "Bronx").astype(np.int8)
        df[f"is_staten_island_{side}"] = (b == "Staten Island").astype(np.int8)
        # outer borough = not Manhattan and not an airport/EWR pseudo-borough
        df[f"is_outer_borough_{side}"] = (
            ~b.isin(["Manhattan", "EWR", "Unknown"])
        ).astype(np.int8)
    return df


# ---------------------------------------------------------------------------
# Metered-fare skeleton (strong distance-based fare prior)
# Requires estimated_surcharges from add_estimated_charges_total.
# ---------------------------------------------------------------------------

def add_metered_fare_estimate(df: pd.DataFrame) -> pd.DataFrame:
    """Rough metered-fare estimate = base + per_mile*distance + surcharges."""
    df = df.copy()
    df["est_metered_fare"] = (
        METERED_FARE["base"]
        + METERED_FARE["per_mile"] * df["trip_distance"]
        + df["estimated_surcharges"]
    )
    return df


# ---------------------------------------------------------------------------
# Learned zone popularity (frequency-based, leak-free)
# ---------------------------------------------------------------------------

def learn_zone_popularity(
    X: pd.DataFrame, n_top: int, prefix: str
) -> Tuple[pd.Series, pd.Series, List[int], List[str]]:
    """Learn pickup/dropoff zone frequencies and the top-N pickup zones.

    Uses counts only (no target), so it is safe to (re)fit inside every
    forward-chaining CV fold without leaking information.

    Returns
    -------
    (pu_freq, do_freq, top_pu_zones, onehot_cols)
    """
    pu_freq = X["PULocationID"].value_counts(normalize=True)
    do_freq = X["DOLocationID"].value_counts(normalize=True)
    top_pu_zones = [int(z) for z in pu_freq.head(n_top).index]
    onehot_cols = [f"{prefix}{z}" for z in top_pu_zones]
    return pu_freq, do_freq, top_pu_zones, onehot_cols


def add_zone_popularity(
    df: pd.DataFrame,
    pu_freq: Optional[pd.Series],
    do_freq: Optional[pd.Series],
) -> pd.DataFrame:
    """Map learned zone frequencies onto pickup/dropoff. Unseen zones → 0."""
    df = df.copy()
    if pu_freq is not None:
        df["pu_zone_popularity"] = df["PULocationID"].map(pu_freq).fillna(0.0)
        df["do_zone_popularity"] = df["DOLocationID"].map(do_freq).fillna(0.0)
    else:
        df["pu_zone_popularity"] = 0.0
        df["do_zone_popularity"] = 0.0
    return df


def add_top_zone_onehot(
    df: pd.DataFrame,
    top_zones: List[int],
    onehot_cols: List[str],
) -> pd.DataFrame:
    """One-hot encode the learned top-N pickup zones as 0/1 columns."""
    df = df.copy()
    for z, col in zip(top_zones, onehot_cols):
        df[col] = (df["PULocationID"] == z).astype(np.int8)
    return df


# ---------------------------------------------------------------------------
# Learned (PU, DO) route-level target encoding + route popularity (leak-free:
# fitted on the training fold only). Route key = PU*1000 + DO (LocationID<266).
# ---------------------------------------------------------------------------

def _route_key(df: pd.DataFrame) -> pd.Series:
    return df["PULocationID"] * 1000 + df["DOLocationID"]


def learn_route_stats(
    X: pd.DataFrame, y: pd.Series, smoothing: float
) -> Tuple[pd.Series, pd.Series, float]:
    """Smoothed route mean-fare + route frequency. Returns (te, freq, global_mean)."""
    df = X[["PULocationID", "DOLocationID"]].copy()
    df["_y"] = y.values if hasattr(y, "values") else y
    df["_route"] = _route_key(df)
    grp = df.groupby("_route")["_y"]
    counts, sums = grp.count(), grp.sum()
    global_mean = float(df["_y"].mean())
    route_te = (sums + smoothing * global_mean) / (counts + smoothing)
    route_freq = counts / counts.sum()
    return route_te, route_freq, global_mean


def add_route_features(
    df: pd.DataFrame,
    route_te: Optional[pd.Series],
    route_freq: Optional[pd.Series],
    global_mean: float,
) -> pd.DataFrame:
    """Map learned route mean-fare + popularity. Unseen routes → global/0."""
    df = df.copy()
    if route_te is not None:
        key = _route_key(df)
        df["route_mean_fare"] = key.map(route_te).fillna(global_mean)
        df["route_popularity"] = key.map(route_freq).fillna(0.0)
    else:
        df["route_mean_fare"] = global_mean
        df["route_popularity"] = 0.0
    return df


def learn_zone_fare_std(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Per-zone fare dispersion (std) for PU and DO."""
    df = X[["PULocationID", "DOLocationID"]].copy()
    df["_y"] = y.values if hasattr(y, "values") else y
    return (
        df.groupby("PULocationID")["_y"].std(),
        df.groupby("DOLocationID")["_y"].std(),
    )


def add_zone_fare_std(
    df: pd.DataFrame,
    pu_std: Optional[pd.Series],
    do_std: Optional[pd.Series],
    fill: float = 0.0,
) -> pd.DataFrame:
    """Map learned per-zone fare std. Unseen / singleton zones → fill."""
    df = df.copy()
    if pu_std is not None:
        df["pu_zone_std_fare"] = df["PULocationID"].map(pu_std).fillna(fill)
        df["do_zone_std_fare"] = df["DOLocationID"].map(do_std).fillna(fill)
    else:
        df["pu_zone_std_fare"] = fill
        df["do_zone_std_fare"] = fill
    return df


# ---------------------------------------------------------------------------
# Unsupervised trip-shape + extra time candidate features
# ---------------------------------------------------------------------------

def add_trip_shape(df: pd.DataFrame) -> pd.DataFrame:
    """Distance shape helpers + same-zone flag."""
    df = df.copy()
    df["sqrt_distance"] = np.sqrt(df["trip_distance"].clip(lower=0))
    df["is_short_trip"] = (df["trip_distance"] < 1.0).astype(np.int8)
    df["is_long_trip"] = (df["trip_distance"] > 10.0).astype(np.int8)
    df["is_same_zone"] = (df["PULocationID"] == df["DOLocationID"]).astype(np.int8)
    return df


def add_extra_time_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Finer time-of-day flags + cyclic month encoding."""
    df = df.copy()
    hour, dow = df["pickup_hour"], df["pickup_dayofweek"]
    weekday = dow < 5
    df["is_late_night"] = ((hour >= 0) & (hour < 5)).astype(np.int8)
    df["is_morning_rush"] = (weekday & hour.between(7, 9)).astype(np.int8)
    df["is_evening_rush"] = (weekday & hour.between(16, 19)).astype(np.int8)
    df["month_sin"] = np.sin(2 * np.pi * df["pickup_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["pickup_month"] / 12)
    return df


# ---------------------------------------------------------------------------
# CBD (Congestion Relief Zone) boundary crossing
# ---------------------------------------------------------------------------

def add_cbd_crossing(df: pd.DataFrame) -> pd.DataFrame:
    """Flag whether a trip crosses the true CRZ (Congestion Relief Zone) boundary.

    Requires is_cbd_pu / is_cbd_do (added by add_zone_features) — the precise
    "Manhattan at/south of 60th St" boundary, not the wider Yellow Zone (96th St).
      - crosses_cbd      : exactly one endpoint inside the CBD (commuter in/out)
      - fully_within_cbd : both endpoints inside the CBD (short intra-core trips)
    These separate the two very different pricing regimes the error analysis
    surfaced (short CBD trips vs. airport trips).
    """
    df = df.copy()
    pu_in = df["is_cbd_pu"].astype(bool)
    do_in = df["is_cbd_do"].astype(bool)
    df["crosses_cbd"] = (pu_in ^ do_in).astype(np.int8)
    df["fully_within_cbd"] = (pu_in & do_in).astype(np.int8)
    # Directional crossing — entering vs leaving the CBD priced differently
    # (crosses_cbd=1 was the highest-error / highest-variance segment).
    df["enters_cbd"] = ((~pu_in) & do_in).astype(np.int8)
    df["exits_cbd"] = (pu_in & (~do_in)).astype(np.int8)
    return df


# ---------------------------------------------------------------------------
# Congestion surcharge
# ---------------------------------------------------------------------------

def add_congestion_surcharge(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate the NYS $2.50 congestion surcharge.

    Applies when the trip starts or ends in a Manhattan Yellow Zone (south of
    96th St).  Yellow Zone service_zone label is a reliable proxy.
    """
    df = df.copy()
    is_congestion_trip = (df["is_yellow_zone_pu"] | df["is_yellow_zone_do"]).astype(bool)
    df["congestion_surcharge_est"] = is_congestion_trip.astype(float) * TLC_RULES["congestion_surcharge"]
    return df


# ---------------------------------------------------------------------------
# CBD Congestion Relief Zone fee (Jan 5, 2025 onwards)
# ---------------------------------------------------------------------------

def add_cbd_fee(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate the $0.75 MTA Congestion Relief Zone (CRZ) toll.

    Applies to trips in/through the precise CRZ (Manhattan at/south of 60th
    St — see config.CBD_ZONES, is_cbd_pu/do) that occur on or after Jan 5,
    2025. We use pickup_year >= 2025 as a practical proxy (edge-case of
    Jan 1-4 2025 is < 0.05% of the 2025 training data).
    """
    df = df.copy()
    is_post_cbd = (df["pickup_year"] >= _CBD_YEAR).astype(bool)
    is_cbd_trip = (df["is_cbd_pu"] | df["is_cbd_do"]).astype(bool)
    df["is_post_cbd"] = is_post_cbd.astype(np.int8)
    df["cbd_fee_est"] = (is_post_cbd & is_cbd_trip).astype(float) * TLC_RULES["cbd_congestion_fee"]
    return df


# ---------------------------------------------------------------------------
# Time-based surcharges
# ---------------------------------------------------------------------------

def add_time_surcharges(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate rush-hour and overnight extras plus fixed per-trip charges.

    NYC TLC extras:
      - $1.00 rush-hour surcharge:  weekdays 16:00–20:00
      - $0.50 overnight surcharge:  20:00–06:00
      - MTA tax $0.50 (all metered trips)
      - Improvement surcharge $1.00 (all metered trips)
    """
    df = df.copy()
    hour = df["pickup_hour"]
    dow = df["pickup_dayofweek"]

    is_weekday = (dow < 5)
    is_rush = (
        is_weekday
        & hour.between(TLC_RULES["rush_hour_start"], TLC_RULES["rush_hour_end"] - 1)
    )
    is_overnight = (hour >= TLC_RULES["overnight_start"]) | (hour < TLC_RULES["overnight_end"])

    df["is_rush_hour"] = is_rush.astype(np.int8)
    df["is_overnight"] = is_overnight.astype(np.int8)

    df["extra_est"] = (
        is_rush.astype(float) * TLC_RULES["extra_rush_hour"]
        + (~is_rush & is_overnight).astype(float) * TLC_RULES["extra_overnight"]
    )
    df["mta_tax_est"] = TLC_RULES["mta_tax"]
    df["improvement_surcharge_est"] = TLC_RULES["improvement_surcharge"]
    return df


# ---------------------------------------------------------------------------
# Summary feature: total estimated non-fare component
# ---------------------------------------------------------------------------

def add_estimated_charges_total(df: pd.DataFrame) -> pd.DataFrame:
    """Sum all per-trip estimated surcharges into one feature."""
    df = df.copy()
    df["estimated_surcharges"] = (
        df["airport_fee_est"]
        + df["congestion_surcharge_est"]
        + df["cbd_fee_est"]
        + df["extra_est"]
        + df["mta_tax_est"]
        + df["improvement_surcharge_est"]
    )
    return df
