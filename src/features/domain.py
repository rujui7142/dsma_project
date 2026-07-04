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
    WEST_VILLAGE_ZONES, HOTSPOT_ZONES,
)

_JFK = TLC_RULES["jfk_zone_id"]
_LGA = TLC_RULES["lga_zone_id"]
_EWR = TLC_RULES["ewr_zone_id"]
_CBD_YEAR = TLC_RULES["cbd_start_year"]
_WEST_VILLAGE = set(WEST_VILLAGE_ZONES)
_HOTSPOTS = set(HOTSPOT_ZONES)


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
# CBD (Congestion Relief Zone) boundary crossing
# ---------------------------------------------------------------------------

def add_cbd_crossing(df: pd.DataFrame) -> pd.DataFrame:
    """Flag whether a trip crosses the Manhattan Yellow Zone / CBD boundary.

    Requires is_yellow_zone_pu / is_yellow_zone_do (added by add_zone_features).
      - crosses_cbd      : exactly one endpoint inside the CBD (commuter in/out)
      - fully_within_cbd : both endpoints inside the CBD (short intra-core trips)
    These separate the two very different pricing regimes the error analysis
    surfaced (short CBD trips vs. airport trips).
    """
    df = df.copy()
    pu_in = df["is_yellow_zone_pu"].astype(bool)
    do_in = df["is_yellow_zone_do"].astype(bool)
    df["crosses_cbd"] = (pu_in ^ do_in).astype(np.int8)
    df["fully_within_cbd"] = (pu_in & do_in).astype(np.int8)
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
    """Estimate the $9.00 MTA Congestion Relief Zone fee.

    Applies to trips in/through Manhattan (Yellow Zone) that occur on or after
    Jan 5, 2025.  We use pickup_year >= 2025 as a practical proxy (edge-case
    of Jan 1–4 2025 is < 0.05% of the 2025 training data).
    """
    df = df.copy()
    is_post_cbd = (df["pickup_year"] >= _CBD_YEAR).astype(bool)
    is_manhattan_trip = (df["is_yellow_zone_pu"] | df["is_yellow_zone_do"]).astype(bool)
    df["is_post_cbd"] = is_post_cbd.astype(np.int8)
    df["cbd_fee_est"] = (is_post_cbd & is_manhattan_trip).astype(float) * TLC_RULES["cbd_congestion_fee"]
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
