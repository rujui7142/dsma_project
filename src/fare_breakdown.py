"""Customer-facing fare breakdown: decompose a single predicted fare into
the rate-card components that make it up, rather than showing one opaque
number.

The engineered feature set already computes every TLC rate-card component
as its own column (add_metered_fare_estimate, add_estimated_charges_total,
etc. in src/features/domain.py) -- they just get summed into
est_metered_fare for the model and mostly dropped from SELECTED_FEATURES
because they add no predictive signal on top of each other. That's
irrelevant here: for a receipt-style breakdown we want every deterministic
component, whether or not the model needed it to predict well.

Structure returned for each trip:
  base_or_flat_fare       -- distance-based meter estimate, or the flat
                             $70 JFK<->Manhattan rate when it applies
  extra                   -- rush-hour / overnight surcharge
  mta_tax                 -- fixed $0.50
  improvement_surcharge   -- fixed $1.00
  congestion_surcharge    -- Manhattan congestion zone fee (2025+)
  cbd_fee                 -- Congestion Relief Zone fee (2025+)
  airport_fee             -- flat airport pickup fee
  lga_surcharge           -- LaGuardia-route surcharge
  ewr_surcharge           -- Newark-route surcharge
  rate_card_subtotal      -- sum of the above (== est_metered_fare)
  learned_adjustment      -- model_prediction - rate_card_subtotal: the
                             data-driven correction for route/demand
                             patterns the rate card alone can't capture
  adjustment_label        -- "Discount" (adjustment < 0), "Premium"
                             (adjustment > 0), or "No adjustment"
  adjustment_reason       -- human-readable driver of that adjustment,
                             from per-trip TreeSHAP contributions
  predicted_total         -- the model's actual fare prediction (reconciles
                             exactly: rate_card_subtotal + learned_adjustment)

The adjustment is never shown as a bare, unexplained number: its sign
picks a customer-legible label (Discount / Premium) and its reason is the
single largest TreeSHAP-contributing feature group behind it (route
history, time-of-day demand, distance-curve effects, etc.) -- excluding
the features that are already shown as their own rate-card line items,
so nothing is explained twice.

Run:
    python -m src.fare_breakdown [--tag metadata-final]
"""

import argparse
import sys
from typing import Dict

import numpy as np
import pandas as pd

from src.data.loader import load_parquet_files, load_taxi_zones
from src.data.cleaner import clean_test_data
from src.config import DATA_PATHS
from src.features.engineer import get_raw_input_features
from src.models.registry import load_inference_artifacts
from src.models.shap_analysis import tree_shap_values

COMPONENT_COLS = {
    "extra": "extra_est",
    "mta_tax": "mta_tax_est",
    "improvement_surcharge": "improvement_surcharge_est",
    "congestion_surcharge": "congestion_surcharge_est",
    "cbd_fee": "cbd_fee_est",
    "airport_fee": "airport_fee_est",
    "lga_surcharge": "lga_surcharge_est",
    "ewr_surcharge": "ewr_surcharge_est",
}

# Model-input features NOT already shown as a rate-card line item, grouped
# into customer-legible "reason" buckets for adjustment justification.
# Features tied to a *_est fee or the flat/base fare itself are deliberately
# left out here -- they're already on the receipt, so attributing the
# adjustment to them too would explain the same dollar twice.
FEATURE_TO_BUCKET = {
    "route_mean_fare": "route_history",
    "route_mean_duration_min": "route_history",
    "zone_manhattan_distance": "geo_distance",
    "zone_euclidean_distance": "geo_distance",
    "distance_sq": "distance_curve",
    "log_distance": "distance_curve",
    "hour_sin": "time_of_day",
    "hour_cos": "time_of_day",
    "pickup_hour": "time_of_day",
    "hour_x_distance": "time_of_day",
    "dow_sin": "day_of_week",
    "dow_cos": "day_of_week",
    "pickup_dayofweek": "day_of_week",
    "pickup_month": "seasonal",
    "distance_x_manhattan": "route_pattern",
    "distance_x_cross_borough": "route_pattern",
    "PULocationID": "zone_pattern",
    "DOLocationID": "zone_pattern",
    "VendorID": "vendor_pattern",
    "passenger_count": "group_size",
    "store_and_fwd_flag_enc": "connectivity_pattern",
}

BUCKET_PHRASES = {
    "route_history": "typical historical fares and durations for this exact pickup-dropoff route",
    "geo_distance": "the actual geographic distance between the pickup and dropoff areas",
    "distance_curve": "how fares scale over longer or shorter trips",
    "time_of_day": "typical demand at this time of day",
    "day_of_week": "typical demand patterns for this day of the week",
    "seasonal": "seasonal fare patterns for this time of year",
    "route_pattern": "typical patterns for this type of cross-area trip",
    "zone_pattern": "historical fare patterns for these specific pickup/dropoff zones",
    "vendor_pattern": "patterns associated with this taxi vendor's fleet",
    "group_size": "patterns associated with the number of passengers",
    "connectivity_pattern": "dispatch/connectivity patterns for this trip",
}
DEFAULT_REASON = "overall route and timing patterns learned from historical trips"


def _adjustment_reason(shap_row: np.ndarray, feature_names: list) -> str:
    bucket_totals: Dict[str, float] = {}
    for val, name in zip(shap_row, feature_names):
        bucket = FEATURE_TO_BUCKET.get(name)
        if bucket:
            bucket_totals[bucket] = bucket_totals.get(bucket, 0.0) + val
    if not bucket_totals:
        return DEFAULT_REASON
    top_bucket = max(bucket_totals, key=lambda b: abs(bucket_totals[b]))
    return BUCKET_PHRASES[top_bucket]


def explain_fares(raw_trips: pd.DataFrame, engineer, model, scaler=None) -> pd.DataFrame:
    """Given raw trip rows (RAW_INPUT_COLS schema), return a per-trip
    fare breakdown DataFrame that reconciles exactly to the model's
    predicted total, with the adjustment justified rather than left as
    a bare unexplained number."""
    X_eng = engineer.transform(raw_trips)
    X_feat = engineer.get_tree_features(X_eng)

    model_name = type(model).__name__
    if "Ridge" in model_name and scaler is not None:
        y_pred = model.predict(scaler.transform(X_feat.values))
    else:
        y_pred = model.predict(X_feat)

    is_flat = X_eng["is_jfk_manhattan_flat_route"].astype(bool).reset_index(drop=True)
    rate_card_subtotal = X_eng["est_metered_fare"].reset_index(drop=True)
    base_or_flat = rate_card_subtotal - X_eng["estimated_surcharges"].reset_index(drop=True)

    out = pd.DataFrame({"base_or_flat_fare": base_or_flat})
    out["fare_type"] = pd.Series(is_flat).map({True: "JFK<->Manhattan flat rate", False: "Metered (base + distance)"})
    for label, col in COMPONENT_COLS.items():
        out[label] = X_eng[col].reset_index(drop=True)
    out["rate_card_subtotal"] = rate_card_subtotal
    out["predicted_total"] = y_pred
    out["learned_adjustment"] = out["predicted_total"] - out["rate_card_subtotal"]
    out["adjustment_label"] = out["learned_adjustment"].apply(
        lambda v: "Discount" if v < -0.005 else ("Premium" if v > 0.005 else "No adjustment")
    )

    shap_mat = tree_shap_values(model, X_feat)
    feature_names = list(X_feat.columns)
    if shap_mat is not None:
        out["adjustment_reason"] = [_adjustment_reason(row, feature_names) for row in shap_mat]
    else:
        out["adjustment_reason"] = DEFAULT_REASON
    return out


def _print_breakdown(row: pd.Series, idx: int):
    print(f"\n{'=' * 60}\nTrip {idx}  ({row['fare_type']})\n{'=' * 60}")
    label = "Flat JFK<->Manhattan rate" if "flat" in row["fare_type"] else "Base + distance fare"
    print(f"  {label:28} ${row['base_or_flat_fare']:>6.2f}")
    for label in COMPONENT_COLS:
        val = row[label]
        if val:
            print(f"  {label.replace('_', ' ').title():28} ${val:>6.2f}")
    print(f"  {'-' * 42}")
    print(f"  {'Rate-card subtotal':28} ${row['rate_card_subtotal']:>6.2f}")
    if row["adjustment_label"] != "No adjustment":
        sign = "-" if row["adjustment_label"] == "Discount" else "+"
        print(f"  {row['adjustment_label']:28} ${sign}{abs(row['learned_adjustment']):>5.2f}")
        print(f"    -> based on {row['adjustment_reason']}")
    print(f"  {'=' * 42}")
    print(f"  {'PREDICTED PRICE':28} ${row['predicted_total']:>6.2f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="metadata-final")
    p.add_argument("--n", type=int, default=6, help="number of example trips to show")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n=== Loading artifacts (tag='{args.tag}') ===")
    engineer, scaler, model = load_inference_artifacts(run_tag=args.tag)

    print("\n=== Loading a few real trips (2026 test set) for demonstration ===")
    zones_df = load_taxi_zones(DATA_PATHS["taxi_zones"])
    raw_df = load_parquet_files(DATA_PATHS["test"], n_per_file=2000)
    test_df = clean_test_data(raw_df)

    X_raw_all = get_raw_input_features(test_df)
    X_eng_all = engineer.transform(X_raw_all)

    # Hand-pick a mix of trip types so the breakdown demo shows every
    # component at least once: a plain trip, a JFK flat-rate trip, an
    # LGA/EWR route, and a rush-hour trip.
    picks = []
    flat_idx = X_eng_all.index[X_eng_all["is_jfk_manhattan_flat_route"] == 1]
    lga_idx = X_eng_all.index[X_eng_all.get("is_lga_route", 0) == 1] if "is_lga_route" in X_eng_all else []
    rush_idx = X_eng_all.index[X_eng_all.get("is_rush_hour", 0) == 1] if "is_rush_hour" in X_eng_all else []
    plain_idx = X_eng_all.index[
        (X_eng_all["is_jfk_manhattan_flat_route"] == 0)
        & (X_eng_all.get("is_airport_route", 0) == 0)
    ]
    for pool in (flat_idx, lga_idx, rush_idx, plain_idx, plain_idx, plain_idx):
        if len(pool) > 0:
            picks.append(pool[0])
    picks = list(dict.fromkeys(picks))[: args.n]  # de-dup, preserve order

    sample_raw = X_raw_all.loc[picks]
    breakdown = explain_fares(sample_raw, engineer, model, scaler)

    print(f"\n=== Fare breakdown for {len(breakdown)} example trips ===")
    for i, (_, row) in enumerate(breakdown.iterrows(), start=1):
        _print_breakdown(row, i)
    return 0


if __name__ == "__main__":
    sys.exit(main())
