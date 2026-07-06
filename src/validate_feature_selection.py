"""One-off confirmatory check: does the 39-feature set found by
feature_selection_temporal.py (trial4-protected, --sample 30000) actually
hold up at PRODUCTION training scale (150k rows/month, SAMPLE_CONFIG
default)?

This exists because of a documented precedent in this exact project (see
config.py's SELECTED_FEATURES note): a prior 30k-sample selection run found
a smaller subset that looked better, but at full 150k/month scale the
pruning hurt badly -- rare zones/routes are under-sampled at 30k, making
high-cardinality features (zone one-hots, PU/DO-derived encodings) look like
noise when they aren't at full scale. That selection was reverted. The
current 39-feature set drops exactly that category of feature (all
pu_top_zone_XXX one-hots, zone popularity/fare-std, borough/service-zone
encodings), so it needs to be re-checked at full scale before trusting it,
not assumed safe by analogy to the smaller-sample result alone.

No re-sweep of K here -- just a direct ablation (full 147 vs. this specific
39-feature set) via forward-chaining CV MAE + Val MAE, at production scale.

Run:
    python -m src.validate_feature_selection
"""

import sys

import numpy as np

from src.config import DATA_PATHS, SAMPLE_CONFIG, TARGET_COL
from src.data.loader import load_parquet_files, load_taxi_zones
from src.data.cleaner import clean_training_data
from src.features.engineer import FeatureEngineer, get_raw_input_features
from src.models.trainer import train_model
from src.train import temporal_split, forward_chain_splits

MODEL_NAME = "lgbm"

SELECTED_39 = [
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
]


def main():
    print(f"\n=== Loading + cleaning training data (production scale, "
          f"{SAMPLE_CONFIG['n_per_month_train']:,} rows/month) ===")
    raw_df = load_parquet_files(DATA_PATHS["training"], n_per_file=SAMPLE_CONFIG["n_per_month_train"])
    zones_df = load_taxi_zones(DATA_PATHS["taxi_zones"])
    clean_df = clean_training_data(raw_df)

    train_df, val_df = temporal_split(clean_df)
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}")

    print("\n=== Forward-chaining CV: full (147) vs selected (39) ===")
    cv_mae_full, cv_mae_selected = [], []
    for fold, tr_df, vl_df in forward_chain_splits(train_df, n_splits=5):
        X_tr_raw, X_vl_raw = get_raw_input_features(tr_df), get_raw_input_features(vl_df)
        y_tr = tr_df[TARGET_COL].reset_index(drop=True)
        y_vl = vl_df[TARGET_COL].reset_index(drop=True)

        eng = FeatureEngineer(zones_df)
        eng.fit(X_tr_raw, y_tr, duration=tr_df["trip_duration_min"].reset_index(drop=True))
        X_tr = eng.get_tree_features(eng.transform(X_tr_raw))
        X_vl = eng.get_tree_features(eng.transform(X_vl_raw))
        selected = [c for c in SELECTED_39 if c in X_tr.columns]

        _, m_full = train_model(MODEL_NAME, X_tr, y_tr, X_vl, y_vl)
        _, m_sel = train_model(MODEL_NAME, X_tr[selected], y_tr, X_vl[selected], y_vl)
        cv_mae_full.append(m_full["mae"])
        cv_mae_selected.append(m_sel["mae"])
        print(f"  Fold {fold + 1}: full MAE={m_full['mae']:.4f}  selected MAE={m_sel['mae']:.4f}")

    mean_cv_full = float(np.mean(cv_mae_full))
    mean_cv_selected = float(np.mean(cv_mae_selected))

    print("\n=== Val ablation: fit on full train_df, score once on val_df ===")
    X_train_raw = get_raw_input_features(train_df)
    X_val_raw = get_raw_input_features(val_df)
    y_train = train_df[TARGET_COL].reset_index(drop=True)
    y_val = val_df[TARGET_COL].reset_index(drop=True)

    eng_full = FeatureEngineer(zones_df)
    eng_full.fit(X_train_raw, y_train, duration=train_df["trip_duration_min"].reset_index(drop=True))
    X_train_feat = eng_full.get_tree_features(eng_full.transform(X_train_raw))
    X_val_feat = eng_full.get_tree_features(eng_full.transform(X_val_raw))
    selected = [c for c in SELECTED_39 if c in X_train_feat.columns]

    _, val_full = train_model(MODEL_NAME, X_train_feat, y_train, X_val_feat, y_val)
    _, val_sel = train_model(MODEL_NAME, X_train_feat[selected], y_train, X_val_feat[selected], y_val)

    print(f"\n  {'':22} {'Full (147)':>12} {'Selected (39)':>14}")
    print(f"  {'CV MAE (mean)':22} {mean_cv_full:>12.4f} {mean_cv_selected:>14.4f}")
    print(f"  {'Val MAE':22} {val_full['mae']:>12.4f} {val_sel['mae']:>14.4f}")

    cv_ok = mean_cv_selected <= mean_cv_full * 1.005
    val_ok = val_sel["mae"] <= val_full["mae"] * 1.005
    print(f"\n  CV: {'HOLDS' if cv_ok else 'REGRESSES'} at production scale.")
    print(f"  Val: {'HOLDS' if val_ok else 'REGRESSES'} at production scale.")
    if cv_ok and val_ok:
        print("\n  RESULT: SUCCESS -- the 39-feature selection holds at production scale.")
        return 0
    else:
        print("\n  RESULT: FAILURE -- selection does NOT hold at production scale "
              "(likely the documented 30k-sample under-sampling issue). Do not commit.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
