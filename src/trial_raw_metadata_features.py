"""One-off trial: does adding VendorID / passenger_count / store_and_fwd_flag
help, even though none has a known causal link to fare (see the data
dictionary discussion) -- testing empirically rather than assuming, in case
any of them correlates with something else (e.g. certain vendors' fleets
skew toward specific boroughs, or store-and-forward trips correlate with
poor-connectivity areas that also tend to be longer trips).

Ablation only (current 39 SELECTED_FEATURES vs. 39+3), LGBM only, same
CV + Val rigor as prior trials in this project. Does not touch the real
2026 test set.

Run:
    python -m src.trial_raw_metadata_features [--sample 50000]
"""

import argparse
import sys

import numpy as np
import pandas as pd

from src.config import DATA_PATHS, SAMPLE_CONFIG, TARGET_COL, SELECTED_FEATURES
from src.data.loader import load_parquet_files, load_taxi_zones
from src.data.cleaner import clean_training_data
from src.features.engineer import FeatureEngineer, get_raw_input_features
from src.models.trainer import train_model
from src.train import temporal_split, forward_chain_splits

MODEL_NAME = "lgbm"
EXTRA_COLS = ["VendorID", "passenger_count", "store_and_fwd_flag_enc"]


def add_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """VendorID (categorical code, no NaN in practice), passenger_count
    (numeric, NaN -> mode of 1), store_and_fwd_flag (Y/N -> 1/0, NaN -> 0,
    the overwhelming majority class)."""
    df = df.copy()
    df["passenger_count"] = df["passenger_count"].fillna(1.0)
    df["store_and_fwd_flag_enc"] = (df["store_and_fwd_flag"] == "Y").astype(int)
    return df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=SAMPLE_CONFIG["n_per_month_train"])
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n=== Loading + cleaning training data ({args.sample:,} rows/month) ===")
    raw_df = load_parquet_files(DATA_PATHS["training"], n_per_file=args.sample)
    zones_df = load_taxi_zones(DATA_PATHS["taxi_zones"])
    clean_df = clean_training_data(raw_df)
    clean_df = add_metadata_features(clean_df)

    train_df, val_df = temporal_split(clean_df)
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}")

    baseline_cols = list(SELECTED_FEATURES)
    extended_cols = baseline_cols + EXTRA_COLS

    print("\n=== Forward-chaining CV: 39 features (baseline) vs. 42 (with raw metadata) ===")
    cv_mae_base, cv_mae_ext = [], []
    for fold, tr_df, vl_df in forward_chain_splits(train_df, n_splits=5):
        X_tr_raw, X_vl_raw = get_raw_input_features(tr_df), get_raw_input_features(vl_df)
        y_tr = tr_df[TARGET_COL].reset_index(drop=True)
        y_vl = vl_df[TARGET_COL].reset_index(drop=True)

        eng = FeatureEngineer(zones_df)
        eng.fit(X_tr_raw, y_tr, duration=tr_df["trip_duration_min"].reset_index(drop=True))
        X_tr = eng.get_tree_features(eng.transform(X_tr_raw))
        X_vl = eng.get_tree_features(eng.transform(X_vl_raw))

        # Splice in the 3 raw metadata columns (not part of FeatureEngineer's
        # candidate set -- added directly from the cleaned df, aligned by index).
        for col in EXTRA_COLS:
            X_tr[col] = tr_df[col].reset_index(drop=True).values
            X_vl[col] = vl_df[col].reset_index(drop=True).values

        base_cols = [c for c in baseline_cols if c in X_tr.columns]
        ext_cols = [c for c in extended_cols if c in X_tr.columns]

        _, m_base = train_model(MODEL_NAME, X_tr[base_cols], y_tr, X_vl[base_cols], y_vl)
        _, m_ext = train_model(MODEL_NAME, X_tr[ext_cols], y_tr, X_vl[ext_cols], y_vl)
        cv_mae_base.append(m_base["mae"])
        cv_mae_ext.append(m_ext["mae"])
        print(f"  Fold {fold + 1}: baseline MAE={m_base['mae']:.4f}  +metadata MAE={m_ext['mae']:.4f}")

    mean_cv_base = float(np.mean(cv_mae_base))
    mean_cv_ext = float(np.mean(cv_mae_ext))

    print("\n=== Val ablation: fit on full train_df, score once on val_df ===")
    X_train_raw = get_raw_input_features(train_df)
    X_val_raw = get_raw_input_features(val_df)
    y_train = train_df[TARGET_COL].reset_index(drop=True)
    y_val = val_df[TARGET_COL].reset_index(drop=True)

    eng_full = FeatureEngineer(zones_df)
    eng_full.fit(X_train_raw, y_train, duration=train_df["trip_duration_min"].reset_index(drop=True))
    X_train_feat = eng_full.get_tree_features(eng_full.transform(X_train_raw))
    X_val_feat = eng_full.get_tree_features(eng_full.transform(X_val_raw))
    for col in EXTRA_COLS:
        X_train_feat[col] = train_df[col].reset_index(drop=True).values
        X_val_feat[col] = val_df[col].reset_index(drop=True).values

    base_cols = [c for c in baseline_cols if c in X_train_feat.columns]
    ext_cols = [c for c in extended_cols if c in X_train_feat.columns]

    _, val_base = train_model(MODEL_NAME, X_train_feat[base_cols], y_train, X_val_feat[base_cols], y_val)
    _, val_ext = train_model(MODEL_NAME, X_train_feat[ext_cols], y_train, X_val_feat[ext_cols], y_val)

    print(f"\n  {'':22} {'Baseline (39)':>14} {'+metadata (42)':>16}")
    print(f"  {'CV MAE (mean)':22} {mean_cv_base:>14.4f} {mean_cv_ext:>16.4f}")
    print(f"  {'Val MAE':22} {val_base['mae']:>14.4f} {val_ext['mae']:>16.4f}")

    cv_delta = mean_cv_ext - mean_cv_base
    val_delta = val_ext["mae"] - val_base["mae"]
    print(f"\n  CV MAE delta: {cv_delta:+.4f}   Val MAE delta: {val_delta:+.4f}")
    if cv_delta <= 0 and val_delta <= 0:
        print("  RESULT: raw metadata features helped (or were free) on both CV and Val.")
    elif cv_delta > 0 and val_delta > 0:
        print("  RESULT: raw metadata features hurt on both CV and Val -- do not add.")
    else:
        print("  RESULT: mixed signal -- inspect fold-level detail above before deciding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
