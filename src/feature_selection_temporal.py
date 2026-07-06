"""Temporal feature selection: find the smallest feature set that doesn't
hurt EITHER cross-validated or near-future (Val) generalization -- gated by
real ablation evidence at every step, not a fixed importance threshold.

First attempt (see git history) flagged features via fixed thresholds (mean
SHAP share, and a trend score for declining usefulness -- see below), then
batch-removed everything flagged in one shot. That was too blunt: it flagged
95/147 features, including trip_distance (13% mean importance -- one of the
strongest features in the model, just DECLINING in relative share due to
collinearity with distance_sq/log_distance/the geo-distance features) purely
because its trend was negative, ignoring that its absolute importance was
still substantial. Batch-ablating all 95 at once regressed both CV and Val
MAE -- correctly rejected, but uninformative about which subset of the 95
was actually the problem.

Redesigned as an incremental, evidence-gated sweep instead:
  1. Split off Val (Nov-Dec 2025, same VAL_YEARS_MONTHS convention as
     train.py) from the rest of the training data. Forward-chaining CV (5
     folds) runs ONLY within the remaining train_df -- Val is held out from
     CV entirely. The real 2026 test set is never touched here.
  2. Per-fold TreeSHAP importance (LGBM) on the full candidate feature set,
     cached once (feature engineering doesn't need to be redone per K --
     only which COLUMNS are selected changes). For each feature: mean_pct
     (average importance share across folds) and trend_score
     ((late-fold avg - early-fold avg) / mean_pct) -- reported for context,
     NOT used to trigger removal on its own.
  3. Rank ALL features by mean_pct descending. Sweep K (features kept) over
     a grid from small to the full candidate count, computing BOTH mean CV
     MAE and Val MAE at each K using the cached matrices (just column
     subsetting -- no re-fitting FeatureEngineer per K).
  4. Pick the smallest K within --tol of the best observed CV MAE AND the
     best observed Val MAE (both must hold) -- the cut stops exactly where
     removal starts to actually hurt, rather than pre-committing to a
     magic-number importance floor.
  5. Report which surviving features still show a declining trend (a watch
     list, not evidence to remove them -- their absolute importance earned
     their spot in the ablation-gated sweep).

Trial run scoped to LGBM only, to check whether this method is worthwhile
before extending it to all 4 models.

Run:
    python -m src.feature_selection_temporal [--sample 30000] [--tol 0.005] [--tag trial]
"""

import argparse
import sys
from typing import List

import numpy as np
import pandas as pd

from src.config import DATA_PATHS, TARGET_COL
from src.data.loader import load_parquet_files, load_taxi_zones
from src.data.cleaner import clean_training_data
from src.features.engineer import FeatureEngineer, get_raw_input_features
from src.models.trainer import train_model
from src.models.shap_analysis import shap_importance, shap_importance_over_folds
from src.train import temporal_split, forward_chain_splits

MODEL_NAME = "lgbm"  # trial scope: LGBM only
K_GRID_STEP = 10      # grid resolution below the full feature count

# Features exempt from removal regardless of aggregate SHAP ranking. All of
# these were purpose-built earlier this session to fix the worst-performing
# SEGMENT (20+mi/JFK bucket, MAE 28 -> confirmed fixed, e.g. is_jfk_manhattan_
# flat_route dropped JFK<->Manhattan MAE from ~4.3 to 1.98), not to move the
# aggregate metric -- the segment is a tiny fraction of total volume, so an
# aggregate-CV/Val-MAE-only sweep has no way to "see" their value and will
# rank them low. mta_tax_est/improvement_surcharge_est are deliberately NOT
# protected: they're literally constant across every row (flat $0.50/$1.00
# fees), so a tree can never split on them -- removing those is correct, not
# a segment-blindness casualty.
PROTECTED_FEATURES = [
    "is_jfk_manhattan_flat_route", "distance_x_jfk_flat",
    "is_jfk_pu", "is_jfk_do", "is_lga_pu", "is_lga_do", "is_lga_route",
    "is_airport_route", "is_airport_pickup",
    "airport_fee_est", "lga_surcharge_est", "ewr_surcharge_est",
    "cbd_fee_est", "is_post_cbd", "congestion_surcharge_est",
    "is_legal_holiday",
    "is_outside_nyc_pu", "is_outside_nyc_do", "is_outside_nyc_route",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=30_000)
    p.add_argument("--shap-sample", type=int, default=5000)
    p.add_argument("--tol", type=float, default=0.005,
                   help="max relative MAE loss vs best when picking the smallest K")
    p.add_argument("--tag", type=str, default="trial")
    return p.parse_args()


def _trend_score(row: pd.Series, fold_cols: List[str]) -> float:
    """(late-fold avg - early-fold avg) / mean_pct. Negative = declining."""
    early = row[fold_cols[:2]].mean()
    late = row[fold_cols[-2:]].mean()
    mean_pct = row["mean_pct"]
    if mean_pct <= 0:
        return 0.0
    return (late - early) / mean_pct


def main():
    args = parse_args()

    print("\n=== Loading + cleaning training data ===")
    raw_df = load_parquet_files(DATA_PATHS["training"], n_per_file=args.sample)
    zones_df = load_taxi_zones(DATA_PATHS["taxi_zones"])
    clean_df = clean_training_data(raw_df)

    train_df, val_df = temporal_split(clean_df)
    print(f"  Train: {len(train_df):,}  Val (held out from CV): {len(val_df):,}")

    # ------------------------------------------------------------------
    # Stage 1: forward-chaining CV within train_df, per-fold SHAP profiling.
    # Cache each fold's engineered matrices -- reused across the whole K
    # sweep below, so feature engineering only happens once per fold.
    # ------------------------------------------------------------------
    print("\n=== Stage 1: forward-chaining CV + per-fold SHAP (LGBM) ===")
    fold_labels, fold_importances, fold_cache = [], {}, []
    cv_mae_full = []
    all_features = None

    for fold, tr_df, vl_df in forward_chain_splits(train_df, n_splits=5):
        label = f"F{fold + 1}"
        fold_labels.append(label)
        X_tr_raw, X_vl_raw = get_raw_input_features(tr_df), get_raw_input_features(vl_df)
        y_tr = tr_df[TARGET_COL].reset_index(drop=True)
        y_vl = vl_df[TARGET_COL].reset_index(drop=True)

        eng = FeatureEngineer(zones_df)
        eng.fit(X_tr_raw, y_tr, duration=tr_df["trip_duration_min"].reset_index(drop=True))
        X_tr = eng.get_tree_features(eng.transform(X_tr_raw))
        X_vl = eng.get_tree_features(eng.transform(X_vl_raw))
        all_features = list(X_tr.columns)
        fold_cache.append((X_tr, y_tr, X_vl, y_vl))

        model, metrics = train_model(MODEL_NAME, X_tr, y_tr, X_vl, y_vl)
        cv_mae_full.append(metrics["mae"])

        shap_X = X_vl.sample(min(args.shap_sample, len(X_vl)), random_state=42)
        fold_importances[label] = shap_importance(model, shap_X, list(shap_X.columns))
        print(f"  {label}: val MAE={metrics['mae']:.4f}")

    shap_wide = shap_importance_over_folds(fold_importances)
    shap_wide["trend_score"] = shap_wide.apply(lambda r: _trend_score(r, fold_labels), axis=1)

    n_feat = len(all_features)
    mean_cv_mae_full = float(np.mean(cv_mae_full))
    print(f"\n  Full feature set ({n_feat} features): mean CV MAE = {mean_cv_mae_full:.4f}")

    # Rank by mean SHAP share, most important first. Features SHAP never saw
    # (e.g. absent from a fold's engineered matrix) go last.
    ranking = [f for f in shap_wide.sort_values("mean_pct", ascending=False).index if f in all_features]
    ranking += [f for f in all_features if f not in ranking]

    # Protected features go first, regardless of SHAP rank -- guarantees
    # they're included at every K in the sweep below (see PROTECTED_FEATURES
    # docstring for why aggregate SHAP importance can't be trusted for them).
    protected_present = [f for f in PROTECTED_FEATURES if f in ranking]
    ranking = protected_present + [f for f in ranking if f not in protected_present]
    print(f"\n  Protected features (always kept): {len(protected_present)} -- {protected_present}")

    # ------------------------------------------------------------------
    # Cache the outer train_df -> val_df split once too (fit on the FULL
    # train_df, scored on Val -- separate from the 5 CV folds above).
    # ------------------------------------------------------------------
    X_train_raw = get_raw_input_features(train_df)
    X_val_raw = get_raw_input_features(val_df)
    y_train = train_df[TARGET_COL].reset_index(drop=True)
    y_val = val_df[TARGET_COL].reset_index(drop=True)

    eng_full = FeatureEngineer(zones_df)
    eng_full.fit(X_train_raw, y_train, duration=train_df["trip_duration_min"].reset_index(drop=True))
    X_train_feat = eng_full.get_tree_features(eng_full.transform(X_train_raw))
    X_val_feat = eng_full.get_tree_features(eng_full.transform(X_val_raw))

    _, metrics_full_val = train_model(MODEL_NAME, X_train_feat, y_train, X_val_feat, y_val)
    val_mae_full = metrics_full_val["mae"]
    print(f"  Full feature set: Val MAE = {val_mae_full:.4f}")

    # ------------------------------------------------------------------
    # Stage 2: incremental K sweep, gated by BOTH CV MAE and Val MAE.
    # ------------------------------------------------------------------
    print("\n=== Stage 2: incremental K sweep (CV MAE + Val MAE, both must hold) ===")
    # Smallest K tested must be >= the protected count, or an early grid
    # point would silently exclude some protected features rather than
    # testing "protected + N top-ranked extras" as intended.
    min_k = max(K_GRID_STEP, len(protected_present))
    grid = sorted({k for k in list(range(min_k, n_feat, K_GRID_STEP)) + [min_k, n_feat]})
    sweep = []
    for k in grid:
        cols = ranking[:k]
        cv_maes = []
        for X_tr, y_tr, X_vl, y_vl in fold_cache:
            # Top-zone one-hot columns are learned PER FOLD (each fold's
            # FeatureEngineer picks its own top-N most frequent pickup
            # zones), so a column present in `ranking` (built from the
            # union of all folds' SHAP tables) isn't guaranteed to exist in
            # every individual fold's cached matrix -- same guard
            # select_features.py's _cv_rmse already uses for this reason.
            use = [c for c in cols if c in X_tr.columns]
            _, metrics = train_model(MODEL_NAME, X_tr[use], y_tr, X_vl[use], y_vl)
            cv_maes.append(metrics["mae"])
        mean_cv_mae = float(np.mean(cv_maes))

        use_val = [c for c in cols if c in X_train_feat.columns]
        _, metrics_val = train_model(MODEL_NAME, X_train_feat[use_val], y_train, X_val_feat[use_val], y_val)
        val_mae = metrics_val["mae"]

        sweep.append({"k": k, "cv_mae": mean_cv_mae, "val_mae": val_mae})
        print(f"  K={k:>3}  CV MAE={mean_cv_mae:.4f}  Val MAE={val_mae:.4f}")

    sweep_df = pd.DataFrame(sweep)
    best_cv_mae = sweep_df["cv_mae"].min()
    best_val_mae = sweep_df["val_mae"].min()
    cv_threshold = best_cv_mae * (1.0 + args.tol)
    val_threshold = best_val_mae * (1.0 + args.tol)

    ok = sweep_df[(sweep_df["cv_mae"] <= cv_threshold) & (sweep_df["val_mae"] <= val_threshold)]
    best_k = int(ok["k"].min()) if len(ok) > 0 else n_feat
    selected = ranking[:best_k]
    removed = ranking[best_k:]

    print(f"\n  Best CV MAE={best_cv_mae:.4f} (tol {args.tol:.1%} -> {cv_threshold:.4f})")
    print(f"  Best Val MAE={best_val_mae:.4f} (tol {args.tol:.1%} -> {val_threshold:.4f})")
    print(f"  Selected K={best_k}  ({best_k}/{n_feat} features kept, {len(removed)} removed)")

    # ------------------------------------------------------------------
    # Report: surviving features with a declining trend (watch list, not
    # removal evidence -- their absolute importance earned their spot).
    # ------------------------------------------------------------------
    watch_list = shap_wide.loc[[f for f in selected if f in shap_wide.index]]
    watch_list = watch_list[watch_list["trend_score"] < -0.5].sort_values("trend_score")
    if len(watch_list) > 0:
        print(f"\n  Watch list -- kept (importance earned their spot) but trend is declining:")
        for feat, row in watch_list.iterrows():
            print(f"    {feat:<35} mean_pct={row['mean_pct']:.3f}  trend={row['trend_score']:.2f}")

    print(f"\n  Removed features ({len(removed)}): {removed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
