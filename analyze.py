"""Validation & drift analysis pipeline — train/validation data only.

This script NEVER touches the test set (2026). It runs the forward-chaining CV
and, across all four models and every fold, produces the analysis the lecturer
asked for:

  Step 2  Segmented error analysis across all models  -> weak-segment tables + plots
          (drives "which features need more EDA / engineering")
  Step 4  TreeSHAP feature importance per fold         -> predictive value over time
  Drift   Per-fold feature drift (PSI, fixed reference) + per-model MAE over folds

All artifacts are saved to outputs/plots + logs/ and logged to W&B so results
can drive the model/feature adjustments in step 3.

Run:
    python analyze.py [--sample N] [--no-wandb] [--tag analysis] [--shap-sample N]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_PATHS, SAMPLE_CONFIG, TARGET_COL, LOGS_DIR, WANDB_PROJECT,
)
from src.data.loader import load_parquet_files, load_taxi_zones
from src.data.cleaner import clean_training_data
from src.features.engineer import FeatureEngineer, get_raw_input_features
from src.models.trainer import train_model, build_ridge_scaler
from src.models.evaluator import (
    error_analysis, aggregate_segment_errors, plot_segment_error,
    plot_metric_over_folds,
)
from src.models.shap_analysis import (
    shap_importance, shap_importance_over_folds, plot_shap_bar, plot_shap_over_time,
)
from src.drift.detector import detect_feature_drift, plot_feature_drift_over_folds
from src.tracking.wandb_tracker import ExperimentTracker
from train import forward_chain_splits


_ALL_MODELS = ("lgbm", "xgb", "rf", "ridge")

# Segment dimensions to break error down by (must exist on the engineered df).
SEGMENT_DIMS = [
    "pu_borough", "distance_bin", "pickup_hour",
    "is_airport_route", "is_hotspot_route", "crosses_cbd",
]

OUTPUTS_PLOTS = Path("outputs/plots")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=SAMPLE_CONFIG["n_per_month_train"])
    p.add_argument("--shap-sample", type=int, default=3000,
                   help="rows sampled per fold for TreeSHAP (keeps it fast)")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--tag", type=str, default="analysis")
    return p.parse_args()


def main():
    args = parse_args()
    OUTPUTS_PLOTS.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load train/val data only (never the test set)
    # ------------------------------------------------------------------
    print("\n=== Loading training data (train/val only) ===")
    raw_df = load_parquet_files(
        DATA_PATHS["training"], n_per_file=args.sample,
        random_state=SAMPLE_CONFIG["random_state"],
    )
    zones_df = load_taxi_zones(DATA_PATHS["taxi_zones"])
    clean_df = clean_training_data(raw_df)
    print(f"  Clean shape: {clean_df.shape}")

    # ------------------------------------------------------------------
    # Forward-chaining CV: per fold, per model
    # ------------------------------------------------------------------
    print("\n=== Forward-chaining analysis ===")
    fold_labels = []
    # metrics[model] = [mae per fold]; rmse likewise
    mae_over_folds = {m: [] for m in _ALL_MODELS}
    rmse_over_folds = {m: [] for m in _ALL_MODELS}
    # segment_errors[model][dim] = [per-fold segment DataFrames]
    segment_errors = {m: {d: [] for d in SEGMENT_DIMS} for m in _ALL_MODELS}
    # per-fold stored context for post-hoc SHAP
    fold_models = {}          # fold -> {model_name: fitted_model}
    fold_shap_X = {}          # fold -> subsampled X_val_feat for SHAP
    feature_names = None

    # Fixed drift reference (first fold's engineer + train distribution)
    ref_engineer = None
    ref_feat = None
    drift_rows = []           # tidy (fold, feature, psi)

    for fold, tr_df, vl_df in forward_chain_splits(clean_df, n_splits=5):
        fold_label = f"F{fold + 1}"
        fold_labels.append(fold_label)

        X_tr_raw = get_raw_input_features(tr_df)
        X_vl_raw = get_raw_input_features(vl_df)
        y_tr = tr_df[TARGET_COL].reset_index(drop=True)
        y_vl = vl_df[TARGET_COL].reset_index(drop=True)

        engineer = FeatureEngineer(zones_df)
        engineer.fit(X_tr_raw, y_tr)
        X_tr_eng = engineer.transform(X_tr_raw)
        X_vl_eng = engineer.transform(X_vl_raw)
        X_tr_feat = engineer.get_tree_features(X_tr_eng)
        X_vl_feat = engineer.get_tree_features(X_vl_eng)
        feature_names = list(X_tr_feat.columns)

        scaler = build_ridge_scaler(X_tr_feat)

        # Capture fixed reference on the first fold
        if ref_engineer is None:
            ref_engineer = engineer
            ref_feat = X_tr_feat

        fold_models[fold_label] = {}
        fold_shap_X[fold_label] = X_vl_feat.sample(
            min(args.shap_sample, len(X_vl_feat)),
            random_state=SAMPLE_CONFIG["random_state"],
        )

        for name in _ALL_MODELS:
            model, metrics = train_model(
                name, X_tr_feat, y_tr, X_vl_feat, y_vl, scaler=scaler,
            )
            fold_models[fold_label][name] = model
            mae_over_folds[name].append(metrics["mae"])
            rmse_over_folds[name].append(metrics["rmse"])

            # Predictions for segmented error analysis
            if name == "ridge":
                y_pred = model.predict(scaler.transform(X_vl_feat.values))
            else:
                y_pred = model.predict(X_vl_feat)

            analyses = error_analysis(y_vl, y_pred, X_vl_eng)
            for dim in SEGMENT_DIMS:
                if dim in analyses:
                    seg = analyses[dim].copy()
                    seg["fold"] = fold_label
                    segment_errors[name][dim].append(seg)

        # ---- feature drift vs fixed reference ----
        # Use the reference engineer's columns so drift is measured on a
        # consistent feature basis (one-hot zone names vary per fold).
        curr_feat = ref_engineer.transform(X_vl_raw)
        curr_feat = ref_engineer.get_tree_features(curr_feat)
        fd = detect_feature_drift(ref_feat, curr_feat, list(ref_feat.columns))
        for _, r in fd.iterrows():
            drift_rows.append({"fold": fold_label, "feature": r["feature"], "psi": r["psi"]})

    # ------------------------------------------------------------------
    # Pick CV champion (lowest mean RMSE across folds)
    # ------------------------------------------------------------------
    mean_rmse = {m: float(np.mean(rmse_over_folds[m])) for m in _ALL_MODELS}
    champion = min(mean_rmse, key=mean_rmse.get)
    print("\n=== CV summary (mean RMSE over folds) ===")
    for m in sorted(mean_rmse, key=mean_rmse.get):
        marker = "  <-- champion" if m == champion else ""
        print(f"  {m:<6} RMSE={mean_rmse[m]:.4f}  MAE={np.mean(mae_over_folds[m]):.4f}{marker}")

    figures = {}
    tables = {}

    # ------------------------------------------------------------------
    # Step 2 — segmented error analysis (champion) across folds
    # ------------------------------------------------------------------
    print("\n=== Step 2: CV error analysis (champion = %s) ===" % champion)
    weak_segments = []
    for dim in SEGMENT_DIMS:
        agg = aggregate_segment_errors(segment_errors[champion][dim], dim)
        if agg.empty:
            continue
        tables[f"cv_error_by_{dim}"] = agg
        agg.to_csv(LOGS_DIR / f"cv_error_by_{dim}_{args.tag}.csv", index=False)
        fig = plot_segment_error(
            agg, dim, output_dir=str(OUTPUTS_PLOTS),
            filename=f"cv_error_by_{dim}_{args.tag}.png",
            title=f"CV error by {dim} — {champion}",
        )
        if fig is not None:
            figures[f"cv_error_by_{dim}"] = fig
        # worst segment on this dimension
        worst = agg.iloc[0]
        weak_segments.append({
            "dimension": dim, "worst_segment": worst[dim],
            "mae_mean": worst["mae_mean"], "avg_count": worst["avg_count"],
        })

    weak_df = pd.DataFrame(weak_segments).sort_values("mae_mean", ascending=False)
    tables["weak_segments"] = weak_df
    weak_df.to_csv(LOGS_DIR / f"weak_segments_{args.tag}.csv", index=False)
    print("\n  Weakest segment per dimension (highest MAE -> target these for EDA/features):")
    print(weak_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Performance drift — MAE per model over folds
    # ------------------------------------------------------------------
    fig = plot_metric_over_folds(
        mae_over_folds, fold_labels, metric_name="MAE",
        output_dir=str(OUTPUTS_PLOTS), filename=f"mae_over_folds_{args.tag}.png",
    )
    figures["mae_over_folds"] = fig

    # ------------------------------------------------------------------
    # Step 4 — TreeSHAP feature importance over time (champion)
    # ------------------------------------------------------------------
    print("\n=== Step 4: TreeSHAP predictive value over time ===")
    fold_importances = {}
    for fold_label in fold_labels:
        model = fold_models[fold_label][champion]
        Xs = fold_shap_X[fold_label]
        # Use the fold's own columns — one-hot zone names vary per fold.
        fold_importances[fold_label] = shap_importance(model, Xs, list(Xs.columns))

    shap_wide = shap_importance_over_folds(fold_importances)
    tables["shap_importance_over_time"] = shap_wide.reset_index()
    shap_wide.reset_index().to_csv(LOGS_DIR / f"shap_over_time_{args.tag}.csv", index=False)

    figures["shap_over_time"] = plot_shap_over_time(
        shap_wide, fold_labels, top_n=8, output_dir=str(OUTPUTS_PLOTS),
        filename=f"shap_over_time_{args.tag}.png",
    )
    # aggregate bar (mean across folds)
    shap_bar_df = (
        shap_wide.reset_index()[["feature", "mean_pct"]]
        .rename(columns={"mean_pct": "mean_abs_shap"})
    )
    figures["shap_bar"] = plot_shap_bar(
        shap_bar_df, top_n=20, output_dir=str(OUTPUTS_PLOTS),
        filename=f"shap_bar_{args.tag}.png",
        title=f"SHAP importance (mean over folds) — {champion}",
    )
    print(f"  Top predictive features (mean SHAP %): "
          f"{list(shap_wide.head(5).index)}")

    # ------------------------------------------------------------------
    # Feature drift over folds
    # ------------------------------------------------------------------
    drift_long = pd.DataFrame(drift_rows)
    tables["feature_drift_over_folds"] = drift_long
    drift_long.to_csv(LOGS_DIR / f"feature_drift_over_folds_{args.tag}.csv", index=False)
    figures["feature_drift_over_folds"] = plot_feature_drift_over_folds(
        drift_long, top_n=8, output_dir=str(OUTPUTS_PLOTS),
        filename=f"feature_drift_over_folds_{args.tag}.png",
    )
    if not drift_long.empty:
        peak = drift_long.groupby("feature")["psi"].max().sort_values(ascending=False)
        print(f"  Most-drifting features (peak PSI): {list(peak.head(5).index)}")

    # ------------------------------------------------------------------
    # W&B logging
    # ------------------------------------------------------------------
    print("\n=== Logging to W&B ===")
    tracker = ExperimentTracker(
        project=WANDB_PROJECT,
        run_name=f"validation-analysis-{args.tag}",
        tags=["validation-analysis", args.tag, f"champion-{champion}"],
        config={
            "sample_per_month": args.sample,
            "n_folds": len(fold_labels),
            "champion": champion,
            "models": list(_ALL_MODELS),
        },
        enabled=not args.no_wandb,
    )
    tracker.log_summary({
        "champion": champion,
        "champion_mean_rmse": mean_rmse[champion],
        "champion_mean_mae": float(np.mean(mae_over_folds[champion])),
        **{f"cv_rmse/{m}": mean_rmse[m] for m in _ALL_MODELS},
    })
    for name, fig in figures.items():
        if fig is not None:
            tracker.log_plot(fig, name)
    for name, df in tables.items():
        tracker.log_table(df, name)
    url = tracker.finish()
    if url:
        print(f"  W&B run -> {url}")

    print("\nValidation analysis complete.")
    print(f"  Plots  -> {OUTPUTS_PLOTS}")
    print(f"  Tables -> {LOGS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
