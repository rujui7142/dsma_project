"""Validation & drift analysis pipeline — train/validation data only.

This script NEVER touches the test set (2026). It runs the forward-chaining CV
and, across all four models and every fold, produces the analysis the lecturer
asked for:

  Step 2  Segmented error analysis across all models  -> weak-segment tables + plots
          (drives "which features need more EDA / engineering")
  Step 4  TreeSHAP feature importance per fold         -> predictive value over time
  Step 5  Month-by-month temporal trend (raw fare + champion MAE, 2024 onward)
          -> explains whether a coarse fold's high error reflects a gradual
             trend or a sharp regime change (e.g. the Jan-2025 CBD fee)
  Drift   Per-fold feature drift (PSI, fixed reference) + per-model MAE over folds

All artifacts are saved to outputs/plots + logs/ using FIXED filenames (no run
tag suffix) — every run overwrites the previous one, so the files on disk
always reflect the latest results. --tag only labels the W&B run for history.

Run:
    python -m src.analyze [--sample N] [--no-wandb] [--tag analysis] [--shap-sample N]
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
    plot_metric_over_folds, plot_monthly_temporal_trend,
)
from src.models.shap_analysis import (
    shap_importance, shap_importance_over_folds, plot_shap_bar, plot_shap_over_time,
)
from src.drift.detector import (
    detect_feature_drift, plot_feature_drift_over_folds, plot_psi_heatmap,
)
from src.tracking.wandb_tracker import ExperimentTracker
from src.train import forward_chain_splits


_ALL_MODELS = ("lgbm", "xgb", "rf", "ridge")

# Segment dimensions to break error down by (must exist on the engineered df).
SEGMENT_DIMS = [
    "pu_borough", "distance_bin", "pickup_hour",
    "is_airport_route", "is_hotspot_route", "crosses_cbd",
]

# First month affected by the CBD congestion fee — marked on the temporal plot.
CBD_FEE_START_MONTH = "2025-01"

# Months reserved as the initial training window for the monthly walk-forward
# (Step 5) — kept small so most of 2024 is visible as validated months.
MONTHLY_INIT_MONTHS = 6

OUTPUTS_PLOTS = Path("outputs/plots")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=SAMPLE_CONFIG["n_per_month_train"])
    p.add_argument("--shap-sample", type=int, default=3000,
                   help="rows sampled per fold for TreeSHAP (keeps it fast)")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--tag", type=str, default="analysis",
                   help="W&B run label only — local plot/log filenames are fixed")
    return p.parse_args()


def _monthly_temporal_analysis(
    clean_df: pd.DataFrame,
    zones_df: pd.DataFrame,
    champion: str,
    init_months: int = MONTHLY_INIT_MONTHS,
) -> pd.DataFrame:
    """Expanding walk-forward at single-month granularity for the champion model.

    Returns a DataFrame [month_label, actual_mean_fare, champion_mae] covering
    every calendar month present in clean_df (actual_mean_fare always filled;
    champion_mae is NaN for the initial training-only months).
    """
    months = sorted(clean_df.groupby(["pickup_year", "pickup_month"]).groups.keys())
    month_key = clean_df["pickup_year"] * 100 + clean_df["pickup_month"]

    # Raw actual fare trend — no model, every month.
    rows = {
        f"{yr}-{mo:02d}": {
            "month_label": f"{yr}-{mo:02d}",
            "actual_mean_fare": float(clean_df.loc[month_key == yr * 100 + mo, TARGET_COL].mean()),
            "champion_mae": np.nan,
        }
        for yr, mo in months
    }

    print(f"\n  Monthly walk-forward ({champion}), {init_months}-month initial window:")
    for i in range(init_months, len(months)):
        yr, mo = months[i]
        month_label = f"{yr}-{mo:02d}"
        train_keys = {y * 100 + m for y, m in months[:i]}
        val_keys = {yr * 100 + mo}

        tr_df = clean_df[month_key.isin(train_keys)]
        vl_df = clean_df[month_key.isin(val_keys)]
        if len(vl_df) < 20:
            continue

        X_tr_raw, X_vl_raw = get_raw_input_features(tr_df), get_raw_input_features(vl_df)
        y_tr = tr_df[TARGET_COL].reset_index(drop=True)
        y_vl = vl_df[TARGET_COL].reset_index(drop=True)

        eng = FeatureEngineer(zones_df)
        eng.fit(X_tr_raw, y_tr)
        X_tr_feat = eng.get_tree_features(eng.transform(X_tr_raw))
        X_vl_feat = eng.get_tree_features(eng.transform(X_vl_raw))
        scaler = build_ridge_scaler(X_tr_feat) if champion == "ridge" else None

        _, metrics = train_model(champion, X_tr_feat, y_tr, X_vl_feat, y_vl, scaler=scaler)
        rows[month_label]["champion_mae"] = metrics["mae"]
        print(f"    {month_label}: MAE={metrics['mae']:.3f}  (train={len(tr_df):,} val={len(vl_df):,})")

    return pd.DataFrame(list(rows.values())).sort_values("month_label").reset_index(drop=True)


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
    mae_over_folds = {m: [] for m in _ALL_MODELS}
    rmse_over_folds = {m: [] for m in _ALL_MODELS}
    segment_errors = {m: {d: [] for d in SEGMENT_DIMS} for m in _ALL_MODELS}
    fold_models = {}
    fold_shap_X = {}
    feature_names = None

    ref_engineer = None
    ref_feat = None
    drift_rows = []

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
        agg.to_csv(LOGS_DIR / f"cv_error_by_{dim}.csv", index=False)
        fig = plot_segment_error(
            agg, dim, output_dir=str(OUTPUTS_PLOTS),
            filename=f"cv_error_by_{dim}.png",
            title=f"CV error by {dim} — {champion}",
        )
        if fig is not None:
            figures[f"cv_error_by_{dim}"] = fig
        worst = agg.iloc[0]
        weak_segments.append({
            "dimension": dim, "worst_segment": worst[dim],
            "mae_mean": worst["mae_mean"], "avg_count": worst["avg_count"],
        })

    weak_df = pd.DataFrame(weak_segments).sort_values("mae_mean", ascending=False)
    tables["weak_segments"] = weak_df
    weak_df.to_csv(LOGS_DIR / "weak_segments.csv", index=False)
    print("\n  Weakest segment per dimension (highest MAE -> target these for EDA/features):")
    print(weak_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Performance drift — MAE per model over folds
    # ------------------------------------------------------------------
    fig = plot_metric_over_folds(
        mae_over_folds, fold_labels, metric_name="MAE",
        output_dir=str(OUTPUTS_PLOTS), filename="mae_over_folds.png",
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
        fold_importances[fold_label] = shap_importance(model, Xs, list(Xs.columns))

    shap_wide = shap_importance_over_folds(fold_importances)
    tables["shap_importance_over_time"] = shap_wide.reset_index()
    shap_wide.reset_index().to_csv(LOGS_DIR / "shap_over_time.csv", index=False)

    figures["shap_over_time"] = plot_shap_over_time(
        shap_wide, fold_labels, top_n=8, output_dir=str(OUTPUTS_PLOTS),
        filename="shap_over_time.png",
    )
    shap_bar_df = (
        shap_wide.reset_index()[["feature", "mean_pct"]]
        .rename(columns={"mean_pct": "mean_abs_shap"})
    )
    figures["shap_bar"] = plot_shap_bar(
        shap_bar_df, top_n=20, output_dir=str(OUTPUTS_PLOTS),
        filename="shap_bar.png",
        title=f"SHAP importance (mean over folds) — {champion}",
    )
    print(f"  Top predictive features (mean SHAP %): "
          f"{list(shap_wide.head(5).index)}")

    # ------------------------------------------------------------------
    # Feature drift over folds
    # ------------------------------------------------------------------
    drift_long = pd.DataFrame(drift_rows)
    tables["feature_drift_over_folds"] = drift_long
    drift_long.to_csv(LOGS_DIR / "feature_drift_over_folds.csv", index=False)
    figures["feature_drift_over_folds"] = plot_feature_drift_over_folds(
        drift_long, top_n=8, output_dir=str(OUTPUTS_PLOTS),
        filename="feature_drift_over_folds.png",
    )
    # Full feature x fold PSI heatmap (every candidate feature, not just top-8)
    # for manual drift investigation — see plot_psi_heatmap docstring for why
    # PSI can exceed 1 for step-function features (not a bug).
    figures["psi_heatmap"] = plot_psi_heatmap(
        drift_long, output_dir=str(OUTPUTS_PLOTS), filename="psi_heatmap.png",
    )
    if not drift_long.empty:
        peak = drift_long.groupby("feature")["psi"].max().sort_values(ascending=False)
        print(f"  Most-drifting features (peak PSI): {list(peak.head(5).index)}")

    # ------------------------------------------------------------------
    # Step 5 — Month-by-month temporal trend (2024 onward)
    # ------------------------------------------------------------------
    print("\n=== Step 5: Monthly temporal trend (2024 onward) ===")
    monthly_df = _monthly_temporal_analysis(clean_df, zones_df, champion)
    tables["monthly_temporal_trend"] = monthly_df
    monthly_df.to_csv(LOGS_DIR / "monthly_temporal_trend.csv", index=False)
    figures["monthly_temporal_trend"] = plot_monthly_temporal_trend(
        monthly_df, output_dir=str(OUTPUTS_PLOTS), filename="monthly_temporal_trend.png",
        shift_month_label=CBD_FEE_START_MONTH,
    )

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
