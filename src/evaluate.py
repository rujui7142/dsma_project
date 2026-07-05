"""Evaluate the trained model on the 2026 test set.

Run:
    python evaluate.py [--no-wandb] [--tag RUN_TAG]

Flow:
    1. Load Jan + Feb 2026 test data (full, no sampling)
    2. Clean (same pipeline as training)
    3. Apply feature engineering from saved artifact
    4. Load best saved model
    5. Predict & compute metrics
    6. Drift detection: compare 2025 reference vs 2026 test feature distributions
    7. Segmented error analysis (borough, hour, distance bin, airport flag)
    8. Log everything to W&B
"""

import argparse
import sys

import numpy as np
import pandas as pd

from src.config import DATA_PATHS, TARGET_COL, LOGS_DIR, WANDB_PROJECT, SAMPLE_CONFIG
from src.data.loader import load_parquet_files, load_taxi_zones
from src.data.cleaner import clean_test_data, clean_training_data
from src.features.engineer import get_raw_input_features
from src.models.evaluator import error_analysis, get_feature_importance, residual_summary
from src.models.registry import load_inference_artifacts, save_run_artifacts
from src.models.trainer import compute_metrics
from src.drift.detector import generate_drift_report, detect_performance_drift
from src.drift.evidently_detector import (
    run_evidently_drift_report,
    parse_drift_results,
    run_evidently_concept_drift_report,
    parse_concept_drift_results,
)
from src.tracking.wandb_tracker import WandbTracker, ExperimentTracker


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--tag", type=str, default="latest")
    p.add_argument("--sample", type=int, default=SAMPLE_CONFIG["n_per_month_test"],
                   help="rows sampled per test month (None loads the full month "
                        "-- risks OOM on memory-constrained runners)")
    return p.parse_args()


def main():
    args = parse_args()
    tracker = WandbTracker(enabled=not args.no_wandb)

    # ------------------------------------------------------------------
    # 1. Load test data (2026)
    # ------------------------------------------------------------------
    print("\n=== Loading 2026 test data ===")
    test_raw = load_parquet_files(DATA_PATHS["test"], n_per_file=args.sample)
    zones_df = load_taxi_zones(DATA_PATHS["taxi_zones"])

    # ------------------------------------------------------------------
    # 2. Clean
    # ------------------------------------------------------------------
    print("\n=== Cleaning test data ===")
    test_df = clean_test_data(test_raw)
    print(f"  Test shape: {test_df.shape}")

    # ------------------------------------------------------------------
    # 3. Load saved artifacts
    # ------------------------------------------------------------------
    print(f"\n=== Loading artifacts (tag='{args.tag}') ===")
    engineer, scaler, best_model = load_inference_artifacts(run_tag=args.tag)

    # ------------------------------------------------------------------
    # 4. Feature engineering
    # ------------------------------------------------------------------
    print("\n=== Feature engineering ===")
    X_test_raw = get_raw_input_features(test_df)
    X_test_eng = engineer.transform(X_test_raw)
    X_test_feat = engineer.get_tree_features(X_test_eng)
    y_test = test_df[TARGET_COL].reset_index(drop=True)
    print(f"  Test features: {X_test_feat.shape}")

    # ------------------------------------------------------------------
    # 5. Predict
    # ------------------------------------------------------------------
    print("\n=== Predicting ===")
    model_name = type(best_model).__name__
    is_linear = "Ridge" in model_name
    if is_linear:
        y_pred = best_model.predict(scaler.transform(X_test_feat.values))
    else:
        y_pred = best_model.predict(X_test_feat)

    metrics = compute_metrics(y_test.values, y_pred)
    print(f"  Test RMSE: {metrics['rmse']:.4f}")
    print(f"  Test MAE:  {metrics['mae']:.4f}")
    print(f"  Test R2:   {metrics['r2']:.4f}")
    print(f"  Test MAPE: {metrics['mape']:.2f}%")

    # ------------------------------------------------------------------
    # 6. Error analysis
    # ------------------------------------------------------------------
    print("\n=== Error analysis ===")
    test_meta = test_df.copy().reset_index(drop=True)
    for col in ["pu_borough", "is_airport_route", "trip_distance", "pickup_hour", "pickup_dayofweek"]:
        if col in X_test_eng.columns:
            test_meta[col] = X_test_eng[col].values

    analyses = error_analysis(y_test, y_pred, test_meta)
    residuals = residual_summary(y_test, y_pred)

    print("\nResidual summary (y_pred - y_true):")
    for k, v in residuals.items():
        print(f"  {k}: {v:.3f}")

    for seg_name, seg_df in analyses.items():
        out_path = LOGS_DIR / f"test_error_by_{seg_name}_{args.tag}.csv"
        seg_df.to_csv(out_path, index=False)
        print(f"\nError by {seg_name}:")
        print(seg_df.head(8).to_string(index=False))

    # Performance drift over months
    if "pickup_month" in test_meta.columns:
        perf_drift = detect_performance_drift(y_test, y_pred, test_meta, time_col="pickup_month")
        print("\nPerformance by month (test set):")
        print(perf_drift.to_string(index=False))
        perf_drift.to_csv(LOGS_DIR / f"perf_drift_{args.tag}.csv", index=False)

    # ------------------------------------------------------------------
    # 7. Drift detection: 2025 reference vs 2026 test
    # ------------------------------------------------------------------
    print("\n=== Drift detection (2025 ref -> 2026 test) ===")
    print("  Loading 2025 reference data for drift comparison ...")
    ref_files = sorted(DATA_PATHS["training"].glob("*2025-1*.parquet"))  # Nov + Dec 2025
    if ref_files:
        ref_raw = pd.concat([pd.read_parquet(f).sample(min(30_000, len(pd.read_parquet(f))), random_state=42)
                              for f in ref_files], ignore_index=True)
        ref_df = clean_training_data(ref_raw)
        X_ref_raw = get_raw_input_features(ref_df)
        X_ref_eng = engineer.transform(X_ref_raw)

        drift_features = [
            "trip_distance", "pickup_hour", "pickup_dayofweek",
            "is_airport_route", "estimated_surcharges",
        ]
        drift_report = generate_drift_report(
            df_ref=X_ref_eng,
            df_curr=X_test_eng,
            feature_cols=drift_features,
        )
        print(f"\nDrift summary: {drift_report['summary']}")
        print("\nFeature drift:")
        print(drift_report["feature_drift"].to_string(index=False))
        drift_report["feature_drift"].to_csv(LOGS_DIR / f"feature_drift_{args.tag}.csv", index=False)
    else:
        drift_report = {}
        print("  No 2025 reference files found - skipping drift analysis.")

    # ------------------------------------------------------------------
    # 7b. Evidently concept drift (reference vs 2026 test)
    # ------------------------------------------------------------------
    evidently_results = {}
    if ref_files:
        print("\n=== Evidently concept drift (2025 ref -> 2026 test) ===")
        try:
            from pathlib import Path as _Path
            _Path("outputs").mkdir(exist_ok=True)

            # Dataset + label drift: engineered features + target column
            ref_ev = X_ref_eng.copy()
            ref_ev[TARGET_COL] = ref_df[TARGET_COL].values
            cur_ev = X_test_eng.copy()
            cur_ev[TARGET_COL] = y_test.values

            ev_report = run_evidently_drift_report(ref_ev, cur_ev)
            evidently_results = parse_drift_results(ev_report)
            ev_report.save_html("outputs/evidently_drift_report.html")

            # Concept drift: model predictions on both reference and test
            X_ref_feat = engineer.get_tree_features(X_ref_eng)
            ref_perf = X_ref_feat.copy()
            ref_perf[TARGET_COL] = ref_df[TARGET_COL].values
            if is_linear:
                ref_perf["prediction"] = best_model.predict(scaler.transform(X_ref_feat.values))
            else:
                ref_perf["prediction"] = best_model.predict(X_ref_feat)

            cur_perf = X_test_feat.copy()
            cur_perf[TARGET_COL] = y_test.values
            cur_perf["prediction"] = y_pred

            concept_report = run_evidently_concept_drift_report(ref_perf, cur_perf)
            concept_results = parse_concept_drift_results(concept_report)
            concept_report.save_html("outputs/evidently_concept_drift_report.html")

            print(f"  Overall drift    : {evidently_results['overall_drift']}")
            print(f"  Features drifted : {evidently_results['n_drifted']} ({evidently_results['share_drifted']:.1%})")
            print(f"  Concept drift    : {concept_results['concept_drift_detected']}")
            print(f"  MAE increase     : {concept_results['mae_pct_increase']:.1%}")
            print(f"  Evidently HTML   : outputs/evidently_drift_report.html")
        except Exception as exc:
            print(f"  Evidently analysis skipped: {exc}")

    # ------------------------------------------------------------------
    # 8. Log to W&B
    # ------------------------------------------------------------------
    print("\n=== Logging to W&B ===")
    with tracker.init_run(
        name=f"evaluate-{args.tag}",
        config={"run_tag": args.tag, "n_test": len(test_df), "model": model_name},
        tags=["evaluation", "2026-test", args.tag],
    ):
        tracker.log({f"test/{k}": v for k, v in metrics.items()})
        tracker.log(residuals)
        for seg_name, seg_df in analyses.items():
            tracker.log_dataframe(seg_df, f"test_error_{seg_name}")
        if drift_report:
            tracker.log_drift_report(drift_report)

    print("\nEvaluation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
