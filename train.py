"""Main training script for the NYC TLC fare prediction pipeline.

Run:
    python train.py [--sample N] [--no-wandb] [--tag RUN_TAG]

Arguments:
    --sample N      rows sampled per monthly file (default: 150,000)
    --no-wandb      disable Weights & Biases logging
    --tag RUN_TAG   sub-folder under models/ to save artifacts (default: latest)

Flow:
    1. Load 2024 + 2025 training parquet files (sampled)
    2. Clean data (filters, outlier clipping)
    3. Feature engineering (domain rules + target encoding)
    4. Temporal train/val split: last 2 months (Nov-Dec 2025) held out
    5. Train LightGBM, XGBoost, Random Forest, Ridge
    6. Log all runs to W&B; compare on validation RMSE
    7. Save best model + engineer + scaler
    8. Error analysis on validation set
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_PATHS, SAMPLE_CONFIG, TARGET_COL, VAL_YEARS_MONTHS,
    WANDB_PROJECT, LOGS_DIR,
)
from src.data.loader import load_parquet_files, load_taxi_zones
from src.data.cleaner import clean_training_data
from src.features.engineer import FeatureEngineer, get_raw_input_features
from src.models.trainer import train_all_models, select_best_model
from src.models.evaluator import error_analysis, get_feature_importance, residual_summary
from src.models.registry import save_run_artifacts, get_artifact_paths
from src.tracking.wandb_tracker import WandbTracker


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=SAMPLE_CONFIG["n_per_month_train"])
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--tag", type=str, default="latest")
    return p.parse_args()


def temporal_split(df: pd.DataFrame):
    """Hold out VAL_YEARS_MONTHS as validation; rest is training."""
    is_val = pd.Series(False, index=df.index)
    for yr, mo in VAL_YEARS_MONTHS:
        is_val |= (df["pickup_year"] == yr) & (df["pickup_month"] == mo)
    return df[~is_val].copy(), df[is_val].copy()


def main():
    args = parse_args()
    tracker = WandbTracker(enabled=not args.no_wandb)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n=== Loading training data ===")
    raw_df = load_parquet_files(
        DATA_PATHS["training"],
        n_per_file=args.sample,
        random_state=SAMPLE_CONFIG["random_state"],
    )

    print("\n=== Loading taxi zones ===")
    zones_df = load_taxi_zones(DATA_PATHS["taxi_zones"])

    # ------------------------------------------------------------------
    # 2. Clean
    # ------------------------------------------------------------------
    print("\n=== Cleaning data ===")
    clean_df = clean_training_data(raw_df)
    print(f"  Clean shape: {clean_df.shape}")

    # ------------------------------------------------------------------
    # 3. Temporal split
    # ------------------------------------------------------------------
    print("\n=== Splitting train / validation ===")
    train_df, val_df = temporal_split(clean_df)
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}")

    # ------------------------------------------------------------------
    # 4. Feature engineering
    # ------------------------------------------------------------------
    print("\n=== Feature engineering ===")
    X_train_raw = get_raw_input_features(train_df)
    X_val_raw = get_raw_input_features(val_df)
    y_train = train_df[TARGET_COL].reset_index(drop=True)
    y_val = val_df[TARGET_COL].reset_index(drop=True)

    engineer = FeatureEngineer(zones_df)
    engineer.fit(X_train_raw, y_train)

    X_train_eng = engineer.transform(X_train_raw)
    X_val_eng = engineer.transform(X_val_raw)

    X_train_feat = engineer.get_tree_features(X_train_eng)
    X_val_feat = engineer.get_tree_features(X_val_eng)
    print(f"  Feature matrix: {X_train_feat.shape[1]} features, {len(X_train_feat):,} rows")

    # ------------------------------------------------------------------
    # 5. Train all models
    # ------------------------------------------------------------------
    print("\n=== Training all models ===")
    results, scaler = train_all_models(X_train_feat, y_train, X_val_feat, y_val)

    # ------------------------------------------------------------------
    # 6. Log to W&B
    # ------------------------------------------------------------------
    print("\n=== Logging to W&B ===")
    all_metrics = {name: m for name, (_, m) in results.items()}

    with tracker.init_run(
        name=f"full-run-{args.tag}",
        config={
            "sample_per_month": args.sample,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_features": X_train_feat.shape[1],
            "val_split": str(VAL_YEARS_MONTHS),
        },
        tags=["training", args.tag],
    ):
        for name, metrics in all_metrics.items():
            tracker.log({f"{name}/{k}": v for k, v in metrics.items()})

        artifact_paths = get_artifact_paths(args.tag)
        tracker.log_all_models(artifact_paths, all_metrics)

    # ------------------------------------------------------------------
    # 7. Select and save best model
    # ------------------------------------------------------------------
    best_name = select_best_model(results)
    best_model = results[best_name][0]
    print(f"\n=== Best model: {best_name.upper()} ===")
    print(f"  Val RMSE: {all_metrics[best_name]['rmse']:.4f}")

    save_run_artifacts(
        feature_engineer=engineer,
        scaler=scaler,
        models={name: m for name, (m, _) in results.items()},
        best_model_name=best_name,
        run_tag=args.tag,
    )

    # ------------------------------------------------------------------
    # 8. Error analysis on best model
    # ------------------------------------------------------------------
    print("\n=== Error analysis ===")
    val_df_aligned = val_df.copy()
    # add engineered features for segmentation
    for col in ["pu_borough", "is_airport_route", "trip_distance", "pickup_hour", "pickup_dayofweek"]:
        if col in X_val_eng.columns:
            val_df_aligned[col] = X_val_eng[col].values

    use_scaler = (best_name == "ridge")
    if use_scaler:
        y_pred_val = best_model.predict(scaler.transform(X_val_feat.values))
    else:
        y_pred_val = best_model.predict(X_val_feat)

    analyses = error_analysis(y_val, y_pred_val, val_df_aligned)
    residuals = residual_summary(y_val, y_pred_val)

    print("\nResidual summary (y_pred - y_true):")
    for k, v in residuals.items():
        print(f"  {k}: {v:.3f}")

    # Save error analysis to logs
    for name, df_seg in analyses.items():
        out_path = LOGS_DIR / f"error_by_{name}_{args.tag}.csv"
        df_seg.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    # Feature importance for best tree model
    if best_name != "ridge":
        fi_df = get_feature_importance(best_model, np.array(engineer.get_feature_names()))
        fi_path = LOGS_DIR / f"feature_importance_{args.tag}.csv"
        fi_df.to_csv(fi_path, index=False)
        print(f"  Feature importance saved: {fi_path}")
        print(fi_df.head(15).to_string(index=False))

    # ------------------------------------------------------------------
    # 9. Summary table
    # ------------------------------------------------------------------
    print("\n=== Model comparison ===")
    summary_rows = [{"model": k, **v} for k, v in all_metrics.items()]
    summary_df = pd.DataFrame(summary_rows).sort_values("rmse")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(LOGS_DIR / f"model_comparison_{args.tag}.csv", index=False)

    print("\nTraining complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
