"""Drift detection and mitigation script.

Run:
    python mitigate.py --drift-parquet training_set/yellow_tripdata_2024-09.parquet
                       [--strategy auto|reweight_retrain|drop_features|recalibrate]
                       [--no-wandb]
                       [--tag latest]

Flow:
    1. Load drift-month parquet data
    2. Load saved artifacts (engineer, scaler, best model)
    3. Split drift month: days 1-21 = mitigation train, days 22+ = eval
    4. Load reference data (Nov-Dec 2025 val split) for comparison
    5. Run Evidently drift detection (dataset + label + concept drift)
    6. Apply selected mitigation strategy
    7. Evaluate before / after on held-out eval set
    8. Log everything to W&B
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PATHS, TARGET_COL, LOGS_DIR, WANDB_PROJECT, VAL_YEARS_MONTHS
from src.data.loader import load_parquet_files, load_taxi_zones
from src.data.cleaner import clean_training_data
from src.features.engineer import FeatureEngineer, get_raw_input_features
from src.models.registry import load_inference_artifacts
from src.models.trainer import compute_metrics
from src.drift.evidently_detector import (
    run_evidently_drift_report,
    parse_drift_results,
    run_evidently_concept_drift_report,
    parse_concept_drift_results,
    select_mitigation_strategy,
)
from src.drift.mitigation import mitigate, plot_mitigation_comparison
from src.tracking.wandb_tracker import (
    ExperimentTracker,
    log_data_artifact,
    log_model_artifact,
    log_feature_artifact,
)


DRIFT_TRAIN_DAY_CUTOFF = 21
DRIFT_SEED = 42
DRIFT_TRAIN_SAMPLE = 20_000
DRIFT_EVAL_SAMPLE = 5_000
MITIGATED_MODEL_DIR = "models/mitigated"
OUTPUTS_DIR = Path("outputs")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--drift-parquet", required=True,
        help="Path to a monthly parquet file to analyse for drift",
    )
    p.add_argument(
        "--strategy", default="auto",
        choices=["auto", "reweight_retrain", "drop_features", "recalibrate", "none"],
    )
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--tag", type=str, default="latest")
    return p.parse_args()


def _load_reference(engineer, zones_df):
    """Load and engineer the validation split (Nov-Dec 2025) as the drift reference."""
    print("  Loading reference data (Nov-Dec 2025) ...")
    raw = load_parquet_files(DATA_PATHS["training"], n_per_file=30_000)
    zones_df_loaded = zones_df
    cleaned = clean_training_data(raw)

    is_val = pd.Series(False, index=cleaned.index)
    for yr, mo in VAL_YEARS_MONTHS:
        is_val |= (cleaned["pickup_year"] == yr) & (cleaned["pickup_month"] == mo)

    ref_df = cleaned[is_val].copy().reset_index(drop=True)
    print(f"  Reference rows: {len(ref_df):,}")
    return ref_df


def main():
    args = parse_args()
    enabled_wandb = not args.no_wandb

    OUTPUTS_DIR.mkdir(exist_ok=True)
    (OUTPUTS_DIR / "plots").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load drift month
    # ------------------------------------------------------------------
    drift_path = Path(args.drift_parquet)
    month_label = drift_path.stem.replace("yellow_tripdata_", "").replace("-", " ")
    print(f"\n=== Loading drift data: {drift_path.name} ===")
    drift_raw = pd.read_parquet(drift_path)
    drift_raw["tpep_pickup_datetime"] = pd.to_datetime(drift_raw["tpep_pickup_datetime"])
    drift_df = clean_training_data(drift_raw)
    print(f"  Rows after cleaning: {len(drift_df):,}")

    # Split by calendar day
    day = drift_df["tpep_pickup_datetime"].dt.day if "tpep_pickup_datetime" in drift_df.columns else \
          pd.to_datetime(drift_df.get("pickup_datetime", pd.Series())).dt.day

    drift_train_df = (
        drift_df[drift_df["tpep_pickup_datetime"].dt.day <= DRIFT_TRAIN_DAY_CUTOFF]
        .sample(min(DRIFT_TRAIN_SAMPLE, len(drift_df)), random_state=DRIFT_SEED)
        .reset_index(drop=True)
    )
    drift_eval_df = (
        drift_df[drift_df["tpep_pickup_datetime"].dt.day > DRIFT_TRAIN_DAY_CUTOFF]
        .sample(min(DRIFT_EVAL_SAMPLE, len(drift_df)), random_state=DRIFT_SEED)
        .reset_index(drop=True)
    )
    print(f"  Drift train: {len(drift_train_df):,}  Drift eval: {len(drift_eval_df):,}")

    # ------------------------------------------------------------------
    # 2. Load saved artifacts
    # ------------------------------------------------------------------
    print(f"\n=== Loading artifacts (tag='{args.tag}') ===")
    zones_df = load_taxi_zones(DATA_PATHS["taxi_zones"])
    engineer, scaler, best_model = load_inference_artifacts(run_tag=args.tag)
    best_model_name = type(best_model).__name__.lower().replace("regressor", "").replace("classifier", "")
    # map sklearn class names back to short keys
    _name_map = {"lgbm": "lgbm", "xgb": "xgb", "randomforest": "rf", "ridge": "ridge"}
    model_key = next((v for k, v in _name_map.items() if k in best_model_name), "rf")

    # ------------------------------------------------------------------
    # 3. Load reference data
    # ------------------------------------------------------------------
    print("\n=== Loading reference (val split) ===")
    ref_df = _load_reference(engineer, zones_df)

    # ------------------------------------------------------------------
    # 4. Feature engineering
    # ------------------------------------------------------------------
    print("\n=== Feature engineering ===")

    def _engineer(df):
        X_raw = get_raw_input_features(df)
        X_eng = engineer.transform(X_raw)
        return engineer.get_tree_features(X_eng), df[TARGET_COL].reset_index(drop=True)

    X_ref_feat, y_ref = _engineer(ref_df)
    X_drift_train_feat, y_drift_train = _engineer(drift_train_df)
    X_drift_eval_feat, y_drift_eval = _engineer(drift_eval_df)

    # ------------------------------------------------------------------
    # 5. Pre-mitigation evaluation
    # ------------------------------------------------------------------
    y_pred_ref = best_model.predict(X_ref_feat)
    ref_mae = float(np.mean(np.abs(y_ref.values - y_pred_ref)))

    y_pred_eval_base = best_model.predict(X_drift_eval_feat)
    base_eval_mae = float(np.mean(np.abs(y_drift_eval.values - y_pred_eval_base)))
    print(f"\n  Reference MAE       : {ref_mae:.4f}")
    print(f"  Drift eval MAE (pre): {base_eval_mae:.4f}")

    # ------------------------------------------------------------------
    # 6. Evidently drift detection
    # ------------------------------------------------------------------
    print("\n=== Evidently drift detection ===")

    # Build engineered DataFrames with target column for Evidently
    def _eng_with_target(df):
        X_raw = get_raw_input_features(df)
        X_eng = engineer.transform(X_raw)
        X_feat = engineer.get_tree_features(X_eng)
        X_feat = X_feat.copy()
        X_feat[TARGET_COL] = df[TARGET_COL].reset_index(drop=True).values
        return X_feat

    ref_eng_df = _eng_with_target(ref_df)
    drift_train_eng_df = _eng_with_target(drift_train_df)

    evidently_html = OUTPUTS_DIR / "evidently_drift_report.html"
    evidently_report = run_evidently_drift_report(ref_eng_df, drift_train_eng_df)
    drift_results = parse_drift_results(evidently_report)
    evidently_report.save_html(str(evidently_html))

    print(f"  Overall drift       : {drift_results['overall_drift']}")
    print(f"  Features drifted    : {drift_results['n_drifted']} ({drift_results['share_drifted']:.1%})")
    if drift_results["drifted_features"]:
        print(f"  Drifted features    : {drift_results['drifted_features']}")
    print(f"  Target drift        : {drift_results['target_drift']} (score={drift_results['target_drift_score']:.4f})")
    print(f"  Evidently HTML      : {evidently_html}")

    # Concept drift
    ref_perf = ref_eng_df.copy()
    ref_perf["prediction"] = best_model.predict(X_ref_feat)

    drift_eval_eng_df = _eng_with_target(drift_eval_df)
    cur_perf = drift_eval_eng_df.copy()
    cur_perf["prediction"] = best_model.predict(X_drift_eval_feat)

    concept_report = run_evidently_concept_drift_report(ref_perf, cur_perf)
    concept_results = parse_concept_drift_results(concept_report)
    concept_html = OUTPUTS_DIR / "evidently_concept_drift_report.html"
    concept_report.save_html(str(concept_html))

    print(f"\n  Concept drift       : {concept_results['concept_drift_detected']}")
    print(f"  Ref MAE             : {concept_results['ref_mae']:.4f}")
    print(f"  Current MAE         : {concept_results['cur_mae']:.4f}")
    print(f"  MAE increase        : {concept_results['mae_pct_increase']:.1%}")

    # ------------------------------------------------------------------
    # 7. Strategy selection + mitigation
    # ------------------------------------------------------------------
    strategy = args.strategy if args.strategy != "auto" else \
        select_mitigation_strategy(drift_results, concept_results)

    print(f"\n=== Applying mitigation: {strategy} ===")
    mitigated_model, dropped_features = mitigate(
        strategy=strategy,
        X_train=X_ref_feat,
        y_train=y_ref,
        X_recent=X_drift_train_feat,
        y_recent=y_drift_train,
        model_name=model_key,
        model_dir=MITIGATED_MODEL_DIR,
        base_model=best_model,
        drifted_features=drift_results.get("drifted_features"),
    )

    if mitigated_model is None:
        mitigated_model = best_model

    # ------------------------------------------------------------------
    # 8. Post-mitigation evaluation
    # ------------------------------------------------------------------
    if dropped_features:
        X_eval_mit = X_drift_eval_feat.drop(columns=dropped_features, errors="ignore")
    else:
        X_eval_mit = X_drift_eval_feat

    y_pred_eval_mit = mitigated_model.predict(X_eval_mit)
    mit_eval_mae = float(np.mean(np.abs(y_drift_eval.values - y_pred_eval_mit)))
    improvement = (base_eval_mae - mit_eval_mae) / base_eval_mae * 100

    print(f"\n  Drift eval MAE (post): {mit_eval_mae:.4f}")
    print(f"  Improvement          : {improvement:+.1f}%")

    # comparison plot
    comparison_fig = plot_mitigation_comparison(
        {
            "Champion -- ref (in-dist)": np.abs(y_ref.values - y_pred_ref),
            f"Champion -- {month_label} (drifted)": np.abs(y_drift_eval.values - y_pred_eval_base),
            f"Mitigated ({strategy}) -- {month_label}": np.abs(y_drift_eval.values - y_pred_eval_mit),
        },
        output_dir=str(OUTPUTS_DIR / "plots"),
    )

    # ------------------------------------------------------------------
    # 9. W&B logging
    # ------------------------------------------------------------------
    print("\n=== Logging to W&B ===")
    tracker = ExperimentTracker(
        project=WANDB_PROJECT,
        run_name=f"mitigation-{strategy}-{month_label.replace(' ', '-')}",
        tags=["drift-mitigation", month_label, strategy],
        config={
            "strategy": strategy,
            "month": month_label,
            "drifted_features": drift_results.get("drifted_features", []),
            "n_drift_train": len(drift_train_df),
            "n_drift_eval": len(drift_eval_df),
        },
        enabled=enabled_wandb,
    )
    tracker.log_summary({
        "ref_mae": ref_mae,
        "base_drift_mae": base_eval_mae,
        "mitigated_drift_mae": mit_eval_mae,
        "mae_improvement_pct": improvement,
    })
    tracker.log_plot(comparison_fig, "mitigation_comparison")

    drift_eval_parquet = OUTPUTS_DIR / "drift_eval.parquet"
    drift_eval_df.to_parquet(drift_eval_parquet, index=False)
    log_data_artifact(
        tracker, drift_eval_parquet, f"{month_label.replace(' ', '-')}-eval-set",
        metadata={"month": month_label, "n_rows": len(drift_eval_df)},
    )

    mitigated_pkl = next(
        Path(MITIGATED_MODEL_DIR).glob(f"{model_key}_*.pkl"), None
    )
    if mitigated_pkl and mitigated_pkl.exists():
        log_model_artifact(
            tracker, mitigated_pkl, "mitigated-model",
            metadata={"strategy": strategy, "mae": mit_eval_mae},
        )

    tracker.log_artifact(evidently_html, "evidently-drift-report", "report")
    url = tracker.finish()
    if url:
        print(f"  W&B run -> {url}")

    print("\nMitigation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
