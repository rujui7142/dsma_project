"""Walk-forward drift monitoring + automatic mitigation (train/val only).

Simulates production over time using the chronological train/validation months
(NEVER the 2026 test set). Starting from an initial window it walks forward one
month at a time and, using the course's drift-detection + mitigation functions,
automatically updates the model whenever a *major* drift is detected.

Two models are tracked side by side:
  - frozen   : trained once on the initial window, never updated
  - adaptive : auto-retrained (mitigated) whenever drift crosses the threshold

The output MAE-over-time curve shows the adaptive model staying accurate while
the frozen one degrades — the core "drift mitigation pays off" story.

Reused course functions:
  - src.drift.detector.detect_feature_drift           (which features drifted)
  - src.drift.evidently_detector.select_mitigation_strategy   (choose response)
  - src.drift.mitigation.mitigate                      (apply response)

Run:
    python mitigate_walkforward.py [--sample N] [--model lgbm] [--no-wandb]
                                   [--init-frac 0.6] [--mae-threshold 0.10]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PATHS, SAMPLE_CONFIG, TARGET_COL, LOGS_DIR, WANDB_PROJECT
from src.data.loader import load_parquet_files, load_taxi_zones
from src.data.cleaner import clean_training_data
from src.features.engineer import FeatureEngineer, get_raw_input_features
from src.models.trainer import train_model, compute_metrics
from src.drift.detector import detect_feature_drift
from src.drift.evidently_detector import select_mitigation_strategy
from src.drift.mitigation import mitigate
from src.tracking.wandb_tracker import ExperimentTracker

OUTPUTS_PLOTS = Path("outputs/plots")
MITIGATED_DIR = "models/walkforward"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=SAMPLE_CONFIG["n_per_month_train"])
    p.add_argument("--model", type=str, default="lgbm",
                   choices=["lgbm", "xgb", "rf", "ridge"])
    p.add_argument("--init-frac", type=float, default=0.5,
                   help="fraction of months used for the initial training window "
                        "(default 0.5 keeps the initial window pre-CBD-fee so the "
                        "Jan-2025 regime change lands in the live stream)")
    p.add_argument("--mae-threshold", type=float, default=0.10,
                   help="relative MAE increase that counts as concept drift")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--tag", type=str, default="walkforward")
    return p.parse_args()


def _mae(model, X, y):
    return float(np.mean(np.abs(y.values - model.predict(X))))


def plot_walkforward(records: pd.DataFrame, output_dir: str, filename: str):
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = records["month"].astype(str)
    ax.plot(x, records["frozen_mae"], marker="o", linewidth=2,
            color="#e15759", label="frozen (never updated)")
    ax.plot(x, records["adaptive_mae"], marker="o", linewidth=2,
            color="#59a14f", label="adaptive (auto-mitigated)")
    # mark months where a mitigation fired
    fired = records[records["strategy"] != "none"]
    ax.scatter(fired["month"].astype(str), fired["adaptive_mae"],
               s=140, facecolors="none", edgecolors="black", linewidths=1.5,
               label="mitigation triggered", zorder=5)
    ax.set_xlabel("Month (walk-forward, time →)")
    ax.set_ylabel("MAE ($)")
    ax.set_title("Walk-forward drift mitigation: frozen vs adaptive model")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = Path(output_dir) / filename
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  Walk-forward curve saved -> {out}")
    return fig


def main():
    args = parse_args()
    OUTPUTS_PLOTS.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load train/val data only
    # ------------------------------------------------------------------
    print("\n=== Loading training data (train/val only) ===")
    raw_df = load_parquet_files(
        DATA_PATHS["training"], n_per_file=args.sample,
        random_state=SAMPLE_CONFIG["random_state"],
    )
    zones_df = load_taxi_zones(DATA_PATHS["taxi_zones"])
    clean_df = clean_training_data(raw_df)

    months = sorted(clean_df.groupby(["pickup_year", "pickup_month"]).groups.keys())
    n_init = max(1, int(len(months) * args.init_frac))
    init_months, live_months = months[:n_init], months[n_init:]
    print(f"  {len(months)} months total | init window: {len(init_months)} | live: {len(live_months)}")

    month_key = clean_df["pickup_year"] * 100 + clean_df["pickup_month"]

    def _slice(month_list):
        keys = {yr * 100 + mo for yr, mo in month_list}
        return clean_df[month_key.isin(keys)]

    def _month_slice(yr, mo):
        return clean_df[month_key == yr * 100 + mo]

    if args.model == "ridge":
        # Ridge needs a scaler and mitigate() retrains tree models; keep this to trees.
        raise SystemExit("Use a tree model (lgbm/xgb/rf) for walk-forward mitigation.")

    # ------------------------------------------------------------------
    # Fit engineer + initial model on the initial window (fixed engineer).
    # Reserve the last in-window month as an out-of-sample reference so the
    # drift threshold is meaningful (in-sample MAE would make every month
    # look like a huge regression and fire mitigation every step).
    # ------------------------------------------------------------------
    init_fit_months, init_ref_month = init_months[:-1], init_months[-1:]
    if not init_fit_months:            # tiny data guard
        init_fit_months, init_ref_month = init_months, init_months[-1:]

    init_fit_df = _slice(init_fit_months)
    X_fit_raw = get_raw_input_features(init_fit_df)
    y_fit = init_fit_df[TARGET_COL].reset_index(drop=True)

    engineer = FeatureEngineer(zones_df)
    engineer.fit(X_fit_raw, y_fit, duration=init_fit_df["trip_duration_min"].reset_index(drop=True))

    def _featurize(df):
        X_raw = get_raw_input_features(df)
        X_feat = engineer.get_tree_features(engineer.transform(X_raw))
        y = df[TARGET_COL].reset_index(drop=True)
        return X_feat, y

    X_fit, y_fit = _featurize(init_fit_df)
    X_ref, y_ref = _featurize(_slice(init_ref_month))
    feature_cols = list(X_fit.columns)

    print(f"\n=== Training initial '{args.model}' model (fit {len(init_fit_months)} mo, "
          f"ref {init_ref_month[0]}) ===")
    frozen_model, _ = train_model(args.model, X_fit, y_fit, X_ref, y_ref)

    ref_mae = _mae(frozen_model, X_ref, y_ref)
    print(f"  Out-of-sample reference MAE: {ref_mae:.4f}")

    # Adaptive model starts identical to the frozen one; training pool grows.
    # X_init/y_init reference the full initial window for drift comparison + pool.
    X_init = pd.concat([X_fit, X_ref], ignore_index=True)
    adaptive_model = frozen_model
    adaptive_cols = list(feature_cols)   # columns the current adaptive model expects
    pool_X = X_init.copy()
    pool_y = pd.concat([y_fit, y_ref], ignore_index=True)

    # ------------------------------------------------------------------
    # Walk forward month by month
    # ------------------------------------------------------------------
    print("\n=== Walk-forward monitoring ===")
    records = []
    for (yr, mo) in live_months:
        month_label = f"{yr}-{mo:02d}"
        m_df = _month_slice(yr, mo)
        if len(m_df) < 50:
            continue
        X_m, y_m = _featurize(m_df)

        frozen_mae = _mae(frozen_model, X_m, y_m)
        # The adaptive model may have been retrained on a reduced feature set
        # (drop_features strategy), so predict on the columns it expects.
        adaptive_mae_pre = _mae(adaptive_model, X_m[adaptive_cols], y_m)

        # ---- course drift detection ----
        feat_drift = detect_feature_drift(X_init, X_m, feature_cols)
        drifted = feat_drift.loc[feat_drift["drift_level"] == "significant", "feature"].tolist()
        drift_results = {"drifted_features": drifted}

        mae_pct = (adaptive_mae_pre - ref_mae) / max(ref_mae, 1e-8)
        concept_results = {
            "concept_drift_detected": bool(mae_pct > args.mae_threshold),
            "mae_pct_increase": mae_pct,
        }

        # ---- choose + apply mitigation (course functions) ----
        strategy = select_mitigation_strategy(drift_results, concept_results)
        if strategy != "none":
            adaptive_model, dropped = mitigate(
                strategy=strategy,
                X_train=pool_X, y_train=pool_y,
                X_recent=X_m, y_recent=y_m,
                model_name=args.model, model_dir=MITIGATED_DIR,
                base_model=adaptive_model,
                drifted_features=drifted,
            )
            # Track the feature space the retrained model expects (drop_features
            # removes columns; reweight_retrain restores the full set).
            adaptive_cols = (
                [c for c in feature_cols if c not in dropped] if dropped else list(feature_cols)
            )

        records.append({
            "month": month_label,
            "n_trips": len(m_df),
            "frozen_mae": frozen_mae,
            "adaptive_mae": adaptive_mae_pre,
            "mae_pct_increase": mae_pct,
            "n_drifted_features": len(drifted),
            "strategy": strategy,
        })
        print(f"  {month_label}: frozen={frozen_mae:.3f}  adaptive={adaptive_mae_pre:.3f}  "
              f"drift={len(drifted)}  strategy={strategy}")

        # accumulate the now-observed month into the training pool
        pool_X = pd.concat([pool_X, X_m], ignore_index=True)
        pool_y = pd.concat([pool_y, y_m], ignore_index=True)

    records_df = pd.DataFrame(records)
    if records_df.empty:
        print("No live months to evaluate.")
        return 0

    records_df.to_csv(LOGS_DIR / f"walkforward_{args.model}.csv", index=False)

    frozen_avg = records_df["frozen_mae"].mean()
    adaptive_avg = records_df["adaptive_mae"].mean()
    improvement = (frozen_avg - adaptive_avg) / frozen_avg * 100
    n_fired = int((records_df["strategy"] != "none").sum())
    print(f"\n  Avg MAE  frozen={frozen_avg:.4f}  adaptive={adaptive_avg:.4f}  "
          f"improvement={improvement:+.1f}%  (mitigations fired: {n_fired})")

    fig = plot_walkforward(records_df, str(OUTPUTS_PLOTS), f"walkforward_{args.model}.png")

    # ------------------------------------------------------------------
    # W&B logging
    # ------------------------------------------------------------------
    if not args.no_wandb:
        print("\n=== Logging to W&B ===")
    tracker = ExperimentTracker(
        project=WANDB_PROJECT,
        run_name=f"walkforward-{args.model}-{args.tag}",
        tags=["drift-walkforward", args.model, args.tag],
        config={
            "model": args.model,
            "init_frac": args.init_frac,
            "mae_threshold": args.mae_threshold,
            "n_live_months": len(records_df),
        },
        enabled=not args.no_wandb,
    )
    tracker.log_summary({
        "frozen_avg_mae": frozen_avg,
        "adaptive_avg_mae": adaptive_avg,
        "improvement_pct": improvement,
        "mitigations_fired": n_fired,
    })
    tracker.log_table(records_df, "walkforward_records")
    tracker.log_plot(fig, "walkforward_curve")
    url = tracker.finish()
    if url:
        print(f"  W&B run -> {url}")

    print("\nWalk-forward mitigation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
