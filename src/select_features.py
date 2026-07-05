"""Feature selection via forward-chaining CV (step 3).

Casts the wide candidate net (~92 features) and keeps the most predictive
subset, using forward-chaining cross-validation as the validation method
throughout — never the test set.

Method
------
1. Forward-chaining CV with ALL candidate features; pick the champion model
   (lowest mean CV RMSE) unless one is fixed with --model.
2. Rank features by mean |SHAP| aggregated across folds (predictive value).
3. Top-K sweep: for K in a grid, restrict to the top-K features and re-run
   forward-chaining CV (reusing the cached per-fold matrices). Record mean RMSE.
4. Keep the smallest K whose CV RMSE is within --tol of the best (a parsimonious
   "most predictive" set). Write it to logs/selected_features_<tag>.json and
   print a config-ready snippet; plot RMSE-vs-K + the SHAP ranking; log to W&B.

Run:
    python select_features.py [--sample N] [--model auto|lgbm|xgb|rf|ridge]
                              [--tol 0.005] [--no-wandb] [--tag select]
"""

import argparse
import json
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
from src.models.shap_analysis import (
    shap_importance, shap_importance_over_folds, plot_shap_bar,
)
from src.tracking.wandb_tracker import ExperimentTracker
from src.train import forward_chain_splits

_ALL_MODELS = ("lgbm", "xgb", "rf", "ridge")
OUTPUTS_PLOTS = Path("outputs/plots")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=60_000,
                   help="rows sampled per monthly file (lower default: CV is run many times)")
    p.add_argument("--model", type=str, default="auto",
                   choices=["auto", "lgbm", "xgb", "rf", "ridge"])
    p.add_argument("--shap-sample", type=int, default=3000)
    p.add_argument("--tol", type=float, default=0.005,
                   help="max relative RMSE loss vs best when picking the smallest K")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--tag", type=str, default="select")
    return p.parse_args()


def _cv_rmse(folds, model_name, cols):
    """Mean forward-chaining CV RMSE for one model on a given column subset."""
    rmses = []
    for (X_tr, y_tr, X_vl, y_vl, scaler, _shap_X) in folds:
        use = [c for c in cols if c in X_tr.columns]
        _, metrics = train_model(
            model_name, X_tr[use], y_tr, X_vl[use], y_vl,
            scaler=build_ridge_scaler(X_tr[use]) if model_name == "ridge" else scaler,
        )
        rmses.append(metrics["rmse"])
    return float(np.mean(rmses))


def main():
    args = parse_args()
    OUTPUTS_PLOTS.mkdir(parents=True, exist_ok=True)

    # Selection must evaluate the FULL candidate net, regardless of any subset
    # currently pinned in config.SELECTED_FEATURES.
    import src.config as _cfg
    _cfg.SELECTED_FEATURES = None

    print("\n=== Loading training data (train/val only) ===")
    raw_df = load_parquet_files(
        DATA_PATHS["training"], n_per_file=args.sample,
        random_state=SAMPLE_CONFIG["random_state"],
    )
    zones_df = load_taxi_zones(DATA_PATHS["taxi_zones"])
    clean_df = clean_training_data(raw_df)

    # ------------------------------------------------------------------
    # Pass 1 — cache per-fold engineered matrices; train all models to
    # pick the champion; collect per-fold champion candidates for SHAP.
    # ------------------------------------------------------------------
    print("\n=== Forward-chaining CV on the full candidate set ===")
    folds = []                       # (X_tr, y_tr, X_vl, y_vl, scaler, shap_X)
    fold_labels = []
    fold_models = []                 # per fold: {model_name: fitted_model}
    rmse_by_model = {m: [] for m in _ALL_MODELS}
    all_features = None

    for fold, tr_df, vl_df in forward_chain_splits(clean_df, n_splits=5):
        fold_labels.append(f"F{fold + 1}")
        X_tr_raw, X_vl_raw = get_raw_input_features(tr_df), get_raw_input_features(vl_df)
        y_tr = tr_df[TARGET_COL].reset_index(drop=True)
        y_vl = vl_df[TARGET_COL].reset_index(drop=True)

        eng = FeatureEngineer(zones_df)
        eng.fit(X_tr_raw, y_tr, duration=tr_df["trip_duration_min"].reset_index(drop=True))
        X_tr = eng.get_tree_features(eng.transform(X_tr_raw))
        X_vl = eng.get_tree_features(eng.transform(X_vl_raw))
        all_features = list(X_tr.columns)
        scaler = build_ridge_scaler(X_tr)
        shap_X = X_vl.sample(min(args.shap_sample, len(X_vl)), random_state=42)

        models = {}
        for name in _ALL_MODELS:
            model, metrics = train_model(name, X_tr, y_tr, X_vl, y_vl, scaler=scaler)
            models[name] = model
            rmse_by_model[name].append(metrics["rmse"])
        fold_models.append(models)
        folds.append((X_tr, y_tr, X_vl, y_vl, scaler, shap_X))

    mean_rmse = {m: float(np.mean(rmse_by_model[m])) for m in _ALL_MODELS}
    champion = args.model if args.model != "auto" else min(mean_rmse, key=mean_rmse.get)
    full_rmse = mean_rmse[champion]
    print("\n  Full-set CV RMSE:  " +
          "  ".join(f"{m}={mean_rmse[m]:.4f}" for m in _ALL_MODELS))
    print(f"  Champion: {champion}  (full-set RMSE {full_rmse:.4f}, {len(all_features)} features)")

    # ------------------------------------------------------------------
    # Rank features by mean |SHAP| across folds (champion model)
    # ------------------------------------------------------------------
    print("\n=== Ranking features by TreeSHAP over folds ===")
    fold_importances = {}
    for label, models, (_, _, _, _, _, shap_X) in zip(fold_labels, fold_models, folds):
        fold_importances[label] = shap_importance(
            models[champion], shap_X, list(shap_X.columns)
        )
    shap_wide = shap_importance_over_folds(fold_importances)
    ranking = list(shap_wide.index)             # most predictive first
    print(f"  Top 10: {ranking[:10]}")

    # ------------------------------------------------------------------
    # Pass 2 — top-K sweep, validated by forward-chaining CV
    # ------------------------------------------------------------------
    print("\n=== Top-K sweep (forward-chaining CV) ===")
    n_feat = len(ranking)
    grid = sorted({k for k in [8, 12, 16, 20, 25, 30, 40, 55, 70, n_feat] if k <= n_feat})
    sweep = []
    for k in grid:
        rmse_k = _cv_rmse(folds, champion, ranking[:k])
        sweep.append({"k": k, "cv_rmse": rmse_k})
        print(f"  K={k:>3}  CV RMSE={rmse_k:.4f}")
    sweep_df = pd.DataFrame(sweep)

    best_rmse = sweep_df["cv_rmse"].min()
    threshold = best_rmse * (1.0 + args.tol)
    ok = sweep_df[sweep_df["cv_rmse"] <= threshold]
    best_k = int(ok["k"].min())
    selected = ranking[:best_k]

    print(f"\n  Best CV RMSE {best_rmse:.4f} (tol {args.tol:.1%} -> {threshold:.4f})")
    print(f"  Selected K={best_k}  ({best_k}/{n_feat} features, "
          f"RMSE {float(ok.loc[ok['k'] == best_k, 'cv_rmse'].iloc[0]):.4f})")

    # ------------------------------------------------------------------
    # Persist selection + plots
    # ------------------------------------------------------------------
    out_json = LOGS_DIR / "selected_features.json"
    out_json.write_text(json.dumps({
        "champion": champion,
        "best_k": best_k,
        "full_rmse": full_rmse,
        "selected_rmse": best_rmse,
        "selected_features": selected,
        "ranking": ranking,
        "sweep": sweep,
    }, indent=2))
    print(f"  Selection written -> {out_json}")
    print("\n  Paste into src/config.py:\n  SELECTED_FEATURES = " + json.dumps(selected))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(sweep_df["k"], sweep_df["cv_rmse"], marker="o")
    ax.axhline(threshold, ls="--", color="grey", label=f"best +{args.tol:.1%}")
    ax.axvline(best_k, ls=":", color="green", label=f"selected K={best_k}")
    ax.set_xlabel("Number of top features (by SHAP)")
    ax.set_ylabel("Forward-chaining CV RMSE")
    ax.set_title(f"Feature-count vs CV RMSE — {champion}")
    ax.legend()
    plt.tight_layout()
    sweep_png = OUTPUTS_PLOTS / "feature_selection_sweep.png"
    fig.savefig(sweep_png, dpi=120, bbox_inches="tight")
    print(f"  Sweep plot -> {sweep_png}")

    shap_bar_df = (
        shap_wide.reset_index()[["feature", "mean_pct"]]
        .rename(columns={"mean_pct": "mean_abs_shap"})
    )
    bar_fig = plot_shap_bar(
        shap_bar_df, top_n=min(30, n_feat), output_dir=str(OUTPUTS_PLOTS),
        filename="feature_selection_shap.png",
        title=f"Candidate feature importance (SHAP over folds) — {champion}",
    )

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    tracker = ExperimentTracker(
        project=WANDB_PROJECT,
        run_name=f"feature-selection-{args.tag}",
        tags=["feature-selection", args.tag, f"champion-{champion}"],
        config={"n_candidate_features": n_feat, "champion": champion, "tol": args.tol},
        enabled=not args.no_wandb,
    )
    tracker.log_summary({
        "champion": champion,
        "full_rmse": full_rmse,
        "selected_rmse": best_rmse,
        "best_k": best_k,
        "n_candidate_features": n_feat,
    })
    tracker.log_table(sweep_df, "kfeatures_sweep")
    tracker.log_table(shap_wide.reset_index(), "shap_ranking")
    tracker.log_plot(fig, "feature_selection_sweep")
    if bar_fig is not None:
        tracker.log_plot(bar_fig, "feature_selection_shap")
    url = tracker.finish()
    if url:
        print(f"  W&B run -> {url}")

    print("\nFeature selection complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
