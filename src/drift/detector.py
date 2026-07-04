"""Drift detection: data drift (PSI, KS test), performance drift, and monthly monitoring."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Population Stability Index (PSI)
# ---------------------------------------------------------------------------

def compute_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
    epsilon: float = 1e-8,
) -> float:
    """Compute the Population Stability Index between two distributions.

    Interpretation (for well-behaved, continuous features):
        PSI < 0.10  → no significant shift
        0.10–0.25   → moderate shift
        > 0.25      → significant shift

    IMPORTANT: PSI is bounded below by 0 (every summed term is non-negative)
    but has NO upper bound of 1 — that is a common misconception. For a
    binary/step-like feature that goes from 100% one value in `reference` to
    100% a different value in `current` (complete separation, e.g. a flag
    that is always 0 pre-cutoff and always 1 post-cutoff), the epsilon-
    smoothed formula above genuinely produces PSI ~ -2*ln(epsilon), which is
    ~36.8 for epsilon=1e-8. This is mathematically correct, not a bug — it
    means "this feature's distribution completely changed," the most extreme
    drift possible. The 0.1/0.25 thresholds still apply directionally (higher
    = more drift) but shouldn't be read as if PSI were capped near 1.
    """
    combined = pd.concat([reference, current])
    breakpoints = np.linspace(combined.min(), combined.max(), bins + 1)
    breakpoints[0] -= 1e-9
    breakpoints[-1] += 1e-9

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = ref_counts / len(reference) + epsilon
    cur_pct = cur_counts / len(current) + epsilon

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


# ---------------------------------------------------------------------------
# KS test
# ---------------------------------------------------------------------------

def compute_ks_test(
    reference: pd.Series,
    current: pd.Series,
) -> Tuple[float, float]:
    """Two-sample KS test. Returns (statistic, p_value)."""
    stat, p_val = stats.ks_2samp(reference.dropna(), current.dropna())
    return float(stat), float(p_val)


# ---------------------------------------------------------------------------
# Feature drift report
# ---------------------------------------------------------------------------

def detect_feature_drift(
    df_ref: pd.DataFrame,
    df_curr: pd.DataFrame,
    feature_cols: List[str],
    psi_bins: int = 10,
) -> pd.DataFrame:
    """Generate a drift report for each feature column.

    Returns a DataFrame with columns:
      feature, psi, ks_stat, ks_pval, drift_level
    """
    records = []
    for col in feature_cols:
        if col not in df_ref.columns or col not in df_curr.columns:
            continue
        ref_col = pd.to_numeric(df_ref[col], errors="coerce").dropna()
        cur_col = pd.to_numeric(df_curr[col], errors="coerce").dropna()
        if len(ref_col) == 0 or len(cur_col) == 0:
            continue

        psi = compute_psi(ref_col, cur_col, bins=psi_bins)
        ks_stat, ks_pval = compute_ks_test(ref_col, cur_col)

        if psi < 0.10:
            level = "stable"
        elif psi < 0.25:
            level = "moderate"
        else:
            level = "significant"

        records.append({
            "feature": col,
            "psi": round(psi, 4),
            "ks_stat": round(ks_stat, 4),
            "ks_pval": round(ks_pval, 4),
            "drift_level": level,
        })

    return pd.DataFrame(records).sort_values("psi", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Performance drift over time
# ---------------------------------------------------------------------------

def detect_performance_drift(
    y_true: pd.Series,
    y_pred: np.ndarray,
    meta_df: pd.DataFrame,
    time_col: str = "pickup_month",
) -> pd.DataFrame:
    """Compute RMSE and MAE per time period to detect concept drift."""
    from src.models.trainer import compute_metrics

    meta = meta_df.copy().reset_index(drop=True)
    y_t = y_true.reset_index(drop=True)
    y_p = pd.Series(y_pred).reset_index(drop=True)

    records = []
    for period, idx in meta.groupby(time_col).groups.items():
        m = compute_metrics(y_t.iloc[idx].values, y_p.iloc[idx].values)
        m[time_col] = period
        m["count"] = len(idx)
        records.append(m)

    return pd.DataFrame(records).sort_values(time_col).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full drift report
# ---------------------------------------------------------------------------

def generate_drift_report(
    df_ref: pd.DataFrame,
    df_curr: pd.DataFrame,
    feature_cols: List[str],
    y_true_ref: Optional[pd.Series] = None,
    y_pred_ref: Optional[np.ndarray] = None,
    y_true_curr: Optional[pd.Series] = None,
    y_pred_curr: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Produce a comprehensive drift report comparing reference vs current data.

    Parameters
    ----------
    df_ref   : training data (e.g. last 2 months of 2025)
    df_curr  : test / production data (e.g. 2026 Jan-Feb)
    """
    from src.models.trainer import compute_metrics

    report: Dict[str, Any] = {}
    report["feature_drift"] = detect_feature_drift(df_ref, df_curr, feature_cols)

    n_significant = (report["feature_drift"]["drift_level"] == "significant").sum()
    report["summary"] = {
        "n_features_checked": len(report["feature_drift"]),
        "n_significant_drift": int(n_significant),
        "n_moderate_drift": int((report["feature_drift"]["drift_level"] == "moderate").sum()),
    }

    if y_true_ref is not None and y_pred_ref is not None:
        report["ref_metrics"] = compute_metrics(y_true_ref.values, y_pred_ref)

    if y_true_curr is not None and y_pred_curr is not None:
        report["curr_metrics"] = compute_metrics(y_true_curr.values, y_pred_curr)

    return report


# ---------------------------------------------------------------------------
# Monthly drift monitoring
# ---------------------------------------------------------------------------

def load_monthly_eval(path: str) -> pd.DataFrame:
    """Load a parquet file containing trips from multiple months."""
    df = pd.read_parquet(path)
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    return df


def run_monthly_drift_analysis(
    monthly_eval_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    engineer: Any,
    model: Any,
    output_dir: str = "outputs/plots",
    ref_model_mae: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute per-month drift metrics vs a reference period.

    Parameters
    ----------
    monthly_eval_df : cleaned DataFrame with pickup_year, pickup_month, TARGET_COL.
    reference_df    : cleaned reference DataFrame (e.g. validation split).
    engineer        : fitted FeatureEngineer.
    model           : fitted model.
    ref_model_mae   : known reference MAE; if None it is computed from reference_df.
    """
    from src.config import TARGET_COL
    from src.features.engineer import get_raw_input_features

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- reference MAE ---
    if ref_model_mae is None:
        X_ref_raw = get_raw_input_features(reference_df)
        X_ref_eng = engineer.transform(X_ref_raw)
        X_ref_feat = engineer.get_tree_features(X_ref_eng)
        y_ref = reference_df[TARGET_COL].reset_index(drop=True)
        y_ref_pred = model.predict(X_ref_feat)
        ref_model_mae = float(np.mean(np.abs(y_ref.values - y_ref_pred)))

    # reference target distribution for PSI
    X_ref_raw_all = get_raw_input_features(reference_df)
    X_ref_eng_all = engineer.transform(X_ref_raw_all)
    ref_target = reference_df[TARGET_COL].reset_index(drop=True)

    feature_cols = [c for c in engineer.get_feature_names() if c in X_ref_eng_all.columns]

    records = []
    drift_reports: Dict[str, Any] = {}

    month_groups = (
        monthly_eval_df
        .groupby(["pickup_year", "pickup_month"])
        .groups
    )

    for (yr, mo), idx in sorted(month_groups.items()):
        month_df = monthly_eval_df.loc[idx].reset_index(drop=True)
        month_label = month_df["tpep_pickup_datetime"].iloc[0].strftime("%b") if "tpep_pickup_datetime" in month_df.columns else f"{yr}-{mo:02d}"
        month_num = int(mo)

        X_cur_raw = get_raw_input_features(month_df)
        X_cur_eng = engineer.transform(X_cur_raw)
        X_cur_feat = engineer.get_tree_features(X_cur_eng)
        y_cur = month_df[TARGET_COL].reset_index(drop=True)

        y_pred = model.predict(X_cur_feat)
        mae = float(np.mean(np.abs(y_cur.values - y_pred)))
        mae_delta = mae - ref_model_mae
        mae_pct = mae_delta / max(ref_model_mae, 1e-8)

        # label drift
        label_psi = compute_psi(ref_target, y_cur)
        label_ks_stat, label_ks_pval = compute_ks_test(ref_target, y_cur)
        label_drifted = label_psi > 0.10

        # feature drift
        feat_drift = detect_feature_drift(X_ref_eng_all, X_cur_eng, feature_cols)
        n_drifted = int((feat_drift["drift_level"] == "significant").sum())

        rec = {
            "month": month_label,
            "month_num": month_num,
            "year": int(yr),
            "mae": mae,
            "mae_delta": mae_delta,
            "mae_pct_increase": mae_pct * 100,
            "n_trips": len(month_df),
            "label_psi": label_psi,
            "label_ks_pvalue": label_ks_pval,
            "label_drifted": label_drifted,
            "label_ref_mean": float(ref_target.mean()),
            "label_cur_mean": float(y_cur.mean()),
            "n_drifted_features": n_drifted,
        }
        records.append(rec)

        drift_reports[month_label] = {
            "feature_drift": feat_drift,
            "summary": {
                "n_features_checked": len(feat_drift),
                "n_significant_drift": n_drifted,
                "n_moderate_drift": int((feat_drift["drift_level"] == "moderate").sum()),
            },
        }

        print(
            f"  {month_label} {yr}: MAE={mae:.2f} (ref={ref_model_mae:.2f}, "
            f"delta={mae_delta:+.2f})  label_drift={label_drifted}  "
            f"feat_drifted={n_drifted}"
        )

    monthly_summary = pd.DataFrame(records)
    return monthly_summary, drift_reports


def plot_monthly_mae_curve(
    monthly_summary: pd.DataFrame,
    output_dir: str = "outputs/plots",
) -> Any:
    """Line chart of MAE over months."""
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly_summary["month"], monthly_summary["mae"], marker="o", linewidth=2)
    ax.axhline(
        monthly_summary["mae"].iloc[0] if len(monthly_summary) else 0,
        linestyle="--", color="grey", label="reference MAE",
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("MAE ($)")
    ax.set_title("Model MAE Over Time (concept drift curve)")
    ax.legend()
    plt.tight_layout()

    out = Path(output_dir) / "monthly_mae_curve.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  MAE curve saved -> {out}")
    return fig


def plot_feature_drift_over_folds(
    drift_long: pd.DataFrame,
    top_n: int = 8,
    output_dir: str = "outputs/plots",
    filename: str = "feature_drift_over_folds.png",
) -> Any:
    """Line chart of per-feature PSI across forward-chaining folds.

    Parameters
    ----------
    drift_long : tidy DataFrame with columns [fold, feature, psi].
    top_n      : number of most-drifting features (by peak PSI) to show.
    """
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if drift_long.empty:
        return None

    peak = drift_long.groupby("feature")["psi"].max().sort_values(ascending=False)
    top_features = list(peak.head(top_n).index)
    folds = sorted(drift_long["fold"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    for feature in top_features:
        sub = drift_long[drift_long["feature"] == feature].set_index("fold").reindex(folds)
        ax.plot([str(f) for f in folds], sub["psi"], marker="o", linewidth=2, label=feature)

    ax.axhline(0.10, linestyle="--", color="grey", alpha=0.7, label="moderate (0.10)")
    ax.axhline(0.25, linestyle="--", color="red", alpha=0.7, label="significant (0.25)")
    ax.set_xlabel("Forward-chaining fold (time →)")
    ax.set_ylabel("PSI")
    ax.set_title("Feature drift over time (PSI per fold)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    out = Path(output_dir) / filename
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  Feature-drift-over-folds chart saved -> {out}")
    return fig


def plot_psi_heatmap(
    drift_long: pd.DataFrame,
    output_dir: str = "outputs/plots",
    filename: str = "psi_heatmap.png",
    vmax: float = 1.0,
    max_features: Optional[int] = None,
) -> Any:
    """Feature x fold PSI heatmap — full view for manual drift investigation.

    Unlike plot_feature_drift_over_folds (which only shows the top-N line
    traces), this shows every feature as a row so you can scan the whole
    candidate set for "what's drifting and where" at a glance.

    Color scale is capped at *vmax* (default 1.0, well above the 0.25
    "significant" threshold) so a few extreme outliers — e.g. a step-function
    feature that goes from 100%-0 to 100%-1, which can score PSI in the tens
    (see compute_psi docstring) — don't wash out the color contrast for every
    other feature. Any cell whose true value exceeds vmax is annotated with
    its exact number so the magnitude is still visible, not just "maxed out".

    Parameters
    ----------
    drift_long   : tidy DataFrame with columns [fold, feature, psi].
    vmax         : color-scale cap. Values above this print in white text.
    max_features : cap the number of rows (most-drifting first); None = all.
    """
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if drift_long.empty:
        return None

    peak = drift_long.groupby("feature")["psi"].max().sort_values(ascending=False)
    features = list(peak.index) if max_features is None else list(peak.head(max_features).index)
    folds = sorted(drift_long["fold"].unique())

    pivot = (
        drift_long[drift_long["feature"].isin(features)]
        .pivot(index="feature", columns="fold", values="psi")
        .reindex(index=features, columns=folds)
    )

    fig_h = max(6.0, 0.24 * len(features))
    fig_w = max(8.0, 1.3 * len(folds) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels([str(f) for f in folds])
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=7)
    ax.set_xlabel("Fold (time →)")
    ax.set_title("PSI heatmap — feature x fold (color capped at "
                  f"{vmax:g}; exact value shown when clipped)")

    # Annotate only clipped (>vmax) cells with their true value so extreme
    # drift is still legible without needing every one of ~500 cells labeled.
    for i, feature in enumerate(features):
        for j, fold in enumerate(folds):
            val = pivot.values[i, j]
            if np.isfinite(val) and val > vmax:
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=6, color="black",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7))

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(f"PSI (capped at {vmax:g})")
    plt.tight_layout()

    out = Path(output_dir) / filename
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  PSI heatmap saved -> {out}  ({len(features)} features x {len(folds)} folds)")
    return fig


def plot_label_drift_distribution(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    ref_label: str = "Reference",
    cur_label: str = "Current",
    output_dir: str = "outputs/plots",
) -> Any:
    """Overlapping histogram of target distributions (reference vs current)."""
    import matplotlib.pyplot as plt
    from src.config import TARGET_COL

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ref_vals = reference_df[TARGET_COL].dropna()
    cur_vals = current_df[TARGET_COL].dropna()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ref_vals, bins=50, alpha=0.5, label=ref_label, density=True)
    ax.hist(cur_vals, bins=50, alpha=0.5, label=cur_label, density=True)
    ax.set_xlabel("Total Fare Amount ($)")
    ax.set_ylabel("Density")
    ax.set_title("Label Drift: Fare Distribution Shift")
    ax.legend()
    plt.tight_layout()

    out = Path(output_dir) / "label_drift_distribution.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  Label drift plot saved -> {out}")
    return fig
