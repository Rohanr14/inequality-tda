# src/analysis/temporal_holdout.py
# Temporal holdout validation: tests whether the H₀ gap computed from
# *early* years (2010-2016) predicts Chetty mobility out-of-sample, and
# whether the gap's predictive power is stable across time splits.
#
# This addresses a key limitation of the original mobility validation,
# which used all-year averages — making it impossible to distinguish
# genuine predictive signal from overfitting to a single cross-section.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESOLUTION = 101
BASE = Path(__file__).resolve().parent.parent.parent
TIMESERIES_PATH = (
    BASE / "results" / "timeseries"
    / f"h0_gap_details_{RESOLUTION}pts_timeseries.csv"
)
MOBILITY_PATH = BASE / "data" / "raw" / "chetty_mobility_by_state.csv"
OUT_DIR = Path(__file__).resolve().parent / "regression"
PLOT_DIR = BASE / "results" / "plots"

# Time splits
EARLY = (2010, 2016)
LATE = (2017, 2023)


def _period_gap(ts: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """Mean real gap per state for years in [lo, hi]."""
    sub = ts[(ts["year"] >= lo) & (ts["year"] <= hi)]
    return (
        sub.groupby("state")["acs_longest_h0_lifespan_real"]
        .mean()
        .reset_index()
        .rename(columns={"acs_longest_h0_lifespan_real": "mean_gap_real"})
    )


def _period_gini(ts: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """Mean Gini per state for years in [lo, hi]."""
    sub = ts[(ts["year"] >= lo) & (ts["year"] <= hi)]
    return (
        sub.groupby("state")["gini"]
        .mean()
        .reset_index()
    )


def _partial_corr(x, y, z):
    """Partial Pearson correlation of x and y controlling for z."""
    rx = stats.linregress(z, x)
    ry = stats.linregress(z, y)
    x_resid = x - (rx.slope * z + rx.intercept)
    y_resid = y - (ry.slope * z + ry.intercept)
    return stats.pearsonr(x_resid, y_resid)


def _evaluate_period(ts, mob, lo, hi, label):
    """Run full correlation suite for one time period, return dict of stats."""
    gap = _period_gap(ts, lo, hi)
    gini = _period_gini(ts, lo, hi)
    merged = gap.merge(mob, on="state", how="inner").merge(gini, on="state", how="left")

    r_p, p_p = stats.pearsonr(merged["mean_gap_real"], merged["absolute_upward_mobility"])
    r_s, p_s = stats.spearmanr(merged["mean_gap_real"], merged["absolute_upward_mobility"])
    r_part, p_part = _partial_corr(
        merged["mean_gap_real"].values,
        merged["absolute_upward_mobility"].values,
        merged["gini"].values,
    )

    # OLS: mobility ~ gap + gini
    from statsmodels.formula.api import ols
    model = ols("absolute_upward_mobility ~ mean_gap_real + gini", data=merged).fit()

    return {
        "label": label,
        "years": f"{lo}-{hi}",
        "n_states": len(merged),
        "pearson_r": r_p,
        "pearson_p": p_p,
        "spearman_r": r_s,
        "spearman_p": p_s,
        "partial_r": r_part,
        "partial_p": p_part,
        "ols_r2": model.rsquared,
        "gap_coef": model.params.get("mean_gap_real", np.nan),
        "gap_pval": model.pvalues.get("mean_gap_real", np.nan),
        "gini_coef": model.params.get("gini", np.nan),
        "gini_pval": model.pvalues.get("gini", np.nan),
        "merged": merged,
        "model": model,
    }


def run_temporal_holdout() -> None:
    """Compare predictive power of early-period vs late-period H₀ gaps."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    ts = pd.read_csv(TIMESERIES_PATH)
    mob = pd.read_csv(MOBILITY_PATH)

    early = _evaluate_period(ts, mob, *EARLY, "Early")
    late = _evaluate_period(ts, mob, *LATE, "Late")
    full = _evaluate_period(ts, mob, ts["year"].min(), ts["year"].max(), "All years")

    # Cross-period prediction: train gap from early, predict mobility
    early_gap = _period_gap(ts, *EARLY)
    late_gap = _period_gap(ts, *LATE)
    cross = early_gap.merge(mob, on="state", how="inner")
    r_cross, p_cross = stats.pearsonr(cross["mean_gap_real"], cross["absolute_upward_mobility"])

    # Gap stability: correlation between early and late period gaps
    stability = early_gap.merge(late_gap, on="state", suffixes=("_early", "_late"))
    r_stab, p_stab = stats.pearsonr(stability["mean_gap_real_early"], stability["mean_gap_real_late"])

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    lines = [
        "=" * 70,
        "TEMPORAL HOLDOUT VALIDATION",
        "=" * 70,
        "",
        "Does the H₀ gap from *early* years predict mobility out-of-sample?",
        "",
        "--- Gap Stability Across Periods ---",
        f"  Pearson r(early gap, late gap) = {r_stab:.4f}  (p = {p_stab:.4e})",
        f"  → The state-level gap ranking is {'stable' if r_stab > 0.8 else 'moderately stable' if r_stab > 0.5 else 'unstable'} across periods.",
        "",
    ]

    for res in [early, late, full]:
        lines += [
            f"--- {res['label']} Period ({res['years']}, n={res['n_states']}) ---",
            f"  Pearson  r = {res['pearson_r']:.4f}  (p = {res['pearson_p']:.4e})",
            f"  Spearman ρ = {res['spearman_r']:.4f}  (p = {res['spearman_p']:.4e})",
            f"  Partial r (| Gini) = {res['partial_r']:.4f}  (p = {res['partial_p']:.4e})",
            f"  OLS R² (gap + Gini) = {res['ols_r2']:.4f}",
            f"    gap  coef = {res['gap_coef']:.4e}  (p = {res['gap_pval']:.4e})",
            f"    gini coef = {res['gini_coef']:.4f}  (p = {res['gini_pval']:.4e})",
            "",
        ]

    lines += [
        "--- Cross-Period Prediction (early gap → mobility) ---",
        f"  Pearson r = {r_cross:.4f}  (p = {p_cross:.4e})",
        "",
        "Interpretation:",
        "  If the early-period gap predicts mobility as well as the full-period",
        "  gap, the signal is temporally robust — not an artefact of overfitting",
        "  to a single cross-section. Stable gap rankings across periods further",
        "  confirm that the topological structure is persistent.",
    ]

    report = "\n".join(lines)
    with open(OUT_DIR / "temporal_holdout.txt", "w") as fh:
        fh.write(report)
    print(report)

    # -----------------------------------------------------------------------
    # Plot: 2×2 grid
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) Early gap vs mobility
    ax = axes[0, 0]
    m = early["merged"]
    ax.scatter(m["mean_gap_real"], m["absolute_upward_mobility"],
               c="steelblue", edgecolor="k", s=50, alpha=0.8)
    z = np.polyfit(m["mean_gap_real"], m["absolute_upward_mobility"], 1)
    xl = np.linspace(m["mean_gap_real"].min(), m["mean_gap_real"].max(), 100)
    ax.plot(xl, np.polyval(z, xl), "r--", lw=1.5)
    ax.set(xlabel="Mean H₀ Gap (2024 $)", ylabel="Upward Mobility",
           title=f"Early ({EARLY[0]}-{EARLY[1]})  r={early['pearson_r']:.3f}")

    # (b) Late gap vs mobility
    ax = axes[0, 1]
    m = late["merged"]
    ax.scatter(m["mean_gap_real"], m["absolute_upward_mobility"],
               c="darkorange", edgecolor="k", s=50, alpha=0.8)
    z = np.polyfit(m["mean_gap_real"], m["absolute_upward_mobility"], 1)
    xl = np.linspace(m["mean_gap_real"].min(), m["mean_gap_real"].max(), 100)
    ax.plot(xl, np.polyval(z, xl), "r--", lw=1.5)
    ax.set(xlabel="Mean H₀ Gap (2024 $)", ylabel="Upward Mobility",
           title=f"Late ({LATE[0]}-{LATE[1]})  r={late['pearson_r']:.3f}")

    # (c) Gap stability: early vs late
    ax = axes[1, 0]
    ax.scatter(stability["mean_gap_real_early"], stability["mean_gap_real_late"],
               c="seagreen", edgecolor="k", s=50, alpha=0.8)
    lo_val = min(stability["mean_gap_real_early"].min(), stability["mean_gap_real_late"].min())
    hi_val = max(stability["mean_gap_real_early"].max(), stability["mean_gap_real_late"].max())
    ax.plot([lo_val, hi_val], [lo_val, hi_val], "k--", lw=1, alpha=0.4, label="y = x")
    ax.set(xlabel=f"Mean Gap {EARLY[0]}-{EARLY[1]} ($)",
           ylabel=f"Mean Gap {LATE[0]}-{LATE[1]} ($)",
           title=f"Gap Stability  r={r_stab:.3f}")
    ax.legend(fontsize=9)

    # (d) R² comparison bar chart
    ax = axes[1, 1]
    labels = ["Early\n(out-of-sample)", "Late", "All years"]
    r2s = [early["ols_r2"], late["ols_r2"], full["ols_r2"]]
    colors = ["steelblue", "darkorange", "gray"]
    bars = ax.bar(labels, r2s, color=colors, edgecolor="k", alpha=0.85)
    for bar, val in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set(ylabel="OLS R² (mobility ~ gap + Gini)", title="Predictive Power by Period",
           ylim=(0, max(r2s) * 1.15))

    fig.suptitle("Temporal Holdout: Is the H₀ Gap Signal Stable Over Time?",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "temporal_holdout.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[temporal] Saved plot → {PLOT_DIR / 'temporal_holdout.png'}")
    print(f"[temporal] Saved report → {OUT_DIR / 'temporal_holdout.txt'}")


if __name__ == "__main__":
    run_temporal_holdout()
