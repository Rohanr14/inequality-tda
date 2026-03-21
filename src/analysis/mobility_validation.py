# src/analysis/mobility_validation.py
# Validates the H₀ gap metric against Chetty et al. absolute upward mobility
# estimates by state. If states with larger income gaps have lower mobility,
# the topological metric captures economically meaningful structure.
#
# Data source: Chetty, Hendren, Kline & Saez (2014) "Where is the Land of
# Opportunity?", state-level absolute upward mobility (mean income rank of
# children born to parents at the 25th percentile).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to src/analysis/)
# ---------------------------------------------------------------------------
RESOLUTION = 101
BASE = Path(__file__).resolve().parent.parent.parent
TIMESERIES_PATH = BASE / "results" / "timeseries" / f"h0_gap_details_{RESOLUTION}pts_timeseries.csv"
MOBILITY_PATH = BASE / "data" / "raw" / "chetty_mobility_by_state.csv"
OUT_DIR = Path(__file__).resolve().parent / "regression"
PLOT_DIR = BASE / "results" / "plots"


def run_mobility_validation() -> None:
    """Correlate mean H₀ gap (real $) per state with Chetty upward mobility."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load pipeline results and mobility data
    ts = pd.read_csv(TIMESERIES_PATH)
    mob = pd.read_csv(MOBILITY_PATH)

    # 2. Compute mean gap per state across all years
    gap_by_state = (
        ts.groupby("state")["acs_longest_h0_lifespan_real"]
        .mean()
        .reset_index()
        .rename(columns={"acs_longest_h0_lifespan_real": "mean_gap_real"})
    )

    # 3. Merge with mobility data
    merged = gap_by_state.merge(mob, on="state", how="inner")
    if merged.empty:
        print("[mobility] No matching states between gap data and mobility data.")
        return

    # 4. Pearson & Spearman correlations
    r_pearson, p_pearson = stats.pearsonr(merged["mean_gap_real"], merged["absolute_upward_mobility"])
    r_spearman, p_spearman = stats.spearmanr(merged["mean_gap_real"], merged["absolute_upward_mobility"])

    # 5. Partial correlation: gap → mobility controlling for Gini
    gini_by_state = ts.groupby("state")["gini"].mean().reset_index()
    merged = merged.merge(gini_by_state, on="state", how="left")

    # Residualise gap and mobility on Gini
    gap_resid = stats.linregress(merged["gini"], merged["mean_gap_real"])
    mob_resid = stats.linregress(merged["gini"], merged["absolute_upward_mobility"])
    gap_r = merged["mean_gap_real"] - (gap_resid.slope * merged["gini"] + gap_resid.intercept)
    mob_r = merged["absolute_upward_mobility"] - (mob_resid.slope * merged["gini"] + mob_resid.intercept)
    r_partial, p_partial = stats.pearsonr(gap_r, mob_r)

    # 6. OLS: mobility ~ gap + gini (multivariate)
    from statsmodels.formula.api import ols
    model = ols("absolute_upward_mobility ~ mean_gap_real + gini", data=merged).fit()

    # 7. Write results
    report = [
        "=" * 70,
        "MOBILITY VALIDATION: H₀ Gap vs Chetty Upward Mobility",
        "=" * 70,
        f"States matched: {len(merged)}",
        "",
        "--- Bivariate Correlations ---",
        f"  Pearson  r = {r_pearson:.4f}  (p = {p_pearson:.4e})",
        f"  Spearman ρ = {r_spearman:.4f}  (p = {p_spearman:.4e})",
        "",
        "--- Partial Correlation (controlling for Gini) ---",
        f"  r(gap, mobility | Gini) = {r_partial:.4f}  (p = {p_partial:.4e})",
        "",
        "--- OLS: mobility ~ gap + gini ---",
        model.summary().as_text(),
        "",
        "Interpretation:",
        "  A significant negative correlation between the H₀ gap and upward",
        "  mobility would suggest that topological income stratification is",
        "  associated with reduced economic opportunity — validating the gap",
        "  metric as economically meaningful beyond what Gini alone captures.",
    ]
    report_text = "\n".join(report)
    with open(OUT_DIR / "mobility_validation.txt", "w") as fh:
        fh.write(report_text)
    print(report_text)

    # 8. Scatter plot: Gap vs Mobility
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Gap vs mobility
    ax = axes[0]
    ax.scatter(merged["mean_gap_real"], merged["absolute_upward_mobility"],
               c=merged["gini"], cmap="RdYlGn_r", edgecolor="k", s=60, alpha=0.8)
    for _, row in merged.iterrows():
        ax.annotate(row["state"][:2], (row["mean_gap_real"], row["absolute_upward_mobility"]),
                     fontsize=6, alpha=0.7)
    # Trend line
    z = np.polyfit(merged["mean_gap_real"], merged["absolute_upward_mobility"], 1)
    x_line = np.linspace(merged["mean_gap_real"].min(), merged["mean_gap_real"].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "r--", lw=1.5, alpha=0.7)
    ax.set(xlabel="Mean H₀ Gap (2024 $)", ylabel="Absolute Upward Mobility",
           title=f"H₀ Gap vs Mobility (r={r_pearson:.3f}, p={p_pearson:.3e})")
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r",
                                norm=plt.Normalize(merged["gini"].min(), merged["gini"].max()))
    plt.colorbar(sm, ax=ax, label="Mean Gini")

    # (b) Partial correlation (residuals)
    ax = axes[1]
    ax.scatter(gap_r, mob_r, c="steelblue", edgecolor="k", s=60, alpha=0.8)
    z2 = np.polyfit(gap_r, mob_r, 1)
    x2 = np.linspace(gap_r.min(), gap_r.max(), 100)
    ax.plot(x2, np.polyval(z2, x2), "r--", lw=1.5, alpha=0.7)
    ax.set(xlabel="H₀ Gap residual (Gini partialled out)",
           ylabel="Mobility residual (Gini partialled out)",
           title=f"Partial correlation (r={r_partial:.3f}, p={p_partial:.3e})")

    fig.suptitle("Validation: Does the H₀ Gap Predict Economic Mobility?", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "mobility_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[mobility] Saved plot → {PLOT_DIR / 'mobility_validation.png'}")
    print(f"[mobility] Saved report → {OUT_DIR / 'mobility_validation.txt'}")


if __name__ == "__main__":
    run_mobility_validation()
