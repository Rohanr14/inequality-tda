# src/analysis/cross_outcome_validation.py
# Validates the H₀ gap against multiple external economic outcomes beyond
# Chetty mobility: state-level poverty rate and median household income.
#
# This addresses a key limitation: the original validation relied on a single
# external dataset. If the gap predicts multiple distinct outcomes (controlling
# for Gini), the "complementary signal" claim is substantially stronger.
#
# Data sources:
#   - Poverty rate: US Census SAIPE (Small Area Income & Poverty Estimates),
#     loaded from bundled CSV or computed from ACS S1701.
#   - Median household income: ACS B19013, loaded from bundled CSV.
#   - Chetty mobility: already in data/raw/chetty_mobility_by_state.csv

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
OUTCOMES_PATH = BASE / "data" / "raw" / "state_economic_outcomes.csv"
OUT_DIR = Path(__file__).resolve().parent / "regression"
PLOT_DIR = BASE / "results" / "plots"

# SAIPE poverty rates and ACS median income by state (2019 snapshot).
# Bundled as a fallback when the CSV is absent — avoids an API dependency.
# Sources: Census SAIPE 2019, ACS 1-Year 2019 B19013.
_FALLBACK_OUTCOMES = {
    "Alabama": (15.5, 51734),
    "Alaska": (10.1, 75463),
    "Arizona": (14.1, 62055),
    "Arkansas": (16.2, 48952),
    "California": (11.8, 80440),
    "Colorado": (9.3, 77127),
    "Connecticut": (9.8, 78444),
    "Delaware": (11.3, 70176),
    "District of Columbia": (13.5, 92266),
    "Florida": (12.7, 59227),
    "Georgia": (13.5, 61980),
    "Hawaii": (9.3, 83102),
    "Idaho": (11.2, 60999),
    "Illinois": (11.5, 69187),
    "Indiana": (11.9, 57603),
    "Iowa": (11.2, 61691),
    "Kansas": (11.4, 62087),
    "Kentucky": (16.3, 52295),
    "Louisiana": (19.0, 51073),
    "Maine": (11.6, 58924),
    "Maryland": (9.0, 86738),
    "Massachusetts": (10.3, 85843),
    "Michigan": (13.0, 59584),
    "Minnesota": (9.0, 74593),
    "Mississippi": (19.6, 45081),
    "Missouri": (12.9, 57409),
    "Montana": (12.6, 57153),
    "Nebraska": (10.8, 63229),
    "Nevada": (12.5, 63276),
    "New Hampshire": (7.3, 77933),
    "New Jersey": (9.4, 85751),
    "New Mexico": (18.2, 51945),
    "New York": (13.1, 72108),
    "North Carolina": (13.6, 57341),
    "North Dakota": (10.6, 65315),
    "Ohio": (13.1, 58642),
    "Oklahoma": (15.2, 54449),
    "Oregon": (11.4, 67058),
    "Pennsylvania": (11.8, 63463),
    "Rhode Island": (10.3, 67167),
    "South Carolina": (13.8, 56227),
    "South Dakota": (13.1, 59533),
    "Tennessee": (13.9, 56071),
    "Texas": (13.6, 64034),
    "Utah": (8.9, 74197),
    "Vermont": (10.2, 63001),
    "Virginia": (9.9, 76456),
    "Washington": (10.3, 78687),
    "West Virginia": (16.0, 48850),
    "Wisconsin": (10.4, 63293),
    "Wyoming": (9.6, 65003),
}


def _load_outcomes() -> pd.DataFrame:
    """Load state-level poverty rate and median income."""
    if OUTCOMES_PATH.exists():
        return pd.read_csv(OUTCOMES_PATH)
    # Use bundled fallback
    rows = [
        {"state": st, "poverty_rate": pov, "median_income": med}
        for st, (pov, med) in _FALLBACK_OUTCOMES.items()
    ]
    df = pd.DataFrame(rows)
    # Save for future runs
    OUTCOMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTCOMES_PATH, index=False)
    print(f"[cross-outcome] Created {OUTCOMES_PATH} from bundled data.")
    return df


def _partial_corr(x, y, z):
    """Partial Pearson correlation of x and y controlling for z."""
    rx = stats.linregress(z, x)
    ry = stats.linregress(z, y)
    x_resid = x - (rx.slope * z + rx.intercept)
    y_resid = y - (ry.slope * z + ry.intercept)
    return stats.pearsonr(x_resid, y_resid)


def _analyse_outcome(merged, outcome_col, outcome_label):
    """Compute bivariate, partial, and OLS stats for one outcome."""
    gap = merged["mean_gap_real"].values
    y = merged[outcome_col].values
    gini = merged["gini"].values

    r_p, p_p = stats.pearsonr(gap, y)
    r_s, p_s = stats.spearmanr(gap, y)
    r_part, p_part = _partial_corr(gap, y, gini)

    from statsmodels.formula.api import ols
    formula = f"{outcome_col} ~ mean_gap_real + gini"
    model = ols(formula, data=merged).fit()

    return {
        "outcome": outcome_label,
        "col": outcome_col,
        "n": len(merged),
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
        "model": model,
    }


def run_cross_outcome_validation() -> None:
    """Validate H₀ gap against poverty, median income, and mobility."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    ts = pd.read_csv(TIMESERIES_PATH)
    mob = pd.read_csv(MOBILITY_PATH)
    outcomes = _load_outcomes()

    # Mean gap and gini per state
    agg = (
        ts.groupby("state")
        .agg(mean_gap_real=("acs_longest_h0_lifespan_real", "mean"),
             gini=("gini", "mean"))
        .reset_index()
    )

    merged = agg.merge(mob, on="state", how="inner").merge(outcomes, on="state", how="inner")
    if merged.empty:
        print("[cross-outcome] No matching states. Aborting.")
        return

    # Analyse each outcome
    results = [
        _analyse_outcome(merged, "absolute_upward_mobility", "Upward Mobility (Chetty)"),
        _analyse_outcome(merged, "poverty_rate", "Poverty Rate (SAIPE 2019)"),
        _analyse_outcome(merged, "median_income", "Median Household Income (ACS 2019)"),
    ]

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    lines = [
        "=" * 70,
        "CROSS-OUTCOME VALIDATION: H₀ Gap vs Multiple Economic Indicators",
        "=" * 70,
        f"States matched: {len(merged)}",
        "",
    ]

    for r in results:
        sig_gap = "***" if r["gap_pval"] < 0.001 else "**" if r["gap_pval"] < 0.01 else "*" if r["gap_pval"] < 0.05 else "n.s."
        lines += [
            f"--- {r['outcome']} (n={r['n']}) ---",
            f"  Pearson  r = {r['pearson_r']:.4f}  (p = {r['pearson_p']:.4e})",
            f"  Spearman ρ = {r['spearman_r']:.4f}  (p = {r['spearman_p']:.4e})",
            f"  Partial r (| Gini) = {r['partial_r']:.4f}  (p = {r['partial_p']:.4e})",
            f"  OLS R² (gap + Gini) = {r['ols_r2']:.4f}",
            f"    gap  coef = {r['gap_coef']:.4e}  (p = {r['gap_pval']:.4e}) {sig_gap}",
            f"    gini coef = {r['gini_coef']:.4f}  (p = {r['gini_pval']:.4e})",
            "",
        ]

    # Summary table
    lines += [
        "--- Summary: Partial r(gap, outcome | Gini) ---",
        f"  {'Outcome':<40} {'Partial r':>10} {'p-value':>12} {'Sig':>5}",
        f"  {'-'*40} {'-'*10} {'-'*12} {'-'*5}",
    ]
    for r in results:
        sig = "***" if r["partial_p"] < 0.001 else "**" if r["partial_p"] < 0.01 else "*" if r["partial_p"] < 0.05 else ""
        lines.append(f"  {r['outcome']:<40} {r['partial_r']:>10.4f} {r['partial_p']:>12.4e} {sig:>5}")

    lines += [
        "",
        "Interpretation:",
        "  If the H₀ gap shows significant partial correlations with multiple",
        "  distinct outcomes (controlling for Gini), the 'complementary signal'",
        "  claim is robust — not an artefact of a single validation dataset.",
        "  Significant results across poverty, income, and mobility indicate",
        "  the topological gap captures a fundamental structural dimension of",
        "  inequality that traditional scalar metrics miss.",
    ]

    report = "\n".join(lines)
    with open(OUT_DIR / "cross_outcome_validation.txt", "w") as fh:
        fh.write(report)
    print(report)

    # -----------------------------------------------------------------------
    # Plot: 1×3 scatter grid
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    configs = [
        ("absolute_upward_mobility", "Upward Mobility", "steelblue", results[0]),
        ("poverty_rate", "Poverty Rate (%)", "firebrick", results[1]),
        ("median_income", "Median HH Income ($)", "seagreen", results[2]),
    ]

    for ax, (col, ylabel, color, r) in zip(axes, configs):
        ax.scatter(merged["mean_gap_real"], merged[col],
                   c=color, edgecolor="k", s=50, alpha=0.8)
        z = np.polyfit(merged["mean_gap_real"], merged[col], 1)
        xl = np.linspace(merged["mean_gap_real"].min(), merged["mean_gap_real"].max(), 100)
        ax.plot(xl, np.polyval(z, xl), "r--", lw=1.5)
        for _, row in merged.iterrows():
            ax.annotate(row["state"][:2], (row["mean_gap_real"], row[col]),
                        fontsize=5, alpha=0.6)
        ax.set(xlabel="Mean H₀ Gap (2024 $)", ylabel=ylabel,
               title=f"r={r['pearson_r']:.3f}, partial r={r['partial_r']:.3f}")

    fig.suptitle(
        "Cross-Outcome Validation: Does the H₀ Gap Predict Multiple Economic Indicators?",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "cross_outcome_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[cross-outcome] Saved plot → {PLOT_DIR / 'cross_outcome_validation.png'}")
    print(f"[cross-outcome] Saved report → {OUT_DIR / 'cross_outcome_validation.txt'}")


if __name__ == "__main__":
    run_cross_outcome_validation()
