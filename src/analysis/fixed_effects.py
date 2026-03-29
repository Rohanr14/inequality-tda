# src/analysis/fixed_effects.py
# State/year fixed-effects regression on the real-$ H₀ gap.
#
# Uses the within (demeaning) estimator: state effects are absorbed via
# demeaning rather than estimated as explicit dummies, which avoids the
# degenerate cluster-robust SEs that arise when clustering on the same
# variable used for fixed effects.

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning

warnings.filterwarnings("ignore", category=ValueWarning)

RESOLUTION = 101
BASE = Path(__file__).resolve().parent.parent.parent
TIMESERIES_PATH = BASE / "results" / "timeseries" / f"h0_gap_details_{RESOLUTION}pts_timeseries.csv"
OUT_DIR = Path(__file__).resolve().parent / "regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_fixed_effects() -> None:
    # 1. Load data
    df = pd.read_csv(TIMESERIES_PATH)
    df = df.rename(columns={"acs_longest_h0_lifespan_real": "gap_real"})

    # 2. Within transformation: demean by state to absorb state FEs
    state_means = df.groupby("state")["gap_real"].transform("mean")
    df["gap_demean"] = df["gap_real"] - state_means

    # 3. Regress demeaned gap on year dummies (state effects absorbed)
    #    Use HC1 robust SEs (valid for heteroskedasticity without the
    #    cluster-on-FE degeneracy problem).
    mod = smf.ols(
        formula="gap_demean ~ C(year) - 1",
        data=df,
    ).fit(cov_type="HC1")

    # 4. Also run the full dummy model with nonrobust SEs for R² and state FEs
    mod_full = smf.ols(
        formula="gap_real ~ C(year) + C(state)",
        data=df,
    ).fit()

    # 5. Extract top/bottom state effects for the report
    state_params = {
        k.replace("C(state)[T.", "").rstrip("]"): v
        for k, v in mod_full.params.items()
        if k.startswith("C(state)")
    }
    state_se = {
        k.replace("C(state)[T.", "").rstrip("]"): v
        for k, v in mod_full.bse.items()
        if k.startswith("C(state)")
    }
    sorted_states = sorted(state_params.items(), key=lambda x: x[1], reverse=True)

    # 6. Build report
    lines = [
        "=" * 70,
        "STATE/YEAR FIXED-EFFECTS REGRESSION: Real H₀ Gap (2024 $)",
        "=" * 70,
        f"Observations: {len(df)}  |  States: {df['state'].nunique()}  |  Years: {df['year'].nunique()}",
        f"Overall R² (full model): {mod_full.rsquared:.4f}",
        f"Adj. R²: {mod_full.rsquared_adj:.4f}",
        "",
        "--- Year Fixed Effects (within estimator, HC1 robust SEs) ---",
        f"  {'Year':<8} {'Coef':>12} {'Std Err':>12} {'t':>8} {'p-value':>12}",
        f"  {'-'*8} {'-'*12} {'-'*12} {'-'*8} {'-'*12}",
    ]

    # Reference year (intercept absorbed by demeaning, so year coefficients
    # represent deviation from state mean)
    for k, v in sorted(mod.params.items(), key=lambda x: x[0]):
        year_label = k.replace("C(year)[", "").rstrip("]")
        se = mod.bse[k]
        t = mod.tvalues[k]
        p = mod.pvalues[k]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        lines.append(f"  {year_label:<8} {v:>12,.0f} {se:>12,.0f} {t:>8.2f} {p:>12.4e} {sig}")

    lines += [
        "",
        "--- State Fixed Effects (top 10 / bottom 10, nonrobust SEs) ---",
        f"  Base state: Alabama (intercept = {mod_full.params['Intercept']:,.0f})",
        "",
        f"  {'State':<25} {'Coef':>12} {'Std Err':>12}",
        f"  {'-'*25} {'-'*12} {'-'*12}",
    ]
    for name, coef in sorted_states[:10]:
        se = state_se[name]
        lines.append(f"  {name:<25} {coef:>12,.0f} {se:>12,.0f}")
    lines.append(f"  {'...':^49}")
    for name, coef in sorted_states[-10:]:
        se = state_se[name]
        lines.append(f"  {name:<25} {coef:>12,.0f} {se:>12,.0f}")

    lines += [
        "",
        "Note: Year FEs use within-estimator (state-demeaned) with HC1 robust",
        "SEs. State FEs use nonrobust SEs from the full dummy model. Previous",
        "versions used cluster-robust SEs on the state variable, which produced",
        "degenerate (near-zero) SEs for state dummies because each state dummy",
        "has zero within-cluster variation.",
    ]

    report = "\n".join(lines)
    with open(OUT_DIR / "state_fe_gap_real.txt", "w") as fh:
        fh.write(report)
    print(report)
    print(f"\n[fe] Saved → {OUT_DIR / 'state_fe_gap_real.txt'}")


if __name__ == "__main__":
    run_fixed_effects()