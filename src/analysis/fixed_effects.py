# src/analysis/fixed_effects.py
# Script for running fixed-effects regressions on income percentile data to test robustness of results.

import pandas as pd
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

# 1. Load data
df = pd.read_csv(TIMESERIES_PATH)

# 2. Use the real‑$ column (2024 dollars)
df = df.rename(columns={"acs_longest_h0_lifespan_real": "gap_real"})

# 3. Run state fixed‑effects OLS with year dummies
mod = smf.ols(
    formula="gap_real ~ C(year) + C(state)",
    data=df
).fit(cov_type="cluster", cov_kwds={"groups": df["state"]})

# 4. Save regression summary to TXT
with open(OUT_DIR / "state_fe_gap_real.txt", "w") as fh:
    fh.write(mod.summary().as_text())

print("Finished. Results written to regression/state_fe_gap_real.txt")