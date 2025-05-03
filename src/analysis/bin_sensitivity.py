# src/analysis/bin_sensitivity.py
# Script for analyzing the sensitivity of the income gap to binning resolution, measuring Pearson correlation.

import pandas as pd
from pathlib import Path

BASE = Path("../../results/timeseries")

# Load both CSVs – you already output 101‑pt, now generate 201‑pt before running this
df101 = pd.read_csv(BASE / "h0_gap_details_101pts_timeseries.csv")
df201 = pd.read_csv(BASE / "h0_gap_details_201pts_timeseries.csv")

# Keep only the metric in real dollars
df101 = df101.rename(columns={"acs_longest_h0_lifespan_real": "gap_real_101"})
df201 = df201.rename(columns={"acs_longest_h0_lifespan_real": "gap_real_201"})

merged = (
    df101[["state", "year", "gap_real_101"]]
    .merge(df201[["state", "year", "gap_real_201"]], on=["state", "year"])
)

rho = merged["gap_real_101"].corr(merged["gap_real_201"])
print(f"Pearson ρ(101 vs 201 bins) = {rho:.4f}")

out = Path("regression")
out.mkdir(parents=True, exist_ok=True)
with open(out / "bin_sensitivity.txt", "w") as fh:
    fh.write(f"Pearson correlation between 101‑pt and 201‑pt real‑$ gaps: {rho:.4f}\n")
