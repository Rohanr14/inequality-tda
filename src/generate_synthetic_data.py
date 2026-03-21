# src/generate_synthetic_data.py
# Generates realistic synthetic ACS B19001 bracket data for all 50 states + DC
# across 2010-2023, using log-normal income distributions calibrated to known
# US income statistics. Used when the Census API is unreachable.

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import (
    ACS_BRACKET_UPPER_BOUNDS, ACS_BRACKET_VARS, NUM_PERCENTILES,
    _calculate_percentiles_from_brackets, load_cpi,
)

# Bracket upper bounds (finite only, plus inf)
BRACKET_BOUNDS = [0, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000,
                  50000, 60000, 75000, 100000, 125000, 150000, 200000, np.inf]

# State-level median income parameters (approximate 2022 values, used as anchors)
# Format: {state: (median_income, inequality_factor)}
# inequality_factor > 1 means more unequal
STATE_PARAMS = {
    "Alabama": (56726, 1.10), "Alaska": (80287, 0.90), "Arizona": (65913, 1.05),
    "Arkansas": (52528, 1.12), "California": (84907, 1.15), "Colorado": (82254, 0.95),
    "Connecticut": (83771, 1.08), "Delaware": (72724, 1.00), "District of Columbia": (101722, 1.25),
    "Florida": (63062, 1.10), "Georgia": (65030, 1.12), "Hawaii": (88005, 0.92),
    "Idaho": (63377, 0.93), "Illinois": (72205, 1.08), "Indiana": (61944, 1.02),
    "Iowa": (65573, 0.88), "Kansas": (66178, 0.95), "Kentucky": (55573, 1.10),
    "Louisiana": (52295, 1.15), "Maine": (64767, 0.95), "Maryland": (90203, 1.05),
    "Massachusetts": (89645, 1.10), "Michigan": (63498, 1.05), "Minnesota": (77706, 0.90),
    "Mississippi": (48610, 1.18), "Missouri": (61043, 1.05), "Montana": (60560, 0.93),
    "Nebraska": (66644, 0.90), "Nevada": (63276, 1.05), "New Hampshire": (83449, 0.88),
    "New Jersey": (89296, 1.10), "New Mexico": (53992, 1.12), "New York": (74314, 1.20),
    "North Carolina": (60516, 1.08), "North Dakota": (68131, 0.85), "Ohio": (61938, 1.05),
    "Oklahoma": (55826, 1.08), "Oregon": (70084, 1.00), "Pennsylvania": (67587, 1.05),
    "Rhode Island": (71169, 1.02), "South Carolina": (58234, 1.10), "South Dakota": (62520, 0.88),
    "Tennessee": (59695, 1.10), "Texas": (67321, 1.12), "Utah": (79449, 0.85),
    "Vermont": (63477, 0.90), "Virginia": (80615, 1.05), "Washington": (82228, 1.00),
    "West Virginia": (48037, 1.10), "Wisconsin": (67125, 0.92), "Wyoming": (68002, 0.90),
}


def _lognormal_bracket_counts(median: float, sigma: float, n_households: int,
                               rng: np.random.Generator) -> np.ndarray:
    """Generate household counts per ACS bracket from a log-normal distribution."""
    mu = np.log(median)
    incomes = rng.lognormal(mu, sigma, n_households)
    counts = np.zeros(len(BRACKET_BOUNDS) - 1, dtype=int)
    for i in range(len(counts)):
        lo = BRACKET_BOUNDS[i]
        hi = BRACKET_BOUNDS[i + 1]
        counts[i] = int(np.sum((incomes >= lo) & (incomes < hi)))
    return counts


def generate_all_years():
    """Generate synthetic ACS data for 2010-2023, process, and save as pickles."""
    rng = np.random.default_rng(42)
    processed_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Year-over-year income growth (nominal, ~3% avg with variation)
    year_growth = {
        2010: 1.000, 2011: 1.02, 2012: 1.04, 2013: 1.06, 2014: 1.09,
        2015: 1.13, 2016: 1.16, 2017: 1.19, 2018: 1.23, 2019: 1.27,
        2020: 1.25, 2021: 1.30, 2022: 1.38, 2023: 1.42,
    }

    for year in range(2010, 2024):
        growth = year_growth[year]
        rows = []
        for state, (median_2022, ineq_factor) in STATE_PARAMS.items():
            # Scale median by year growth (relative to 2022 baseline)
            median_yr = median_2022 * growth / year_growth[2022]
            # Log-normal sigma controls inequality spread
            sigma = 0.85 * ineq_factor + rng.normal(0, 0.02)
            sigma = max(0.5, min(sigma, 1.3))

            n_households = rng.integers(400_000, 5_000_000)
            counts = _lognormal_bracket_counts(median_yr, sigma, n_households, rng)

            row = {"NAME": state}
            for var, count in zip(ACS_BRACKET_VARS, counts):
                row[var] = count
            rows.append(row)

        df = pd.DataFrame(rows)

        # Calculate percentile vectors using the (now Pareto-tailed) interpolation
        p_col = f"income_percentiles_{NUM_PERCENTILES}"
        df[p_col] = df.apply(
            lambda r: _calculate_percentiles_from_brackets(r, NUM_PERCENTILES), axis=1
        )
        df = df[["NAME", p_col]].dropna().reset_index(drop=True)

        save_path = os.path.join(processed_dir, f"acs_percentiles_{year}_{NUM_PERCENTILES}pts.pkl")
        df.to_pickle(save_path)
        print(f"[synth] Saved {year}: {len(df)} states → {save_path}")

    print(f"\n[synth] Generated synthetic data for 2010-2023 in {processed_dir}")


if __name__ == "__main__":
    generate_all_years()
