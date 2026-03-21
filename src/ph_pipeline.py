# src/ph_pipeline.py
# Functions for computing persistent homology on income percentile data, including gap localization.

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional
from numpy.random import default_rng
from data_loader import load_cpi, NUM_PERCENTILES

# --- Constants ---
PERCENTILE_COL_PREFIX = 'income_percentiles_'
DEFAULT_FLOAT_VALUE = np.nan # Use NaN for missing numeric values
DEFAULT_INT_VALUE = -1 # Use -1 or similar for missing integer indices/percentiles

# --- Helper Functions ---

def load_processed_acs_data(file_path: str) -> Optional[pd.DataFrame]:
    """Loads processed ACS percentile data from a pickle file."""
    if not os.path.exists(file_path):
        print(f"Error: Processed ACS file not found: {file_path}")
        return None
    try:
        df = pd.read_pickle(file_path)
        # Find the percentile column dynamically
        p_col = next((col for col in df.columns if col.startswith(PERCENTILE_COL_PREFIX)), None)
        if not p_col:
             print(f"Error: Could not find percentile column starting with '{PERCENTILE_COL_PREFIX}' in {file_path}")
             return None
        # Ensure the percentile column actually contains numpy arrays
        mask = df[p_col].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0)
        if not mask.all():
            print(f"Error: Column '{p_col}' in {file_path} does not contain numpy arrays.")
            df = df[mask].copy()  # Keep only rows with valid arrays
            if df.empty:
                print("Error: No valid data remaining after filtering.")
                return None

        print(f"Successfully loaded processed ACS data from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading pickle file {file_path}: {e}")
        return None

def _find_closest_percentile_rank(value: float, sorted_vector: np.ndarray) -> int:
    """Finds the index (percentile rank) in the sorted vector closest to the given value."""
    if sorted_vector is None or sorted_vector.size == 0 or not np.isfinite(value):
        return DEFAULT_INT_VALUE
    # np.argmin finds the index of the minimum value.
    # np.abs(sorted_vector - value) calculates the absolute difference between the value and each element.
    # The index of the minimum difference corresponds to the element closest to 'value'.
    closest_idx = np.argmin(np.abs(sorted_vector - value))
    return int(closest_idx) # Index directly corresponds to percentile rank (0-100 for 101 points)

def locate_largest_gap(sorted_vector: np.ndarray) -> Tuple[int, int, float]:
    """
    Returns (lower_idx, upper_idx, gap_width) where lower_idx is the percentile
    *before* the gap and upper_idx is the one *after* it.
    """
    diffs = np.diff(sorted_vector)
    if diffs.size == 0:
        return -1, -1, np.nan

    j = int(np.argmax(diffs))          # index of the biggest jump
    return j, j + 1, float(diffs[j])   # indices and the raw $ gap

def bootstrap_gap(vec: np.ndarray, n_boot: int = 1_000, seed: int = 0) -> tuple[float, float]:
    """
    Bootstrap 95% CI for the largest $-jump in a 1‑D income vector.
    Returns:
         (lo, hi) in current dollars.
    """
    rng = default_rng(seed)
    gaps = []
    for _ in range(n_boot):
        samp = rng.choice(vec, size=len(vec), replace=True)
        gaps.append(np.max(np.diff(np.sort(samp))))
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return float(lo), float(hi)


# --- Core TDA Functions ---

def get_longest_finite_h0_feature_details(diagram: np.ndarray) -> Tuple[float, float, float]:
    """
    Extracts the details (lifespan, birth, death) of the *longest-living finite* H0 feature.

    Args:
        diagram: Persistence diagram array [birth, death, dimension].

    Returns:
        A tuple (max_lifespan, birth_value, death_value).
        Returns (NaN, NaN, NaN) if no finite H0 features exist or diagram is invalid.
    """
    default_return = (DEFAULT_FLOAT_VALUE, DEFAULT_FLOAT_VALUE, DEFAULT_FLOAT_VALUE)
    if diagram is None or diagram.ndim != 2 or diagram.shape[0] == 0 or diagram.shape[1] != 3:
        return default_return

    try:
        # Filter for H0 features (dimension == 0)
        h0_features = diagram[diagram[:, 2] == 0]

        if h0_features.shape[0] == 0:
            return default_return # Should not happen if input had points

        # Calculate lifespans (death - birth) for finite H0 features
        # Ignore infinite death values if any (shouldn't happen often in 1D with full filtration)
        finite_h0 = h0_features[np.isfinite(h0_features[:, 1])]
        if finite_h0.shape[0] == 0:
             # This means only one component existed (e.g., input had only 1 unique point after processing?)
             return default_return

        lifespans = finite_h0[:, 1] - finite_h0[:, 0]

        # Ignore tiny lifespans potentially caused by floating point issues
        valid_lifespans = lifespans > 1e-9
        if not np.any(valid_lifespans):
            # print("Warning: All finite H0 lifespans are near zero.")
            return default_return  # Treat as no significant gap

        # Find the max among valid lifespans
        max_lifespan_idx = np.argmax(lifespans[valid_lifespans])
        original_idx = np.where(valid_lifespans)[0][max_lifespan_idx] # Get the original index in the finite_h0 array

        max_lifespan = float(lifespans[valid_lifespans][max_lifespan_idx])
        birth_value = float(finite_h0[original_idx, 0])
        death_value = float(finite_h0[original_idx, 1])

        return max_lifespan, birth_value, death_value

    except Exception as e:
        print(f"Error calculating H0 feature details: {e}")
        return default_return



# --- Analysis Orchestration ---

def run_ph_analysis_for_year(
    year: int,
    processed_acs_path_template: str,
    num_percentiles: int = NUM_PERCENTILES
) -> Optional[pd.DataFrame]:
    """
    Runs the persistent homology analysis for all states in a given year,
    extracting details of the longest finite H0 gap.

    Args:
        year: The year to analyze.
        processed_acs_path_template: Path template for processed ACS pkl files
        num_percentiles: Number of percentiles used in ACS data (must match file).

    Returns:
        A DataFrame containing state names and details of their longest finite H0 gaps
        (lifespan, birth/death income, birth/death percentile) for the year.
        Returns None if data loading fails for the year.
    """
    print(f"\n--- Running PH Analysis for {year} ---")

    # --- Load ACS Data ---
    acs_file_path = processed_acs_path_template.format(year=year, num_pts=num_percentiles)
    acs_df = load_processed_acs_data(acs_file_path)
    if acs_df is None or acs_df.empty:
        print(f"Skipping year {year} due to ACS data loading failure or empty data.")
        return None

    percentile_col = next((col for col in acs_df.columns if col.startswith(PERCENTILE_COL_PREFIX)), None)
    if not percentile_col: # Should have been caught in load function, but double-check
         print(f"Critical Error: Percentile column not found after loading {acs_file_path}")
         return None

    # --- Perform Analysis ---
    results = []
    for index, row in acs_df.iterrows():
        state_name = row['NAME']
        acs_vector = row[percentile_col]

        # Initialize result row with defaults
        result_row = {
            'year': year,
            'state': state_name,
            'acs_longest_h0_lifespan': DEFAULT_FLOAT_VALUE,
            'acs_gap_lo': DEFAULT_FLOAT_VALUE,
            'acs_gap_hi': DEFAULT_FLOAT_VALUE,
            'acs_birth_income': DEFAULT_FLOAT_VALUE,
            'acs_death_income': DEFAULT_FLOAT_VALUE,
            'acs_birth_percentile': DEFAULT_INT_VALUE,
            'acs_death_percentile': DEFAULT_INT_VALUE,
            'gini': DEFAULT_FLOAT_VALUE,
            'theil': DEFAULT_FLOAT_VALUE
        }

        # --- Analyze ACS Data ---
        if acs_vector is not None and acs_vector.size >= 2:
            # 1-D vector is already sorted from the loader
            lo_idx, hi_idx, gap_width = locate_largest_gap(acs_vector)
            gap_lo, gap_hi = bootstrap_gap(acs_vector)

            result_row.update({
                'acs_longest_h0_lifespan': gap_width,
                'acs_gap_lo': gap_lo,
                'acs_gap_hi': gap_hi,
                'acs_birth_income': acs_vector[lo_idx] if lo_idx >= 0 else np.nan,
                'acs_death_income': acs_vector[hi_idx] if hi_idx >= 0 else np.nan,
                'acs_birth_percentile': 100 * lo_idx / len(acs_vector),
                'acs_death_percentile': 100 * hi_idx / len(acs_vector),
            })

            # Include Gini coefficient and Theil index for ACS data comparison
            vec = acs_vector.astype(float)
            mean = vec.mean()
            gini = 1 - 2 * np.trapezoid(np.cumsum(vec) / vec.sum(), dx=1 / len(vec))
            mask = vec > 0  # ignore zero-income entries
            vals = vec[mask] / mean
            theil = (vals * np.log(vals)).mean()

            result_row.update({"gini": gini, "theil": theil})
        else:
            print(f"Skipping ACS analysis for {state_name}, {year} due to invalid vector.")

        results.append(result_row)
        # Optional: Print progress
        # if (index + 1) % 10 == 0:
        #     print(f"Processed {index + 1}/{len(acs_df)} states for {year}...")

    results_df = pd.DataFrame(results)
    print(f"Finished PH analysis for {year}. Results shape: {results_df.shape}")
    return results_df


# --- Example Usage ---
if __name__ == "__main__":
    print("Running ph_pipeline.py example...")

    # Define years and paths (adjust relative paths based on project structure)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'timeseries')

    PROCESSED_ACS_TEMPLATE = os.path.join(PROCESSED_DATA_DIR, 'acs_percentiles_{year}_{num_pts}pts.pkl')

    # Years with available processed data
    potential_acs_years = list(range(2010, 2024))
    available_acs_years = [
        yr for yr in potential_acs_years
        if os.path.exists(PROCESSED_ACS_TEMPLATE.format(year=yr, num_pts=NUM_PERCENTILES))
    ]

    if not available_acs_years:
        print("Error: No processed ACS files found matching the template. Cannot run analysis.")
        print(f"Template checked: {PROCESSED_ACS_TEMPLATE.format(year='YYYY', num_pts=NUM_PERCENTILES)}")
        exit()  # Or handle differently

    # Analyze years where ACS data is available
    YEARS_TO_ANALYZE = sorted(available_acs_years)
    print(f"Found processed ACS data for years: {YEARS_TO_ANALYZE}")

    # --- Run for all years and save ---
    print("\n--- Running Analysis for All Available Common Years ---")
    all_results = []

    for year in YEARS_TO_ANALYZE:
        year_results = run_ph_analysis_for_year(
            year=year,
            processed_acs_path_template=PROCESSED_ACS_TEMPLATE,
            num_percentiles=NUM_PERCENTILES,
        )
        if year_results is not None and not year_results.empty:
            all_results.append(year_results)
        else:
            print(f"Warning: No results generated for year {year}.")

    if all_results:
        final_results_df = pd.concat(all_results, ignore_index=True)
        final_results_df = final_results_df[final_results_df["state"] != "Puerto Rico"]

        # Deflate with CPI-U
        cpi_data = load_cpi()
        final_results_df = final_results_df.merge(cpi_data, on='year', how='left')
        for col in ["acs_longest_h0_lifespan", "acs_gap_lo", "acs_gap_hi", "acs_birth_income", "acs_death_income"]:
            final_results_df[f"{col}_real"] = final_results_df[col] * final_results_df["deflator"]


        print("\n--- Combined Results ---")
        print(f"Total results shape: {final_results_df.shape}")
        print(final_results_df.head())
        print(final_results_df.tail())

        # --- Display Summary Statistics ---
        print("\n--- Summary Statistics ---\n\n")
        print("\n--- ACS Longest Gap Lifespan ---")
        print(final_results_df['acs_longest_h0_lifespan'].describe())

        print("\n--- ACS Birth Percentile ---")
        # Filter out default value before describing percentile ranks
        valid_acs_birth_p = final_results_df['acs_birth_percentile'][final_results_df['acs_birth_percentile'] != DEFAULT_INT_VALUE]
        print(valid_acs_birth_p.describe())

        print("\n--- ACS Death Percentile ---")
        valid_acs_death_p = final_results_df['acs_death_percentile'][final_results_df['acs_death_percentile'] != DEFAULT_INT_VALUE]
        print(valid_acs_death_p.describe())

        # Save combined results to the results/timeseries directory
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_path = os.path.join(RESULTS_DIR, f'h0_gap_details_{NUM_PERCENTILES}pts_timeseries.csv')
        try:
            final_results_df.to_csv(save_path, index=False)
            print(f"\nSaved combined results to {save_path}")
        except Exception as e:
            print(f"\nError saving combined results: {e}")
    else:
        print("\nNo results generated for any year.")

    print("\nPH Pipeline example finished.")