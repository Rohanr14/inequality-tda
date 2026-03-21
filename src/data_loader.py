# src/data_loader.py
# Functions for loading and processing ACS income data.

import requests
import pandas as pd
import numpy as np
import os
from typing import List, Optional, Any

# --- Constants ---

# ACS B19001 Income Brackets (Upper bounds)
# Based on standard ACS definitions. The last bracket ($200k+) is open-ended.
# We need these bounds for interpolation.
ACS_BRACKET_UPPER_BOUNDS = {
    'B19001_002E': 10000,
    'B19001_003E': 15000,
    'B19001_004E': 20000,
    'B19001_005E': 25000,
    'B19001_006E': 30000,
    'B19001_007E': 35000,
    'B19001_008E': 40000,
    'B19001_009E': 45000,
    'B19001_010E': 50000,
    'B19001_011E': 60000,
    'B19001_012E': 75000,
    'B19001_013E': 100000,
    'B19001_014E': 125000,
    'B19001_015E': 150000,
    'B19001_016E': 200000,
    'B19001_017E': np.inf # Open-ended upper bracket
}
# Ordered list of bracket variable names (Estimate columns)
ACS_BRACKET_VARS = list(ACS_BRACKET_UPPER_BOUNDS.keys())
NUM_PERCENTILES = 101

# --- ACS Data Loading ---

def fetch_acs_data(year: int) -> Optional[pd.DataFrame]:
    """
    Fetches state-level ACS B19001 data for a given year using the Census API.

    Args:
        year: The year for which to fetch data (e.g., 2019).

    Returns:
        A pandas DataFrame containing the raw ACS data for the specified year,
        or None if the API request fails.
    """
    print(f"Fetching ACS B19001 data for {year}...")
    # Construct the API URL
    # GET group(B19001) gets all variables in that table group
    # FOR state:* gets data for all states
    # Note: Older years might use different dataset names (e.g., acs5)
    # Adjust dataset name if needed (e.g., acs1 for 1-year estimates)
    dataset = "acs1" if year != 2020 else "acs5" # Assuming 1-year estimates
    if year < 2010: # Basic check, adjust if using different ACS products
        print(f"Warning: Year {year} might require a different ACS dataset name or may not be available.")

    url = f"https://api.census.gov/data/{year}/acs/{dataset}?get=group(B19001)&ucgid=pseudo(0100000US$0400000)"


    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        data = response.json()
        if not data or len(data) < 2:
            print(f"Error: No data returned from API for year {year}.")
            return None

        # First row is header, subsequent rows are data
        header = data[0]
        data_rows = data[1:]

        df = pd.DataFrame(data_rows, columns=header)
        print(f"Successfully fetched ACS data for {year}. Shape: {df.shape}")
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching ACS data for year {year}: {e}")
        return None
    except Exception as e:
        print(f"Error processing ACS data for year {year}: {e}")
        return None

def load_cpi(path: str = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "raw", "cpi-u_annual.csv")
    cpi = pd.read_csv(path)
    base = cpi.loc[cpi.year == 2024, "cpi-u"].iat[0]
    cpi["deflator"] = base / cpi["cpi-u"]      # >1 → inflates older $ to 2024 $
    return cpi[["year", "deflator"]]

def _estimate_pareto_alpha(frac_above_lower: float, frac_above_upper: float,
                           lower_bound: float, upper_bound: float) -> float:
    """
    Estimate Pareto shape parameter α from two survival fractions.

    Given that fraction f₁ of households earn above `lower_bound` and fraction
    f₂ earn above `upper_bound`, the Pareto α satisfies:
        f₂/f₁ = (lower_bound / upper_bound)^α
    so  α = log(f₁/f₂) / log(upper_bound/lower_bound).

    Falls back to α = 1.5 (a reasonable US income default) if the estimate
    is out of the plausible range [1.2, 5.0].
    """
    DEFAULT_ALPHA = 1.5
    if frac_above_lower <= 0 or frac_above_upper <= 0:
        return DEFAULT_ALPHA
    if frac_above_upper >= frac_above_lower:
        return DEFAULT_ALPHA
    ratio = np.log(frac_above_lower / frac_above_upper)
    denom = np.log(upper_bound / lower_bound)
    if denom <= 0:
        return DEFAULT_ALPHA
    alpha = ratio / denom
    if not (1.2 <= alpha <= 5.0):
        return DEFAULT_ALPHA
    return alpha


def _calculate_percentiles_from_brackets(row: pd.Series, num_percentiles: int) -> Optional[np.ndarray]:
    """
    Calculate income percentiles from ACS bracket counts, using a Pareto tail
    fit for the open-ended top bracket ($200k+) instead of capping.

    The Pareto Type I distribution is the standard approach in economics for
    modelling open-ended upper income brackets (Piketty & Saez, 2003).  Given
    the fraction of households above the two highest finite thresholds ($150k
    and $200k), we estimate the Pareto shape parameter α and use the quantile
    function Q(p) = x_m / (1 − p)^(1/α) for percentiles in the tail.

    Args:
        row: A pandas Series representing a single state's ACS data for one year.
        num_percentiles: The number of percentile points to calculate (e.g., 101 for 0-100).

    Returns:
        A numpy array of income values corresponding to the requested percentiles,
        or None if calculation fails.
    """
    try:
        # Extract bracket counts and convert to numeric, coercing errors to NaN
        bracket_counts = pd.to_numeric(row[ACS_BRACKET_VARS], errors='coerce')

        # Check for NaNs or zero total - cannot compute if counts are invalid
        if bracket_counts.isnull().any() or bracket_counts.sum() <= 0:
            return None

        # Define bracket bounds (use 0 as the lower bound for the first bracket)
        bounds = [0] + [ACS_BRACKET_UPPER_BOUNDS[var] for var in ACS_BRACKET_VARS]

        # Calculate cumulative counts and percentages
        cumulative_counts = bracket_counts.cumsum()
        total_households = float(cumulative_counts.iloc[-1])

        # Cumulative percentages *at the upper bound* of each bracket
        cumulative_percentages = (cumulative_counts / total_households) * 100

        # ── Pareto tail estimation ──────────────────────────────────────────
        # Fraction of households above $150k and $200k thresholds
        # B19001_015E → ≤$150k (cumulative through that bracket)
        # B19001_016E → ≤$200k
        # B19001_017E → >$200k (open-ended)
        cum_below_150k = float(cumulative_counts.iloc[-3])   # through $150k bracket
        cum_below_200k = float(cumulative_counts.iloc[-2])   # through $200k bracket
        frac_above_150k = (total_households - cum_below_150k) / total_households
        frac_above_200k = (total_households - cum_below_200k) / total_households

        x_m = 200_000.0  # Pareto threshold (top bracket lower bound)
        alpha = _estimate_pareto_alpha(frac_above_150k, frac_above_200k,
                                       150_000.0, 200_000.0)

        # Percentile where the Pareto tail begins (CDF at $200k)
        pct_at_xm = float(cumulative_percentages.iloc[-2])  # % ≤ $200k

        # ── Build interpolation points for the non-tail portion ─────────────
        # Use all finite brackets (up to and including $200k)
        finite_bounds = [b for b in bounds if np.isfinite(b)]
        n_finite = len(finite_bounds)
        interp_percentiles = np.array([0.0] + list(cumulative_percentages.iloc[:n_finite - 1]))
        interp_incomes = np.array(finite_bounds[:n_finite])

        # Ensure percentile values are strictly increasing for interpolation
        unique_indices = np.unique(interp_percentiles, return_index=True)[1]
        interp_percentiles = interp_percentiles[unique_indices]
        interp_incomes = interp_incomes[unique_indices]

        if len(interp_percentiles) < 2:
            return None

        # ── Interpolate across all target percentiles ───────────────────────
        target_percentiles = np.linspace(0, 100, num_percentiles)
        income_percentiles = np.empty(num_percentiles)

        for i, p in enumerate(target_percentiles):
            if p <= pct_at_xm:
                # Below the Pareto threshold → linear interpolation
                income_percentiles[i] = np.interp(p, interp_percentiles, interp_incomes)
            else:
                # Pareto tail: Q(p) = x_m / (1 − p_tail)^(1/α)
                # p_tail is the fraction *within the tail* (0 at x_m, 1 at ∞)
                tail_fraction = (p - pct_at_xm) / (100.0 - pct_at_xm)
                # Clip to avoid division by zero at the very top
                tail_fraction = min(tail_fraction, 0.999)
                income_percentiles[i] = x_m / ((1.0 - tail_fraction) ** (1.0 / alpha))

        return income_percentiles

    except Exception as e:
        print(f"Error calculating percentiles for state {row.get('NAME', 'Unknown')}: {e}")
        return None


def process_acs_data(df: pd.DataFrame, num_percentiles: int = NUM_PERCENTILES) -> pd.DataFrame:
    """
    Processes raw ACS DataFrame to calculate income percentile vectors for each state.

    Args:
        df: Raw ACS DataFrame as fetched from the API.
        num_percentiles: The number of percentile points (e.g., 101 for 0-100).

    Returns:
        A DataFrame with state names and their calculated income percentile vectors.
    """
    print(f"Processing ACS data to calculate {num_percentiles} percentiles...")
    if df is None or df.empty:
        print("Error: Input DataFrame is empty or None.")
        return pd.DataFrame()

    # Select relevant columns (State Name and Bracket Estimates)
    required_cols = ['NAME'] + ACS_BRACKET_VARS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in ACS data: {missing_cols}")
        return pd.DataFrame()

    # Apply percentile calculation row-wise (each row is a state)
    percentile_col_name = f'income_percentiles_{num_percentiles}'
    df[percentile_col_name] = df.apply(
        lambda row: _calculate_percentiles_from_brackets(row, num_percentiles),
        axis=1
    )

    # Filter out rows where calculation failed and select final columns
    result_df = df[['NAME', percentile_col_name]].dropna().reset_index(drop=True)
    print(f"Finished processing ACS data. Found percentiles for {len(result_df)} states.")
    return result_df


# --- Main Loading Function ---

def load_all_data(
    acs_years: List[int],
    num_percentiles: int = NUM_PERCENTILES,
    save_processed: bool = True,
    processed_dir: str = 'data/processed'
) -> dict[Any, Any]:
    """
    Loads and processes all ACS data for the specified years.

    Args:
        acs_years: List of years for which to fetch ACS data.
        num_percentiles: Number of percentile points for ACS data.
        save_processed: Whether to save the processed dataframes to CSV/pickle.
        processed_dir: Directory to save processed data.

    Returns:
        A tuple containing:
        - dict_acs: Dictionary mapping year to processed ACS percentile DataFrame.
    """
    all_acs_data = {}
    for year in acs_years:
        raw_acs_df = fetch_acs_data(year)
        if raw_acs_df is not None:
            processed_acs_df = process_acs_data(raw_acs_df, num_percentiles)
            if not processed_acs_df.empty:
                all_acs_data[year] = processed_acs_df
                if save_processed:
                    os.makedirs(processed_dir, exist_ok=True)
                    # Saving as pickle preserves the numpy array structure better
                    save_path = os.path.join(processed_dir, f'acs_percentiles_{year}_{num_percentiles}pts.pkl')
                    try:
                        processed_acs_df.to_pickle(save_path)
                        print(f"Saved processed ACS data for {year} to {save_path}")
                    except Exception as e:
                        print(f"Error saving processed ACS data for {year}: {e}")
            else:
                print(f"Processing failed for ACS data year {year}.")
        else:
            print(f"Fetching failed for ACS data year {year}.")

    return all_acs_data

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    print("Running data_loader.py example...")

    # Define years and directories (adjust paths as needed)
    ACS_YEARS_TO_LOAD = list(range(2010, 2024)) # 2010-2023
    PROCESSED_DIR = '../data/processed'

    # Create processed dir if it doesn't exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)


    # Run the main loading function
    acs_data = load_all_data(
        acs_years=ACS_YEARS_TO_LOAD,
        num_percentiles=NUM_PERCENTILES, # Calculate 101 points (0-100)
        save_processed=True,
        processed_dir=PROCESSED_DIR
    )

    # Display summary of loaded data
    print("\n--- Loading Summary ---")
    print(f"Loaded ACS data for years: {list(acs_data.keys())}")
    if 2022 in acs_data:
        print("Sample ACS Data (2022):")
        print(acs_data[2022].head())
        # Print the first percentile vector to check format
        if not acs_data[2022].empty:
             print("\nSample Percentile Vector (first state, 2022):")
             print(acs_data[2022][f'income_percentiles_{NUM_PERCENTILES}'].iloc[0][:10]) # First 10 points
             print("...")
             print(acs_data[2022][f'income_percentiles_{NUM_PERCENTILES}'].iloc[0][-10:]) # Last 10 points

    print("\nExample finished.")
