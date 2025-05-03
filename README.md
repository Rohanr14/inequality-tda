# Topological Data Analysis of Income Distribution Gaps

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ## Overview

This project applies Topological Data Analysis (TDA), specifically persistent homology (H₀), to analyze the structure of income inequality across US states from 2010 to 2023. Instead of relying solely on scalar summary statistics like the Gini coefficient, this project identifies and quantifies the **largest gap (H₀ feature lifespan)** within the interpolated income percentile distribution for each state and year.

The core idea is that the location and magnitude of the largest jump between adjacent income percentiles can reveal structural features of inequality, such as economic stratification or barriers to income mobility, potentially missed by traditional measures.

The project includes:
* Data fetching and processing from US Census ACS.
* A pipeline to compute the H₀ gap, its location, confidence intervals, and traditional Gini/Theil indices.
* Analysis scripts for robustness checks (fixed effects, bin sensitivity).
* Visualization tools for generating static plots.
* An interactive Streamlit dashboard for exploring the results.
* Makefile for easy execution of common tasks.
* Dockerfile for containerizing the dashboard application.

## Features

* Fetches ACS B19001 (Household Income) data via Census API.
* Processes bracketed income data into 101-point percentile vectors.
* Calculates the largest H₀ gap (lifespan, birth/death income, birth/death percentile) using persistent homology concepts.
* Computes bootstrap 95% confidence intervals for the H₀ gap size.
* Calculates Gini and Theil coefficients for comparison.
* Adjusts income figures for inflation using CPI-U data (real 2024 dollars).
* Performs state/year fixed-effects regression analysis.
* Checks sensitivity to the number of percentile bins.
* Generates static plots (leaderboards, time-series, delta-vectors).
* Provides an interactive Streamlit dashboard for map-based and time-series exploration.

## Directory Structure
````
├── data/
│   ├── raw/
│   │   └── cpi-u_annual.csv         # CPI data for inflation adjustment
│   └── processed/                   # Processed & pickled ACS data (generated)
├── results/
│   ├── timeseries/                  # Main output csv data (generated)
│   └── plots/                       # Generated static plots (generated)
├── src/
│   ├── analysis/
│   │   ├── regression/              # Regression outputs (generated)
│   │   ├── bin_sensitivity.py
│   │   └── fixed_effects.py
│   ├── dashboard/
│   │   └── app.py                   # Streamlit dashboard code
│   ├── data_loader.py               # Fetches and processes ACS data
│   ├── ph_pipeline.py               # Computes TDA gap, Gini, & Theil
│   └── viz.py                       # Generates static visualizations
├── Dockerfile                       # For containerizing the dashboard
├── Makefile                         # For easy task execution
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup script
└── README.md
````

*(Note: `data/processed/`, `results/` directories are created by the scripts if they don't exist)*

## Data

* **ACS Income Data:** US Census Bureau, American Community Survey (ACS) 1-Year Estimates (Table B19001: Household Income). Fetched via Census API using `src/data_loader.py`. Covers years 2010-2023 (excluding potential data gaps). Processed into 101-point percentile vectors (0th-100th percentile).
* **CPI Data:** `data/raw/cpi-u_annual.csv` contains annual average CPI-U values used to deflate nominal income figures to real 2024 dollars.
* **Output Time Series:** `results/timeseries/h0_gap_details_101pts_timeseries.csv` is the main output file, containing state-year level data on the H₀ gap, its location, confidence intervals, Gini/Theil indices, and real dollar values.
* **GeoJSON:** `us-states.json` (provided, source: US Atlas TopoJSON) is used for mapping in the dashboard.

## Methodology

1.  **Percentile Interpolation:** ACS income bracket data (B19001) is linearly interpolated to estimate income levels at 101 percentile points (0 to 100).
2.  **H₀ Gap Calculation:** For each state-year's 1D income percentile vector, the largest absolute dollar difference between adjacent percentiles is identified. This corresponds to the longest-lived finite H₀ feature in the Vietoris-Rips filtration of the 1D point cloud.
3.  **Inflation Adjustment:** Nominal dollar values (gap size, birth/death income) are adjusted to real 2024 dollars using the provided CPI-U data.
4.  **Comparison Metrics:** Gini and Theil indices are calculated from the percentile vectors for comparison.
5.  **Robustness:** Sensitivity to binning resolution (101 vs 201 points) and fixed-effects regression analysis are performed.

## Setup & Installation

1.  **Prerequisites:**
    * Python >= 3.10
    * `git` (for cloning)
    * `make` (optional, for using Makefile commands)
    * `docker` (optional, for using Docker)

2.  **Clone Repository:**
    ```bash
    git clone https://github.com/Rohanr14/inequality-tda.git
    cd inequality-tda
    ```

3.  **Set up Environment (Choose ONE):**

    * **Using Makefile (Recommended):** Creates a virtual environment `.venv` and installs dependencies.
        ```bash
        make env
        ```
        Activate the environment for subsequent commands:
        ```bash
        source .venv/bin/activate  # On Windows use `venv\Scripts\activate`
        ```

    * **Manual Setup:**
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows use `venv\Scripts\activate`
        pip install -r requirements.txt
        ```

## Usage

Ensure you are in the project's root directory and the virtual environment (if used) is activated.

**Using Makefile (Recommended):**

* **Fetch & Process Data:**
    ```bash
    make process
    ```
* **Run TDA Pipeline:** (Requires processed data)
    ```bash
    make analyse
    ```
* **Generate Static Plots:** (Requires pipeline results)
    ```bash
    make plots
    ```
* **Run Fixed-Effects Regression:** (Requires pipeline results)
    ```bash
    make regression
    ```
* **Run Bin Sensitivity Analysis:** (Requires pipeline results for *both* 101 and 201 points)
    ```bash
    make sens
    ```
* **Launch Dashboard:** (Requires pipeline results)
    ```bash
    make dashboard
    ```
* **Run Full Demo (Setup to Dashboard):**
    ```bash
    make demo
    ```

**Manual Execution (Alternative):**

1.  **Fetch & Process Data:**
    ```bash
    python src/data_loader.py
    ```
2.  **Run TDA Pipeline:**
    ```bash
    python src/ph_pipeline.py
    ```
3.  **Run Analyses:**
    ```bash
    # Example:
    python src/analysis/fixed_effects.py
    python src/analysis/bin_sensitivity.py # (Needs 201pt data generated)
    ```
4.  **Generate Plots:**
    ```bash
    python src/viz.py
    ```
5.  **Launch Dashboard:**
    ```bash
    streamlit run src/dashboard/app.py
    ```

## Docker Usage (Dashboard Only)

You can run the Streamlit dashboard within a Docker container.

1.  **Build the Docker Image:**
    ```bash
    docker build -t inequality-tda-dashboard .
    ```

2.  **Run the Docker Container:**
    ```bash
    # Make sure the results CSV exists first (e.g., run `make analyse` locally)
    # Map the results directory into the container and expose the port
    docker run -p 8501:8501 -v "$(pwd)/results:/app/results" inequality-tda-dashboard
    ```
    * Access the dashboard at `http://localhost:8501` in your browser.
    * The `-v` flag mounts your local `results` directory into the container so the dashboard can find the data. Adjust the path before `:` if your results are elsewhere.

## Results

* The primary quantitative results are generated in `results/timeseries/`.
* Regression and sensitivity analysis outputs are generated in `src/analysis/regression/`.
* Static plots are generated in `results/plots/`.
* The interactive dashboard provides map and time-series visualizations.

## Author

* Rohan Ramavajjala (<rovajjala@gmail.com>)

## License

This project is licensed under the MIT License - see the LICENSE file (if created) for details.