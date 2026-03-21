# src/analysis/wasserstein_analysis.py
# Computes Wasserstein-1 (Earth Mover's) distance between state income
# distributions across years, providing a richer measure of distributional
# change than any single scalar statistic.
#
# Key analyses:
# 1. Year-over-year Wasserstein distance per state (how fast is the
#    income distribution shape changing?)
# 2. Cross-state Wasserstein distance matrix for a given year
#    (which states have the most similar/different distributions?)
# 3. National average distributional drift over time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from sklearn.manifold import MDS
from pathlib import Path
import os
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESOLUTION = 101
BASE = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE / "data" / "processed"
TIMESERIES_PATH = BASE / "results" / "timeseries" / f"h0_gap_details_{RESOLUTION}pts_timeseries.csv"
OUT_DIR = Path(__file__).resolve().parent / "regression"
PLOT_DIR = BASE / "results" / "plots"

# Add src to path for data_loader imports
sys.path.insert(0, str(BASE / "src"))


def _load_percentile_vectors() -> dict[tuple[int, str], np.ndarray]:
    """Load all state-year percentile vectors from pickled ACS files."""
    from data_loader import NUM_PERCENTILES
    vectors = {}
    for year in range(2010, 2024):
        pkl_path = PROCESSED_DIR / f"acs_percentiles_{year}_{NUM_PERCENTILES}pts.pkl"
        if not pkl_path.exists():
            continue
        df = pd.read_pickle(pkl_path)
        p_col = next((c for c in df.columns if c.startswith("income_percentiles_")), None)
        if p_col is None:
            continue
        for _, row in df.iterrows():
            state = row["NAME"]
            if state == "Puerto Rico":
                continue
            vec = row[p_col]
            if isinstance(vec, np.ndarray) and vec.size > 0:
                vectors[(year, state)] = vec
    return vectors


def _wasserstein_1d(u: np.ndarray, v: np.ndarray) -> float:
    """Wasserstein-1 distance between two 1-D income distributions."""
    # Treat the percentile vectors as empirical quantile functions;
    # W₁ = mean absolute difference of quantile functions.
    return float(np.mean(np.abs(u - v)))


def run_wasserstein_analysis() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    vectors = _load_percentile_vectors()
    if not vectors:
        print("[wasserstein] No percentile vectors found. Run `make process` first.")
        return

    years = sorted({yr for yr, _ in vectors.keys()})
    states = sorted({st for _, st in vectors.keys()})
    print(f"[wasserstein] Loaded {len(vectors)} state-year vectors "
          f"({len(years)} years, {len(states)} states)")

    # ------------------------------------------------------------------
    # 1. Year-over-year drift per state
    # ------------------------------------------------------------------
    drift_rows = []
    for st in states:
        for i in range(len(years) - 1):
            y1, y2 = years[i], years[i + 1]
            if (y1, st) in vectors and (y2, st) in vectors:
                d = _wasserstein_1d(vectors[(y1, st)], vectors[(y2, st)])
                drift_rows.append({"state": st, "year_from": y1, "year_to": y2, "w1_drift": d})
    drift_df = pd.DataFrame(drift_rows)

    # Also compute cumulative drift from baseline (2010)
    cumul_rows = []
    base_year = years[0]
    for st in states:
        if (base_year, st) not in vectors:
            continue
        for yr in years[1:]:
            if (yr, st) in vectors:
                d = _wasserstein_1d(vectors[(base_year, st)], vectors[(yr, st)])
                cumul_rows.append({"state": st, "year": yr, "w1_from_baseline": d})
    cumul_df = pd.DataFrame(cumul_rows)

    # ------------------------------------------------------------------
    # 2. Cross-state distance matrix for most recent year
    # ------------------------------------------------------------------
    latest_year = years[-1]
    states_latest = [s for s in states if (latest_year, s) in vectors]
    n_st = len(states_latest)
    D = np.zeros((n_st, n_st))
    for i in range(n_st):
        for j in range(i + 1, n_st):
            d = _wasserstein_1d(vectors[(latest_year, states_latest[i])],
                                 vectors[(latest_year, states_latest[j])])
            D[i, j] = d
            D[j, i] = d

    # ------------------------------------------------------------------
    # 3. National summary
    # ------------------------------------------------------------------
    national_drift = drift_df.groupby("year_to")["w1_drift"].agg(["mean", "median", "std"]).reset_index()

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    report_lines = [
        "=" * 70,
        "WASSERSTEIN DISTANCE ANALYSIS",
        "=" * 70,
        f"States: {n_st}  |  Years: {years[0]}–{years[-1]}",
        "",
        "--- Year-over-Year National Drift (W₁, nominal $) ---",
    ]
    for _, r in national_drift.iterrows():
        report_lines.append(
            f"  {int(r['year_to']-1)}→{int(r['year_to'])}: "
            f"mean={r['mean']:,.0f}  median={r['median']:,.0f}  sd={r['std']:,.0f}"
        )

    # Top-5 most changed states (cumulative from baseline)
    if not cumul_df.empty:
        last_cumul = cumul_df[cumul_df["year"] == years[-1]].sort_values("w1_from_baseline", ascending=False)
        report_lines += [
            "",
            f"--- Largest Cumulative Distributional Shift ({base_year}→{years[-1]}) ---",
        ]
        for _, r in last_cumul.head(5).iterrows():
            report_lines.append(f"  {r['state']}: W₁ = ${r['w1_from_baseline']:,.0f}")

        report_lines += ["", f"--- Smallest Cumulative Distributional Shift ({base_year}→{years[-1]}) ---"]
        for _, r in last_cumul.tail(5).iterrows():
            report_lines.append(f"  {r['state']}: W₁ = ${r['w1_from_baseline']:,.0f}")

    report_text = "\n".join(report_lines)
    with open(OUT_DIR / "wasserstein_report.txt", "w") as fh:
        fh.write(report_text)
    print(report_text)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (a) National mean W₁ drift over time
    ax = axes[0, 0]
    ax.plot(national_drift["year_to"], national_drift["mean"], "o-", color="steelblue", lw=2)
    ax.fill_between(national_drift["year_to"],
                     national_drift["mean"] - national_drift["std"],
                     national_drift["mean"] + national_drift["std"],
                     alpha=0.2, color="steelblue")
    ax.set(xlabel="Year", ylabel="Mean W₁ Drift ($)",
           title="Year-over-Year Distributional Change (W₁)")

    # (b) Cumulative drift heatmap (states × years)
    ax = axes[0, 1]
    if not cumul_df.empty:
        pivot = cumul_df.pivot(index="state", columns="year", values="w1_from_baseline")
        # Sort states by final-year drift
        pivot = pivot.sort_values(pivot.columns[-1], ascending=False)
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", xticklabels=True,
                     yticklabels=True, cbar_kws={"label": "W₁ from baseline ($)"})
        ax.set_title(f"Cumulative Distributional Shift from {base_year}")
        ax.tick_params(axis="y", labelsize=5)
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)

    # (c) Cross-state distance matrix (latest year)
    ax = axes[1, 0]
    im = ax.imshow(D, cmap="viridis", aspect="auto")
    ax.set_xticks(range(n_st))
    ax.set_yticks(range(n_st))
    ax.set_xticklabels(states_latest, rotation=90, fontsize=4)
    ax.set_yticklabels(states_latest, fontsize=4)
    ax.set_title(f"Cross-State W₁ Distance ({latest_year})")
    plt.colorbar(im, ax=ax, label="W₁ ($)")

    # (d) MDS embedding of states based on W₁ distance
    ax = axes[1, 1]
    if n_st >= 3:
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
        coords = mds.fit_transform(D)
        ax.scatter(coords[:, 0], coords[:, 1], c="steelblue", edgecolor="k", s=50, alpha=0.8)
        for i, st in enumerate(states_latest):
            ax.annotate(st, (coords[i, 0], coords[i, 1]), fontsize=5, alpha=0.8)
        ax.set_title(f"MDS of State Income Distributions ({latest_year})")
        ax.set_xlabel("MDS-1")
        ax.set_ylabel("MDS-2")
    else:
        ax.text(0.5, 0.5, "Need ≥3 states for MDS", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle("Wasserstein Distance Analysis of Income Distributions", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "wasserstein_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save data
    drift_df.to_csv(OUT_DIR / "wasserstein_drift.csv", index=False)
    if not cumul_df.empty:
        cumul_df.to_csv(OUT_DIR / "wasserstein_cumulative.csv", index=False)

    print(f"[wasserstein] Saved plot → {PLOT_DIR / 'wasserstein_analysis.png'}")
    print(f"[wasserstein] Saved report → {OUT_DIR / 'wasserstein_report.txt'}")


if __name__ == "__main__":
    run_wasserstein_analysis()
