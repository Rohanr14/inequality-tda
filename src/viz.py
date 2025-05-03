# src/viz.py
# Visualization utilities for inequality‑tda.
# ‑‑‑
# ▪ leaderboard barplot (annotated with percentile location)
# ▪ time‑series with Gap + Gini + Theil & recession shading
# ▪ national‑median version of the above
# ▪ Δ‑vector plot (cleaner than a persistence diagram for 1‑D data)

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ph_pipeline import load_processed_acs_data, PERCENTILE_COL_PREFIX
from data_loader import NUM_PERCENTILES

# -----------------------------------------------------------------------------
# Constants & config
# -----------------------------------------------------------------------------

# (start, end) in YEAR.FRACTION
NBER_RECESSIONS: List[Tuple[float, float]] = [
    (2020, 2020.5),  # Covid recession (Feb–Apr 2020)
]

# -----------------------------------------------------------------------------
# Helper loaders
# -----------------------------------------------------------------------------

def load_timeseries_results(csv_path: str) -> Optional[pd.DataFrame]:
    """Read the master timeseries CSV produced by *ph_pipeline.py*."""
    if not os.path.exists(csv_path):
        print(f"[viz]   ✗ results file not found → {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"[viz]   ✗ failed to load {csv_path}: {e}")
        return None


def load_percentile_vector(state: str, year: int, template: str, num_pts: int = 101) -> Optional[np.ndarray]:
    """Return the 1‑D percentile income vector for *state, year* or None."""
    path = template.format(year=year, num_pts=num_pts)
    acs_df = load_processed_acs_data(path)
    if acs_df is None:
        return None
    acs_df = acs_df[acs_df["NAME"] != "Puerto Rico"]
    row = acs_df[acs_df["NAME"] == state]
    if row.empty:
        print(f"[viz]   ✗ {state} not found in {path}")
        return None
    p_col = next((c for c in row.columns if c.startswith(PERCENTILE_COL_PREFIX)), None)
    return row.iloc[0][p_col] if p_col else None

# -----------------------------------------------------------------------------
# Plotting primitives
# -----------------------------------------------------------------------------

def _shade_recessions(ax):
    for start, end in NBER_RECESSIONS:
        ax.axvspan(start, end, color="grey", alpha=0.15)
        ax.text((start + end) / 2,
                 ax.get_ylim()[1] * 0.95,
                 "COVID‑19 recession",
                 color="dimgray", ha="center", va="top",
                 fontsize=8, rotation=90)

# -----------------------------------------------------------------------------
# 1) Leaderboard  (annotated)
# -----------------------------------------------------------------------------

def plot_leaderboard(df: pd.DataFrame, year: int, out_path: str, *, palette: str = "viridis") -> None:
    col = "acs_longest_h0_lifespan_real"
    d = (
        df.query("year == @year")
          .sort_values(col, ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 12))
    sns.barplot(data=d, y="state", x=col, palette=palette, ax=ax, hue="state", legend=False)

    # annotate with percentile location (integer)
    for i, (gap, pct) in enumerate(zip(d[col], d["acs_birth_percentile"])):
        ax.text(gap + 200, i, f"@ {pct:.0f}ᵗʰ", va="center", fontsize=7)

    ax.set(
        title=f"Top {len(d)} States – Largest Income Gap (ACS) {year}",
        xlabel="Largest H₀ gap (2024 $)",
        ylabel="State",
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# -----------------------------------------------------------------------------
# 2) State‑level time‑series with Gap + Gini + Theil
# -----------------------------------------------------------------------------

def plot_state_timeseries(df: pd.DataFrame, states: List[str], out_path: str) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    ax1.set_zorder(1)
    ax1.patch.set_visible(False)

    for st in states:
        sub = df[df["state"] == st]

        # ── Topological gap (constant 2024 $) ───────────────────────────────
        ax1.plot(sub["year"],
                 sub["acs_longest_h0_lifespan_real"],
                 lw=2,
                 label=f"{st} – Gap (real $)")

        # ▼ Bootstrap 95% CI ribbon
        ax1.fill_between(sub["year"],
                         sub["acs_gap_lo_real"],
                         sub["acs_gap_hi_real"],
                         alpha=0.15,
                         color=ax1.lines[-1].get_color())

        # ── Classical metrics (unitless) ───────────────────────────────────
        ax2.plot(sub["year"], sub["gini"],
                 ls="--", alpha=0.55, label=f"{st} – Gini")
        ax2.plot(sub["year"], sub["theil"],
                 ls=":",  alpha=0.55, label=f"{st} – Theil")

    ax1.set_ylabel("Largest H₀ gap (2024 $)", color="steelblue")
    ax2.set_ylabel("Gini / Theil", color="olive")

    _shade_recessions(ax1)

    ax1.set(title="Topological Gap vs Gini & Theil – Selected States", xlabel="Year")
    # build one combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# -----------------------------------------------------------------------------
# 3) National‑median / mean aggregate
# -----------------------------------------------------------------------------

def plot_national_median(df: pd.DataFrame, out_path: str, agg: str = "median"):
    # use real‑$ gap column
    num_cols = ["acs_longest_h0_lifespan_real", "gini", "theil"]

    if agg == "median":
        g = df.groupby("year")[num_cols].median(numeric_only=True)
        agg_title = "Median"
    elif agg == "mean":
        g = df.groupby("year")[num_cols].mean(numeric_only=True)
        agg_title = "Mean"
    else:
        raise ValueError("agg must be 'median' or 'mean'")

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    ax1.plot(g.index, g["acs_longest_h0_lifespan_real"], label=f"Gap ({agg_title} state, real $)")
    ax2.plot(g.index, g["gini"], ls="--", label=f"Gini ({agg_title})")
    ax2.plot(g.index, g["theil"], ls=":", label=f"Theil ({agg_title})")

    _shade_recessions(ax1)

    ax1.set_ylabel("Largest H₀ gap (median state, 2024 $)", color="steelblue")
    ax2.set_ylabel("Gini / Theil", color="olive")
    ax1.set(title=f"H₀ Gap, Gini & Theil – National {agg_title} over Time", xlabel="Year")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2,
               bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# -----------------------------------------------------------------------------
# 4) Δ‑vector plot  (1‑D diagnostic)
# -----------------------------------------------------------------------------

def plot_delta_vector(vector: np.ndarray, state: str, year: int, out_path: str) -> None:
    """Plot consecutive dollar‑differences; mark the max gap."""
    diffs = np.diff(vector)
    idx = int(np.argmax(diffs))

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(range(1, len(vector)), diffs, marker="o")
    ax.scatter(idx + 1, diffs[idx], color="red", zorder=3, label="Largest gap")

    ax.set(
        title=f"Δ‑Vector – {state} {year} (ACS)",
        xlabel="Percentile index",
        ylabel="Income jump ($)",
    )
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Example CLI usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_csv = os.path.join(BASE, "results", "timeseries", f"h0_gap_details_{NUM_PERCENTILES}pts_timeseries.csv")
    plots_dir   = os.path.join(BASE, "results", "plots")
    template    = os.path.join(BASE, "data", "processed", "acs_percentiles_{year}_{num_pts}pts.pkl")

    df_res = load_timeseries_results(results_csv)
    if df_res is None:
        exit()

    # 1) Leaderboard 2023 (top 50)
    plot_leaderboard(df_res, 2023, os.path.join(plots_dir, "leaderboard_2023_annot.png"))

    # 2) Selected‑state time‑series
    some_states = ["Mississippi", "Texas", "Wyoming"]
    plot_state_timeseries(df_res, some_states, os.path.join(plots_dir, "timeseries_selected.png"))

    # 3) National median
    plot_national_median(df_res, os.path.join(plots_dir, "timeseries_national_median.png"), agg="median")

    # 4) Δ‑vector for CA 2020
    vec_ca_2020 = load_percentile_vector("California", 2020, template)
    if vec_ca_2020 is not None:
        plot_delta_vector(vec_ca_2020, "California", 2020, os.path.join(plots_dir, "delta_CA_2020.png"))

    print("[viz] ✓ finished example run")