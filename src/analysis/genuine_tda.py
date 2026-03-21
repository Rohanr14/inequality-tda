# src/analysis/genuine_tda.py
# Applies *genuine* persistent homology to multivariate state-level
# socioeconomic feature vectors using GUDHI.
#
# Instead of computing a simple 1-D max-gap, this module:
# 1. Builds a feature matrix: each row = one state-year, columns = income
#    quintile shares + Gini + Theil + gap percentile location.
# 2. Computes Vietoris-Rips persistence diagrams (H₀ and H₁).
# 3. Computes pairwise Wasserstein distances between per-year persistence
#    diagrams to measure how "inequality topology" evolves.
# 4. Generates persistence diagrams, barcodes, and a distance heatmap.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import gudhi
from gudhi.wasserstein import wasserstein_distance

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESOLUTION = 101
BASE = Path(__file__).resolve().parent.parent.parent
TIMESERIES_PATH = BASE / "results" / "timeseries" / f"h0_gap_details_{RESOLUTION}pts_timeseries.csv"
PROCESSED_DIR = BASE / "data" / "processed"
OUT_DIR = Path(__file__).resolve().parent / "regression"
PLOT_DIR = BASE / "results" / "plots"


def _build_feature_matrix(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Build a multivariate feature matrix from the timeseries results.
    Features per state-year:
      - acs_longest_h0_lifespan_real (gap magnitude)
      - acs_birth_percentile        (gap location)
      - gini
      - theil
      - acs_birth_income_real
      - acs_death_income_real
    """
    feature_cols = [
        "acs_longest_h0_lifespan_real",
        "acs_birth_percentile",
        "gini",
        "theil",
        "acs_birth_income_real",
        "acs_death_income_real",
    ]
    df = ts[["state", "year"] + feature_cols].dropna().copy()
    return df


def _compute_persistence(points: np.ndarray, max_dim: int = 1,
                          max_edge: float = float("inf")) -> list:
    """Compute Vietoris-Rips persistence diagram via GUDHI."""
    rips = gudhi.RipsComplex(points=points, max_edge_length=max_edge)
    st = rips.create_simplex_tree(max_dimension=max_dim + 1)
    st.compute_persistence()
    return st.persistence()


def _persistence_to_array(persistence: list, dim: int) -> np.ndarray:
    """Extract (birth, death) pairs for a given homological dimension."""
    pairs = [(b, d) for (dm, (b, d)) in persistence if dm == dim and d != float("inf")]
    if not pairs:
        return np.empty((0, 2))
    return np.array(pairs)


def run_genuine_tda() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    ts = pd.read_csv(TIMESERIES_PATH)
    feat_df = _build_feature_matrix(ts)

    years = sorted(feat_df["year"].unique())
    feature_cols = [c for c in feat_df.columns if c not in ("state", "year")]

    # --- Per-year persistence diagrams ---
    year_diagrams_h0 = {}
    year_diagrams_h1 = {}
    scaler = StandardScaler()

    report_lines = [
        "=" * 70,
        "GENUINE TDA: Multivariate Persistence Diagrams (GUDHI)",
        "=" * 70,
        f"Features used: {feature_cols}",
        f"Years analysed: {years[0]}–{years[-1]}",
        "",
    ]

    for yr in years:
        yr_data = feat_df[feat_df["year"] == yr][feature_cols].values
        yr_scaled = scaler.fit_transform(yr_data)

        persistence = _compute_persistence(yr_scaled, max_dim=1)
        h0 = _persistence_to_array(persistence, dim=0)
        h1 = _persistence_to_array(persistence, dim=1)

        year_diagrams_h0[yr] = h0
        year_diagrams_h1[yr] = h1

        report_lines.append(
            f"  {yr}: H₀ features={len(h0):3d}, H₁ features={len(h1):3d}"
            f"  | H₁ max lifespan={max((h1[:,1]-h1[:,0]).max(), 0) if len(h1)>0 else 0:.3f}"
        )

    # --- Pairwise Wasserstein distances between yearly H₁ diagrams ---
    n = len(years)
    W_h1 = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d1 = year_diagrams_h1[years[i]]
            d2 = year_diagrams_h1[years[j]]
            # Handle empty diagrams
            if len(d1) == 0 and len(d2) == 0:
                dist = 0.0
            elif len(d1) == 0 or len(d2) == 0:
                nonempty = d1 if len(d1) > 0 else d2
                dist = float(np.sum(nonempty[:, 1] - nonempty[:, 0]) / 2)
            else:
                dist = wasserstein_distance(d1, d2, order=1)
            W_h1[i, j] = dist
            W_h1[j, i] = dist

    report_lines += [
        "",
        "--- Pairwise Wasserstein-1 Distances (H₁ diagrams) ---",
        "  (see heatmap plot for visual; selected pairs below)",
    ]
    for offset in [1, 5, 10]:
        dists = [W_h1[i, i + offset] for i in range(n - offset)]
        if dists:
            report_lines.append(
                f"  Mean W₁ between years {offset} apart: {np.mean(dists):.4f}"
            )

    report_text = "\n".join(report_lines)
    with open(OUT_DIR / "genuine_tda_report.txt", "w") as fh:
        fh.write(report_text)
    print(report_text)

    # ---- PLOTS ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) Persistence diagram for most recent year
    latest_yr = years[-1]
    ax = axes[0, 0]
    h0 = year_diagrams_h0[latest_yr]
    h1 = year_diagrams_h1[latest_yr]
    if len(h0) > 0:
        ax.scatter(h0[:, 0], h0[:, 1], c="tab:blue", s=20, alpha=0.6, label=f"H₀ ({len(h0)})")
    if len(h1) > 0:
        ax.scatter(h1[:, 0], h1[:, 1], c="tab:orange", s=30, alpha=0.8, label=f"H₁ ({len(h1)})")
    lims = ax.get_xlim()
    ax.plot(lims, lims, "k--", lw=0.5, alpha=0.5)
    ax.set(xlabel="Birth", ylabel="Death", title=f"Persistence Diagram – {latest_yr}")
    ax.legend()

    # (b) Persistence barcode for most recent year
    ax = axes[0, 1]
    all_features = [(b, d, dim) for (dim, (b, d)) in _compute_persistence(
        scaler.fit_transform(feat_df[feat_df["year"] == latest_yr][feature_cols].values),
        max_dim=1) if d != float("inf")]
    all_features.sort(key=lambda x: x[2])
    colors = {"0": "tab:blue", "1": "tab:orange"}
    for i, (b, d, dim) in enumerate(all_features):
        ax.barh(i, d - b, left=b, height=0.8, color=colors.get(str(dim), "gray"), alpha=0.7)
    ax.set(xlabel="Filtration value", ylabel="Feature index",
           title=f"Persistence Barcode – {latest_yr}")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="tab:blue", label="H₀"), Patch(color="tab:orange", label="H₁")])

    # (c) Wasserstein distance heatmap (H₁)
    ax = axes[1, 0]
    im = ax.imshow(W_h1, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(years, fontsize=7)
    ax.set_title("Wasserstein-1 Distance (H₁) Between Years")
    plt.colorbar(im, ax=ax)

    # (d) H₁ feature count over time
    ax = axes[1, 1]
    h1_counts = [len(year_diagrams_h1[yr]) for yr in years]
    h1_max_life = [
        float((year_diagrams_h1[yr][:, 1] - year_diagrams_h1[yr][:, 0]).max())
        if len(year_diagrams_h1[yr]) > 0 else 0
        for yr in years
    ]
    ax.bar(years, h1_counts, color="tab:orange", alpha=0.7, label="H₁ count")
    ax2 = ax.twinx()
    ax2.plot(years, h1_max_life, "s-", color="tab:red", label="Max H₁ lifespan")
    ax.set(xlabel="Year", ylabel="# H₁ features", title="H₁ Topological Features Over Time")
    ax2.set_ylabel("Max H₁ lifespan")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)

    fig.suptitle("Genuine TDA: Multivariate Persistent Homology of Income Inequality", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "genuine_tda.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[genuine_tda] Saved plot → {PLOT_DIR / 'genuine_tda.png'}")
    print(f"[genuine_tda] Saved report → {OUT_DIR / 'genuine_tda_report.txt'}")


if __name__ == "__main__":
    run_genuine_tda()
