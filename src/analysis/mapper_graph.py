# src/analysis/mapper_graph.py
# Builds a Mapper graph (a core TDA tool) from state-level income inequality
# features, revealing the topological structure of US economic inequality.
#
# Nodes are clusters of state-years with similar inequality profiles;
# edges connect overlapping clusters. The resulting graph reveals:
#   - Which states cluster together across time
#   - Whether geographic or temporal patterns dominate
#   - Structural "branches" in inequality space (e.g., high-gap/low-Gini
#     vs high-Gini/low-gap trajectories)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from pathlib import Path
import json
import kmapper as km

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESOLUTION = 101
BASE = Path(__file__).resolve().parent.parent.parent
TIMESERIES_PATH = BASE / "results" / "timeseries" / f"h0_gap_details_{RESOLUTION}pts_timeseries.csv"
OUT_DIR = Path(__file__).resolve().parent / "regression"
PLOT_DIR = BASE / "results" / "plots"

# US Census region assignments for colouring
REGIONS = {
    "Northeast": [
        "Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island",
        "Vermont", "New Jersey", "New York", "Pennsylvania",
    ],
    "Midwest": [
        "Illinois", "Indiana", "Iowa", "Kansas", "Michigan", "Minnesota",
        "Missouri", "Nebraska", "North Dakota", "Ohio", "South Dakota", "Wisconsin",
    ],
    "South": [
        "Alabama", "Arkansas", "Delaware", "Florida", "Georgia", "Kentucky",
        "Louisiana", "Maryland", "Mississippi", "North Carolina", "Oklahoma",
        "South Carolina", "Tennessee", "Texas", "Virginia", "West Virginia",
        "District of Columbia",
    ],
    "West": [
        "Alaska", "Arizona", "California", "Colorado", "Hawaii", "Idaho",
        "Montana", "Nevada", "New Mexico", "Oregon", "Utah", "Washington", "Wyoming",
    ],
}

STATE_TO_REGION = {}
for region, sts in REGIONS.items():
    for s in sts:
        STATE_TO_REGION[s] = region


def run_mapper_graph() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    ts = pd.read_csv(TIMESERIES_PATH)

    # --- Build feature matrix ---
    feature_cols = [
        "acs_longest_h0_lifespan_real",
        "acs_birth_percentile",
        "gini",
        "theil",
        "acs_birth_income_real",
        "acs_death_income_real",
    ]
    df = ts[["state", "year"] + feature_cols].dropna().copy()
    df["region"] = df["state"].map(STATE_TO_REGION).fillna("Other")

    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    labels = [f"{r['state']} {r['year']}" for _, r in df.iterrows()]

    # --- Mapper ---
    mapper = km.KeplerMapper(verbose=0)

    # Project onto first 2 PCs for the lens function
    lens = mapper.fit_transform(X_scaled, projection=PCA(n_components=2, random_state=42))

    graph = mapper.map(
        lens,
        X_scaled,
        cover=km.Cover(n_cubes=15, perc_overlap=0.4),
        clusterer=DBSCAN(eps=0.8, min_samples=3),
    )

    # --- Save interactive HTML ---
    html_path = str(PLOT_DIR / "mapper_graph.html")
    color_values = df["year"].values.astype(float)
    mapper.visualize(
        graph,
        path_html=html_path,
        title="Mapper Graph: US Income Inequality Topology",
        custom_tooltips=np.array(labels),
        color_values=color_values,
        color_function_name="Year",
    )
    print(f"[mapper] Saved interactive HTML → {html_path}")

    # --- Static matplotlib plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # (a) Mapper graph drawn manually
    ax = axes[0]
    node_positions = {}
    # Layout: use mean PCA coordinates of members for each node
    for node_id, member_indices in graph["nodes"].items():
        node_positions[node_id] = lens[member_indices].mean(axis=0)

    # Draw edges
    for edge in graph["links"]:
        for target in graph["links"][edge]:
            p1 = node_positions.get(edge)
            p2 = node_positions.get(target)
            if p1 is not None and p2 is not None:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", alpha=0.3, lw=0.5)

    # Draw nodes (coloured by mean year of members)
    if node_positions:
        node_ids = list(node_positions.keys())
        xs = [node_positions[n][0] for n in node_ids]
        ys = [node_positions[n][1] for n in node_ids]
        node_colors = []
        node_sizes = []
        for n in node_ids:
            members = graph["nodes"][n]
            mean_year = df.iloc[members]["year"].mean()
            node_colors.append(mean_year)
            node_sizes.append(len(members) * 8)

        sc = ax.scatter(xs, ys, c=node_colors, s=node_sizes, cmap="viridis",
                        edgecolor="k", linewidth=0.5, alpha=0.85)
        plt.colorbar(sc, ax=ax, label="Mean Year")
    ax.set(xlabel="PCA-1", ylabel="PCA-2", title="Mapper Graph (nodes = clusters of state-years)")

    # (b) PCA scatter coloured by region
    ax = axes[1]
    region_colors = {"Northeast": "tab:blue", "Midwest": "tab:green",
                     "South": "tab:red", "West": "tab:orange", "Other": "gray"}
    for region, color in region_colors.items():
        mask = df["region"] == region
        if mask.any():
            ax.scatter(lens[mask, 0], lens[mask, 1], c=color, s=15,
                       alpha=0.5, label=region, edgecolor="none")
    ax.legend(fontsize=8)
    ax.set(xlabel="PCA-1", ylabel="PCA-2",
           title="State-Years in Feature Space (by Census Region)")

    fig.suptitle("Mapper Graph: Topological Structure of US Income Inequality", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "mapper_graph_static.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Summary report ---
    n_nodes = len(graph["nodes"])
    n_edges = sum(len(v) for v in graph["links"].values())
    report_lines = [
        "=" * 70,
        "MAPPER GRAPH ANALYSIS",
        "=" * 70,
        f"Input: {len(df)} state-year observations, {len(feature_cols)} features",
        f"Mapper graph: {n_nodes} nodes, {n_edges} edges",
        f"Cover: 15 cubes, 40% overlap",
        f"Clusterer: DBSCAN(eps=0.8, min_samples=3)",
        "",
        "--- Node Size Distribution ---",
    ]
    sizes = [len(v) for v in graph["nodes"].values()]
    if sizes:
        report_lines += [
            f"  Min members: {min(sizes)}",
            f"  Max members: {max(sizes)}",
            f"  Mean members: {np.mean(sizes):.1f}",
            f"  Median members: {np.median(sizes):.1f}",
        ]

    # Identify which regions dominate each connected component
    report_lines += ["", "--- Regional Composition of Mapper Nodes (top 5 largest) ---"]
    sorted_nodes = sorted(graph["nodes"].items(), key=lambda x: len(x[1]), reverse=True)
    for node_id, members in sorted_nodes[:5]:
        node_df = df.iloc[members]
        region_counts = node_df["region"].value_counts()
        year_range = f"{node_df['year'].min()}–{node_df['year'].max()}"
        report_lines.append(
            f"  Node {node_id} ({len(members)} members, years {year_range}):"
        )
        for region, count in region_counts.items():
            report_lines.append(f"    {region}: {count}")

    report_text = "\n".join(report_lines)
    with open(OUT_DIR / "mapper_report.txt", "w") as fh:
        fh.write(report_text)
    print(report_text)
    print(f"[mapper] Saved static plot → {PLOT_DIR / 'mapper_graph_static.png'}")
    print(f"[mapper] Saved report → {OUT_DIR / 'mapper_report.txt'}")


if __name__ == "__main__":
    run_mapper_graph()
