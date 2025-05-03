# ─── imports and page config stay the same ────────────────────────────────
import altair as alt
import pandas as pd
import streamlit as st
from pathlib import Path

alt.data_transformers.disable_max_rows()
st.set_page_config(page_title="Income‑Gap Dashboard", layout="wide")
DATA_PATH = (Path(__file__).resolve().parent.parent.parent / "results/timeseries/h0_gap_details_101pts_timeseries.csv")
US_TOPOJSON = "https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH).rename(
        columns={
            "acs_longest_h0_lifespan_real": "gap_real",
            "acs_gap_lo_real": "gap_lo",
            "acs_gap_hi_real": "gap_hi",
        }
    )
    return df

df = load_data()
years  = sorted(df.year.unique())
states = sorted(df.state.unique())

st.title("Topological Income‑Gap Dashboard")
st.caption("Largest H₀ gap · bootstrap 95% CIs · real 2024 dollars")

tab_nat, tab_state = st.tabs(["Nation snapshot", "State drill‑down"])

# ────────────────────────────  Tab 1  ─────────────────────────────────────
with tab_nat:
    yr = st.slider(
        "Select year",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=int(max(years)),
        step=1,
        key="year_slider"
    )
    df_yr = df[df.year == yr]

    states_geo = alt.topo_feature(US_TOPOJSON, feature="states")


    choropleth = (
        alt.Chart(states_geo)
        .mark_geoshape(stroke="white")
        .encode(
            color=alt.Color("gap_real:Q",
                            scale=alt.Scale(scheme="blues"),
                            title="Largest H₀ gap (2024 $)"),
            tooltip=[
                alt.Tooltip("properties.name:N", title="State"),
                alt.Tooltip("gap_real:Q",        title="Gap (2024 $)", format=",.0f"),
            ],
        )
        .project(type="albersUsa")
        .transform_lookup(
            lookup="properties.name",
            from_=alt.LookupData(df_yr, "state", ["gap_real"])
        )
        .properties(width=850, height=500,
                    title=f"{yr}: Largest H₀ gap by state")
    )

    st.altair_chart(choropleth, use_container_width=True)
    st.markdown("---")

# ────────────────────────────  Tab 2  ────────────────────────────────────
with (tab_state):
    st.sidebar.header("State drill‑down controls")
    sel_state = st.sidebar.selectbox("State", states, index=states.index("California"))
    st_df = df[df.state == sel_state]

    hover = alt.selection_point(fields=["year"], nearest=True, on="mouseover", empty=False)

    # Tooltip setup
    gap_tooltip = [
        alt.Tooltip("year:O",     title="Year"),
        alt.Tooltip("gap_real:Q", title="Gap (2024 $)", format=",.0f"),
        alt.Tooltip("gap_hi:Q", title="CI high", format=",.0f"),
        alt.Tooltip("gap_lo:Q",   title="CI low",       format=",.0f"),
    ]
    gini_tooltip = [
        alt.Tooltip("year:O"),
        alt.Tooltip("gini:Q", title="Gini", format=".3f")
    ]
    theil_tooltip = [
        alt.Tooltip("year:O"),
        alt.Tooltip("theil:Q", title="Theil", format=".3f")
    ]

    ribbon = (
        alt.Chart(st_df)
        .mark_area(opacity=0.20, color="#1f77b4")
        .encode(
            x=alt.X("year:O", title="Year"),
            y="gap_hi:Q",
            y2="gap_lo:Q"
        )
        .properties(width=520, height=550)
    )

    # Gap line (solid)
    gap_line = (
        alt.Chart(st_df)
        .mark_line(size=3, color="#1f77b4")
        .encode(
            x="year:O",
            y=alt.Y("gap_real:Q", title="Largest H₀ gap (2024 $)")
        )
    )

    gap_points = (
        alt.Chart(st_df)
        .mark_point(size=60, color="#1f77b4", filled=True)
        .encode(
            x="year:O",
            y=alt.Y("gap_real:Q"),
            tooltip=gap_tooltip,
            opacity=alt.value(1)
        )
    )

    gap_panel = alt.layer(ribbon, gap_line, gap_points).resolve_scale(y="shared")

    # Gini line (dash)
    gini_line = (
        alt.Chart(st_df)
        .mark_line(size=3, strokeDash=[6, 3], color="red")
        .encode(
            x="year:O",
            y=alt.Y("gini:Q", axis=alt.Axis(title="Gini / Theil"))
        )
    )

    gini_points = (
        alt.Chart(st_df)
        .mark_point(size=60, color="orange", filled=True)
        .encode(
            x="year:O",
            y=alt.Y("gini:Q"),
            tooltip=gini_tooltip
        )
    )

    # Theil line (dot‑dash)
    theil_line = (
        alt.Chart(st_df)
        .mark_line(size=3, strokeDash=[2, 2], color="darkgreen")
        .encode(
            x="year:O",
            y="theil:Q"
        )
    )

    theil_points = (
        alt.Chart(st_df)
        .mark_point(size=60, color="goldenrod", filled=True)
        .encode(
            x="year:O",
            y=alt.Y("theil:Q"),
            tooltip=theil_tooltip  # Add tooltips here
        )
    )

    metric_panel = alt.layer(gini_line, theil_line, gini_points, theil_points).resolve_scale(y="shared").properties(width=500, height=550)
    chart = alt.hconcat(gap_panel, metric_panel).resolve_scale(y="independent").properties(title = f"{sel_state}: Gap (95% CI) vs. Gini & Theil")

    st.altair_chart(chart, use_container_width=True)

    # Download button
    st.download_button(
        "Download full timeseries CSV",
        df.to_csv(index=False).encode("utf-8"),
        "income_gap_timeseries.csv",
        "text/csv",
    )