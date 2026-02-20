"""
CSV Plot Editor – load exported sweep CSVs, rename configs, and visualise
with multiple chart types via render_custom_plotly_chart.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.elements.lib.layout_utils import Width

try:
    from utils import render_custom_plotly_chart
except ImportError:

    def render_custom_plotly_chart(fig, width: Width = "stretch", key=None):
        st.plotly_chart(fig, width=width)


# ── Chart type helpers ────────────────────────────────────────────────────────

CHART_TYPES = [
    "Line",
    "Bar",
    "Grouped Bar",
    "Stacked Bar",
    "Stacked Area",
    "Heatmap",
]


def _build_line(fig: go.Figure, df: pd.DataFrame, x_col: str, y_col: str):
    for cfg in df["config"].unique():
        sub = df[df["config"] == cfg].sort_values(x_col)
        fig.add_trace(
            go.Scatter(
                x=sub[x_col],
                y=sub[y_col],
                mode="lines+markers",
                name=cfg,
            )
        )


def _build_bar(fig: go.Figure, df: pd.DataFrame, x_col: str, y_col: str):
    for cfg in df["config"].unique():
        sub = df[df["config"] == cfg].sort_values(x_col)
        fig.add_trace(
            go.Bar(
                x=sub[x_col],
                y=sub[y_col],
                name=cfg,
            )
        )
    fig.update_layout(barmode="overlay")


def _build_grouped_bar(
    fig: go.Figure, df: pd.DataFrame, x_col: str, y_col: str
):
    for cfg in df["config"].unique():
        sub = df[df["config"] == cfg].sort_values(x_col)
        fig.add_trace(
            go.Bar(
                x=sub[x_col],
                y=sub[y_col],
                name=cfg,
            )
        )
    fig.update_layout(barmode="group")


def _build_stacked_bar(
    fig: go.Figure, df: pd.DataFrame, x_col: str, y_col: str
):
    for cfg in df["config"].unique():
        sub = df[df["config"] == cfg].sort_values(x_col)
        fig.add_trace(
            go.Bar(
                x=sub[x_col],
                y=sub[y_col],
                name=cfg,
            )
        )
    fig.update_layout(barmode="stack")


def _build_stacked_area(
    fig: go.Figure, df: pd.DataFrame, x_col: str, y_col: str
):
    for cfg in df["config"].unique():
        sub = df[df["config"] == cfg].sort_values(x_col)
        fig.add_trace(
            go.Scatter(
                x=sub[x_col],
                y=sub[y_col],
                mode="lines",
                stackgroup="one",
                name=cfg,
            )
        )


def _build_heatmap(fig: go.Figure, df: pd.DataFrame, x_col: str, y_col: str):
    """Pivot configs as rows, x_col values as columns, cell = y_col value."""
    piv = df.pivot_table(
        index="config", columns=x_col, values=y_col, aggfunc="mean"
    )
    fig.add_trace(
        go.Heatmap(
            z=piv.values,
            x=[str(c) for c in piv.columns],
            y=list(piv.index),
            colorscale="Viridis",
        )
    )


_BUILDERS = {
    "Line": _build_line,
    "Bar": _build_bar,
    "Grouped Bar": _build_grouped_bar,
    "Stacked Bar": _build_stacked_bar,
    "Stacked Area": _build_stacked_area,
    "Heatmap": _build_heatmap,
}


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    st.set_page_config(page_title="CSV Plot Editor", layout="wide")
    st.title("CSV Plot Editor")
    st.markdown(
        "Upload CSV files exported from **Parameter Sweep Analyzer**, "
        "rename configs, pick a chart type, and render."
    )

    # ── 1. Upload CSVs ────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload one or more sweep-result CSVs",
        type=["csv"],
        accept_multiple_files=True,
    )
    if not uploaded_files:
        st.info("Upload at least one CSV to get started.")
        st.stop()

    # ── 2. Read & combine ─────────────────────────────────────────────────
    frames: list[pd.DataFrame] = []
    for uf in uploaded_files:
        try:
            df = pd.read_csv(uf)
        except Exception as e:
            st.error(f"Failed to read **{uf.name}**: {e}")
            continue
        if "config" not in df.columns:
            st.warning(
                f"**{uf.name}** has no `config` column – adding filename as config."
            )
            df["config"] = uf.name.removesuffix(".csv")
        frames.append(df)

    if not frames:
        st.error("No valid CSVs loaded.")
        st.stop()

    combined = pd.concat(frames, ignore_index=True)
    original_configs = list(combined["config"].unique())

    # ── 3. Rename configs ─────────────────────────────────────────────────
    st.sidebar.header("Config Renaming")
    rename_map: dict[str, str] = {}
    for cfg in original_configs:
        new_name = st.sidebar.text_input(
            f"Rename: {cfg}",
            value=cfg,
            key=f"rename_{cfg}",
        )
        rename_map[cfg] = new_name if new_name is not None else cfg

    combined["config"] = combined["config"].map(rename_map)

    # ── 4. Column / metric selection ──────────────────────────────────────
    non_metric_cols = {"config"}
    numeric_cols = [
        c
        for c in combined.select_dtypes(include="number").columns
        if c not in non_metric_cols
    ]
    all_cols = [c for c in combined.columns if c not in non_metric_cols]

    if not all_cols:
        st.error("CSV contains no usable columns besides `config`.")
        st.stop()

    st.sidebar.header("Axis Configuration")
    x_col = st.sidebar.selectbox("X-Axis column", options=all_cols)
    metric_options = [c for c in numeric_cols if c != x_col]
    if not metric_options:
        st.error("No numeric metric columns available for Y-axis.")
        st.stop()

    selected_metrics = st.sidebar.multiselect(
        "Y-Axis metric(s)", options=metric_options, default=metric_options[:1]
    )
    if not selected_metrics:
        st.info("Select at least one metric.")
        st.stop()

    # ── Axis label renaming ─────────────────────────────────────────────
    st.sidebar.header("Axis Labels")
    x_label = st.sidebar.text_input(
        "X-Axis label", value=x_col, key="x_axis_label"
    )
    x_label = x_label if x_label else x_col
    y_label = st.sidebar.text_input(
        "Y-Axis label",
        value=selected_metrics[0] if len(selected_metrics) == 1 else "Value",
        key="y_axis_label",
    )
    y_label = y_label if y_label else "Value"

    # ── 5. Config filter ──────────────────────────────────────────────────
    st.sidebar.header("Config Filter")
    all_configs = list(combined["config"].unique())
    selected_configs = st.sidebar.multiselect(
        "Show configs",
        options=all_configs,
        default=all_configs,
    )
    if not selected_configs:
        st.info("Select at least one config to display.")
        st.stop()

    plot_df = combined[combined["config"].isin(selected_configs)].copy()

    # ── 6. Chart type ─────────────────────────────────────────────────────
    st.sidebar.header("Chart Type")
    chart_type = st.sidebar.selectbox("Visualization", options=CHART_TYPES)

    # ── 7. Render plots ───────────────────────────────────────────────────
    for idx, metric in enumerate(selected_metrics):
        fig = go.Figure()
        _BUILDERS[chart_type](fig, plot_df, x_col, metric)
        fig.update_layout(
            title=f"{metric} vs {x_label}",
            xaxis_title=x_label,
            yaxis_title=y_label,
            template="plotly_white",
            height=600,
        )
        render_custom_plotly_chart(fig, width="stretch", key=f"csv_plot_{idx}")

    # ── 8. Data table + download ──────────────────────────────────────────
    st.subheader("Data Table")
    st.dataframe(plot_df, use_container_width=True)
    st.download_button(
        "Download combined CSV",
        plot_df.to_csv(index=False),
        "combined_results.csv",
        "text/csv",
    )


if __name__ == "__main__":
    main()
