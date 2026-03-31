"""
CSV Plot Editor v3.1 – load exported sweep CSVs, rename configs, and visualise
with multiple chart types via render_custom_plotly_chart.

Pipeline (JSON mode):
  1. extract_columns    – regex / auto-KV new columns from text
  2. computed_columns   – derive new columns
  3. filters            – drop non-matching rows
  4. x.values filter    – keep only requested x-values (no rename yet)
  5. aggregate          – collapse rows (mean/sum/…) + error bars
  6. x rename + order   – categorical ordering and display labels
  7. transform          – normalize stacks, etc.
  8. group → config     – split into Plotly traces
  9. auto-dedup         – safety net: aggregate any remaining duplicates
  10. render            – build Plotly figure

Features:
  - Multiple JSON spec files (merged)
  - Save all plots: HTML / PNG / SVG
  - Aggregation: mean, median, sum, count, min, max, std
  - Error bars: std, sem, minmax, q25_q75
  - Column extraction from text (regex named groups or auto Key:Value)
  - Computed columns (pd.eval expressions)
  - Normalize transform for stacked charts
  - Data table toggle per plot
"""

from __future__ import annotations

import io
import json
import re
import time
import zipfile
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit.elements.lib.layout_utils import Width
except ImportError:
    from typing import Literal

    Width = Literal["stretch"]  # type: ignore[assignment, misc]

try:
    from utils import render_custom_plotly_chart
except ImportError:

    def render_custom_plotly_chart(
        fig: go.Figure, width: Any = "stretch", key: str | None = None
    ) -> None:
        st.plotly_chart(fig, width=width)


# ── Aggregation helpers ──────────────────────────────────────────────────────

_AGG_FUNCS: dict[str, str] = {
    "mean": "mean",
    "median": "median",
    "sum": "sum",
    "count": "count",
    "min": "min",
    "max": "max",
    "std": "std",
    "first": "first",
    "last": "last",
}


def _aggregate_data(
    df: pd.DataFrame,
    group_keys: list[str],
    value_cols: list[str],
    func: str = "mean",
    error_bars: str | None = None,
) -> pd.DataFrame:
    """Aggregate *df* by *group_keys*, computing *func* on *value_cols*."""
    agg_func = _AGG_FUNCS.get(func, "mean")
    existing_keys = [k for k in group_keys if k in df.columns]
    existing_vals = [v for v in value_cols if v in df.columns]
    if not existing_keys or not existing_vals:
        return df

    agg_dict: dict[str, str] = {col: agg_func for col in existing_vals}
    result: pd.DataFrame = (
        df.groupby(existing_keys, observed=True).agg(agg_dict).reset_index()
    )

    if error_bars and error_bars != "none":
        for col in existing_vals:
            if error_bars == "std":
                err = (
                    df.groupby(existing_keys, observed=True)[col]
                    .std()
                    .reset_index()
                    .rename(columns={col: f"{col}__err"})
                )
                result = result.merge(err, on=existing_keys, how="left")
                result[f"{col}__err"] = result[f"{col}__err"].fillna(0)
            elif error_bars == "sem":
                err = (
                    df.groupby(existing_keys, observed=True)[col]
                    .sem()
                    .reset_index()
                    .rename(columns={col: f"{col}__err"})
                )
                result = result.merge(err, on=existing_keys, how="left")
                result[f"{col}__err"] = result[f"{col}__err"].fillna(0)
            elif error_bars == "minmax":
                mn = (
                    df.groupby(existing_keys, observed=True)[col]
                    .min()
                    .reset_index()
                    .rename(columns={col: f"{col}__err_minus"})
                )
                mx = (
                    df.groupby(existing_keys, observed=True)[col]
                    .max()
                    .reset_index()
                    .rename(columns={col: f"{col}__err_plus"})
                )
                result = result.merge(mn, on=existing_keys, how="left")
                result = result.merge(mx, on=existing_keys, how="left")
                result[f"{col}__err_minus"] = (
                    result[col] - result[f"{col}__err_minus"]
                )
                result[f"{col}__err_plus"] = (
                    result[f"{col}__err_plus"] - result[col]
                )
            elif error_bars == "q25_q75":
                q25 = (
                    df.groupby(existing_keys, observed=True)[col]
                    .quantile(0.25)
                    .reset_index()
                    .rename(columns={col: f"{col}__err_minus"})
                )
                q75 = (
                    df.groupby(existing_keys, observed=True)[col]
                    .quantile(0.75)
                    .reset_index()
                    .rename(columns={col: f"{col}__err_plus"})
                )
                result = result.merge(q25, on=existing_keys, how="left")
                result = result.merge(q75, on=existing_keys, how="left")
                result[f"{col}__err_minus"] = (
                    result[col] - result[f"{col}__err_minus"]
                )
                result[f"{col}__err_plus"] = (
                    result[f"{col}__err_plus"] - result[col]
                )
    return result


def _compute_columns(
    df: pd.DataFrame, computed: list[dict[str, str]]
) -> pd.DataFrame:
    """Add computed columns via ``pd.eval``."""
    for spec in computed:
        name = spec.get("name")
        expr = spec.get("expr")
        if not name or not expr:
            continue
        try:
            df[name] = df.eval(expr)
        except Exception as exc:
            st.warning(f"Computed column `{name}` failed: {exc}")
    return df


def _extract_regex_columns(
    df: pd.DataFrame, extractions: list[dict[str, Any]]
) -> pd.DataFrame:
    """Extract new columns from text columns using regex or auto key:value detection.

    Each extraction spec can be:
      - ``{"source": "config", "pattern": "SNR:(?P<SNR>[^|]+)"}``
        Named-group regex — each ``(?P<name>...)`` becomes a column.
      - ``{"source": "config", "auto_kv": true}``
        Auto-split on ``|`` then ``:`` to discover all Key:Value pairs.
      - ``{"source": "config", "auto_kv": true, "separator": "|",
            "kv_separator": ":", "keys": ["SNR", "Model"]}``
        Same, but keep only listed keys.
    """
    for spec in extractions:
        source = spec.get("source", "config")
        if source not in df.columns:
            st.warning(f"Extract: column `{source}` not found.")
            continue

        pattern = spec.get("pattern")
        auto_kv = spec.get("auto_kv", False)

        if pattern:
            try:
                compiled = re.compile(pattern)
                if not compiled.groupindex:
                    st.warning(
                        "Regex has no named groups — use `(?P<Name>...)` syntax."
                    )
                    continue
                extracted = df[source].astype(str).str.extract(compiled)
                for col in extracted.columns:
                    df[col] = extracted[col].str.strip()
                    numeric = pd.to_numeric(df[col], errors="coerce")
                    if numeric.notna().all():
                        df[col] = numeric
            except re.error as exc:
                st.warning(f"Regex extraction failed: {exc}")

        elif auto_kv:
            separator = spec.get("separator", "|")
            kv_separator = spec.get("kv_separator", ":")
            keys_filter: list[str] | None = spec.get("keys")

            def _parse_kv(
                text: str,
                _sep: str = separator,
                _kv_sep: str = kv_separator,
            ) -> dict[str, str]:
                result: dict[str, str] = {}
                for part in str(text).split(_sep):
                    part = part.strip()
                    if _kv_sep in part:
                        k, v = part.split(_kv_sep, 1)
                        result[k.strip()] = v.strip()
                return result

            parsed = df[source].apply(_parse_kv)
            kv_df = pd.DataFrame(parsed.tolist(), index=df.index)

            if keys_filter:
                kv_df = kv_df[[k for k in keys_filter if k in kv_df.columns]]

            for col in kv_df.columns:
                if not col:
                    continue
                df[col] = kv_df[col]
                numeric = pd.to_numeric(df[col], errors="coerce")
                if numeric.notna().all():
                    df[col] = numeric

    return df


def _normalize_stacked(df: pd.DataFrame, y_columns: list[str]) -> pd.DataFrame:
    """Normalize stacked Y columns so each row sums to 100 %."""
    existing = [c for c in y_columns if c in df.columns]
    if not existing:
        return df
    df = df.copy()
    row_sums: pd.Series = df[existing].sum(axis=1)  # type: ignore[assignment]
    safe_sums: pd.Series = row_sums.where(
        row_sums.astype(float) != 0.0, other=float("nan")
    )
    for c in existing:
        df[c] = df[c] / safe_sums * 100.0
    return df


def _normalize_distribution(
    df: pd.DataFrame,
    y_columns: list[str],
    ref_column: str | None = None,
) -> pd.DataFrame:
    """Normalize Y columns as a per-row distribution (%).

    When *ref_column* is provided, each row's Y values are divided by that
    row's *ref_column* value (per-bin normalization).  Otherwise each column
    is divided by its own sum independently.
    """
    existing = [c for c in y_columns if c in df.columns]
    if not existing:
        return df
    df = df.copy()
    if ref_column and ref_column in df.columns:
        total = float(df[ref_column].sum())
        if total > 0:
            for c in existing:
                df[c] = df[c] / total * 100.0
    else:
        for c in existing:
            col_sum = float(df[c].sum())
            if col_sum > 0:
                df[c] = df[c] / col_sum * 100.0
    return df


# ── Chart builders ───────────────────────────────────────────────────────────

CHART_TYPES: list[str] = [
    "Line",
    "Bar",
    "Grouped Bar",
    "Stacked Bar",
    "Stacked Area",
    "Heatmap",
]

_CHART_TYPE_MAP: dict[str, str] = {
    "line": "Line",
    "bar": "Bar",
    "grouped_bar": "Grouped Bar",
    "stacked_bar": "Stacked Bar",
    "stacked_area": "Stacked Area",
    "heatmap": "Heatmap",
}


def _get_error_arrays(df: pd.DataFrame, y_col: str) -> dict[str, Any] | None:
    sym_err = f"{y_col}__err"
    asym_plus = f"{y_col}__err_plus"
    asym_minus = f"{y_col}__err_minus"
    if sym_err in df.columns:
        return dict(type="data", array=df[sym_err].tolist(), visible=True)
    if asym_plus in df.columns and asym_minus in df.columns:
        return dict(
            type="data",
            symmetric=False,
            array=df[asym_plus].tolist(),
            arrayminus=df[asym_minus].tolist(),
            visible=True,
        )
    return None


_DASH_STYLES: list[str] = [
    "solid",
    "dot",
    "dash",
    "longdash",
    "dashdot",
    "longdashdot",
]

_DASH_MAP: dict[str, str] = {s: s for s in _DASH_STYLES}
_DASH_MAP.update({"dotted": "dot", "dashed": "dash"})


def _resolve_dash(
    cfg: str,
    idx: int,
    line_dash: str | dict[str, str] | None,
) -> str:
    """Return a Plotly dash string for the given config/index."""
    if line_dash is None:
        return "solid"
    if isinstance(line_dash, dict):
        raw = line_dash.get(cfg, "solid")
    else:
        raw = str(line_dash)
    return _DASH_MAP.get(raw, raw)


def _build_line(
    fig: go.Figure,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    line_dash: str | dict[str, str] | None = None,
    opacity: float | None = None,
) -> None:
    for idx, cfg in enumerate(df["config"].unique()):
        sub = df[df["config"] == cfg].sort_values(x_col)
        dash = _resolve_dash(cfg, idx, line_dash)
        trace_kw: dict[str, Any] = dict(
            x=sub[x_col],
            y=sub[y_col],
            mode="lines+markers",
            name=str(cfg),
            error_y=_get_error_arrays(sub, y_col),
            line=dict(dash=dash),
        )
        if opacity is not None:
            trace_kw["opacity"] = opacity
        fig.add_trace(go.Scatter(**trace_kw))


def _build_bar(
    fig: go.Figure,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    line_dash: str | dict[str, str] | None = None,
    opacity: float | None = None,
) -> None:
    for cfg in df["config"].unique():
        sub = df[df["config"] == cfg].sort_values(x_col)
        trace_kw: dict[str, Any] = dict(
            x=sub[x_col],
            y=sub[y_col],
            name=str(cfg),
            error_y=_get_error_arrays(sub, y_col),
        )
        if opacity is not None:
            trace_kw["opacity"] = opacity
        fig.add_trace(go.Bar(**trace_kw))
    fig.update_layout(barmode="overlay")


def _build_grouped_bar(
    fig: go.Figure,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    line_dash: str | dict[str, str] | None = None,
    opacity: float | None = None,
) -> None:
    for cfg in df["config"].unique():
        sub = df[df["config"] == cfg].sort_values(x_col)
        trace_kw: dict[str, Any] = dict(
            x=sub[x_col],
            y=sub[y_col],
            name=str(cfg),
            error_y=_get_error_arrays(sub, y_col),
        )
        if opacity is not None:
            trace_kw["opacity"] = opacity
        fig.add_trace(go.Bar(**trace_kw))
    fig.update_layout(barmode="group")


def _build_stacked_bar(
    fig: go.Figure,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    line_dash: str | dict[str, str] | None = None,
    opacity: float | None = None,
) -> None:
    for cfg in df["config"].unique():
        sub = df[df["config"] == cfg].sort_values(x_col)
        trace_kw: dict[str, Any] = dict(
            x=sub[x_col], y=sub[y_col], name=str(cfg)
        )
        if opacity is not None:
            trace_kw["opacity"] = opacity
        fig.add_trace(go.Bar(**trace_kw))
    fig.update_layout(barmode="stack")


def _build_stacked_area(
    fig: go.Figure,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    line_dash: str | dict[str, str] | None = None,
    opacity: float | None = None,
) -> None:
    for idx, cfg in enumerate(df["config"].unique()):
        sub = df[df["config"] == cfg].sort_values(x_col)
        dash = _resolve_dash(cfg, idx, line_dash)
        trace_kw: dict[str, Any] = dict(
            x=sub[x_col],
            y=sub[y_col],
            mode="lines",
            stackgroup="one",
            name=str(cfg),
            line=dict(dash=dash),
        )
        if opacity is not None:
            trace_kw["opacity"] = opacity
        fig.add_trace(go.Scatter(**trace_kw))


def _build_heatmap(
    fig: go.Figure,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    line_dash: str | dict[str, str] | None = None,
    opacity: float | None = None,
) -> None:
    piv = df.pivot_table(
        index="config", columns=x_col, values=y_col, aggfunc="mean"
    )
    z_vals = piv.to_numpy()
    fig.add_trace(
        go.Heatmap(
            z=z_vals,
            x=[str(c) for c in piv.columns],
            y=[str(r) for r in piv.index],
            colorscale="Viridis",
            text=np.round(z_vals, 3),
            texttemplate="%{text}",
        )
    )


_BUILDERS: dict[str, Any] = {
    "Line": _build_line,
    "Bar": _build_bar,
    "Grouped Bar": _build_grouped_bar,
    "Stacked Bar": _build_stacked_bar,
    "Stacked Area": _build_stacked_area,
    "Heatmap": _build_heatmap,
}


# ── X-axis helpers (split into filter vs format) ────────────────────────────


def _filter_x_values(df: pd.DataFrame, x_cfg: dict[str, Any]) -> pd.DataFrame:
    """Step 3: Keep only rows matching x.values.  NO rename, NO categorical."""
    col: str = x_cfg["column"]
    if "values" not in x_cfg or not x_cfg["values"]:
        return df
    vals = x_cfg["values"]
    str_vals = [str(v) for v in vals]
    mask = df[col].astype(str).isin(str_vals)
    for v in vals:
        try:
            nv = float(v)
            numeric_col = pd.to_numeric(df[col], errors="coerce")
            mask = mask | (numeric_col == nv)
        except (ValueError, TypeError):
            pass
    return df[mask].copy()


def _format_x_axis(df: pd.DataFrame, x_cfg: dict[str, Any]) -> pd.DataFrame:
    """Step 5: Apply categorical ordering and rename AFTER aggregation."""
    col: str = x_cfg["column"]
    if "values" in x_cfg and x_cfg["values"]:
        vals = x_cfg["values"]
        str_vals = [str(v) for v in vals]
        # Convert to string first to ensure consistency
        df[col] = df[col].astype(str)
        df[col] = pd.Categorical(df[col], categories=str_vals, ordered=True)
    else:
        # Auto-sort numerically when all x values are numeric
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().all():
            df[col] = numeric
    if "rename" in x_cfg and x_cfg["rename"]:
        rmap: dict[str, str] = x_cfg["rename"]
        df[col] = df[col].astype(str).map(lambda v, _r=rmap: _r.get(v, v))
        if "values" in x_cfg and x_cfg["values"]:
            renamed_cats = [rmap.get(str(v), str(v)) for v in x_cfg["values"]]
            df[col] = pd.Categorical(
                df[col], categories=renamed_cats, ordered=True
            )
    return df


# ── Group helper ─────────────────────────────────────────────────────────────


def _resolve_group(
    df: pd.DataFrame, group_cfg: dict[str, Any] | None
) -> pd.DataFrame:
    if group_cfg is None:
        return df
    col: str = group_cfg["column"]
    if col not in df.columns:
        return df
    if "values" in group_cfg and group_cfg["values"]:
        vals = group_cfg["values"]
        str_vals = [str(v) for v in vals]
        mask = df[col].astype(str).isin(str_vals)
        df = df[mask].copy()
        df[col] = pd.Categorical(
            df[col].astype(str), categories=str_vals, ordered=True
        )
    if "rename" in group_cfg and group_cfg["rename"]:
        rmap: dict[str, str] = group_cfg["rename"]
        df[col] = df[col].astype(str).map(lambda v, _r=rmap: _r.get(v, v))
        if "values" in group_cfg and group_cfg["values"]:
            renamed_cats = [
                rmap.get(str(v), str(v)) for v in group_cfg["values"]
            ]
            df[col] = pd.Categorical(
                df[col], categories=renamed_cats, ordered=True
            )
    df["config"] = df[col].astype(str)
    return df


# ── Auto-dedup safety net ────────────────────────────────────────────────────


def _auto_dedup(
    df: pd.DataFrame, x_col: str, y_col: str, plot_id: str
) -> pd.DataFrame:
    """
    Step 8: If multiple rows share the same (config, x) pair, collapse them
    via mean.  Prevents zigzag lines and stacked anomalies.
    """
    if "config" not in df.columns or x_col not in df.columns:
        return df

    dup_count = df.groupby(["config", x_col], observed=True).size()
    if (dup_count > 1).any():
        max_dups = int(dup_count.max())
        st.caption(
            f"⚠️ [{plot_id}] Up to {max_dups} rows per (config, x) — "
            f'auto-averaging.  Add `"aggregate"` to spec to control this.'
        )
        # Identify all numeric columns to aggregate
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        value_cols = [c for c in numeric_cols if c != x_col]
        if value_cols:
            agg_dict: dict[str, str] = {c: "mean" for c in value_cols}
            # Keep first value for non-numeric columns
            non_numeric = [
                c
                for c in df.columns
                if c not in numeric_cols and c not in ["config", x_col]
            ]
            for c in non_numeric:
                agg_dict[c] = "first"
            df = (
                df.groupby(["config", x_col], observed=True)
                .agg(agg_dict)
                .reset_index()
            )
    return df


# ── Export / save helpers ────────────────────────────────────────────────────


def _fig_to_png_bytes(fig: go.Figure, w: int = 1200, h: int = 600) -> bytes:
    return fig.to_image(format="png", width=w, height=h, scale=2)  # type: ignore[return-value]


def _fig_to_svg_bytes(fig: go.Figure, w: int = 1200, h: int = 600) -> bytes:
    return fig.to_image(format="svg", width=w, height=h)  # type: ignore[return-value]


def _fig_to_html_str(fig: go.Figure) -> str:
    return fig.to_html(include_plotlyjs="cdn", full_html=True)  # type: ignore[return-value]


def _build_zip(
    figures: list[tuple[str, go.Figure]],
    fmt: str = "html",
    progress_callback: callable | None = None,  # type: ignore[assignment]
) -> bytes:
    """Build ZIP file with optional progress callback."""
    buf = io.BytesIO()
    total = len(figures)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, (plot_id, fig) in enumerate(figures, 1):
            safe = plot_id.replace("/", "_").replace("\\", "_")
            if fmt == "html":
                zf.writestr(f"{safe}.html", _fig_to_html_str(fig))
            elif fmt == "png":
                zf.writestr(f"{safe}.png", _fig_to_png_bytes(fig))
            elif fmt == "svg":
                zf.writestr(f"{safe}.svg", _fig_to_svg_bytes(fig))
            if progress_callback:
                progress_callback(idx / total)
    buf.seek(0)
    return buf.read()


# ── JSON auto-plot core ──────────────────────────────────────────────────────


def _apply_filters(
    df: pd.DataFrame, filters: dict[str, list[Any]]
) -> pd.DataFrame:
    for col, allowed in filters.items():
        if col not in df.columns:
            continue
        col_series = df[col]
        matched = pd.Series(False, index=df.index)
        for val in allowed:
            if isinstance(val, (int, float)):
                numeric_col = pd.to_numeric(col_series, errors="coerce")
                matched = matched | (numeric_col == val)
            matched = matched | (col_series.astype(str) == str(val))
        df = df[matched]
    return df


def _render_json_plot(
    plot_spec: dict[str, Any],
    all_dfs: dict[str, pd.DataFrame],
    combined: pd.DataFrame,
) -> go.Figure | None:
    """Render a single plot from a JSON spec.  Returns the figure (or None)."""
    plot_id: str = plot_spec.get("id", "auto")
    title: str = plot_spec.get("title", "Untitled")
    chart_type_key: str = plot_spec.get("chart_type", "grouped_bar")
    chart_type = _CHART_TYPE_MAP.get(chart_type_key, "Grouped Bar")
    builder = _BUILDERS.get(chart_type)
    if builder is None:
        st.error(f"Unknown chart_type: {chart_type_key}")
        return None

    source = plot_spec.get("source")
    df = (
        all_dfs[source].copy()
        if (source and source in all_dfs)
        else combined.copy()
    )

    # ── 1. Extract columns from text ─────────────────────────────────
    extractions = plot_spec.get("extract_columns", [])
    if extractions:
        df = _extract_regex_columns(df, extractions)

    # ── 2. Computed columns ──────────────────────────────────────────
    computed = plot_spec.get("computed_columns", [])
    if computed:
        df = _compute_columns(df, computed)

    # ── 3. Filters ───────────────────────────────────────────────────
    filters = plot_spec.get("filters", {})
    if filters:
        df = _apply_filters(df, filters)

    x_cfg: dict[str, Any] = plot_spec["x"]
    y_cfg: dict[str, Any] = plot_spec["y"]
    group_cfg: dict[str, Any] | None = plot_spec.get("group")
    agg_cfg: dict[str, Any] = plot_spec.get("aggregate", {})
    x_col: str = x_cfg["column"]
    x_label: str = x_cfg.get("label", x_col)
    y_columns: list[str] = y_cfg.get("columns", [])
    y_label: str = y_cfg.get("label", y_columns[0] if y_columns else "Value")
    y_rename: dict[str, str] = y_cfg.get("rename", {})

    if x_col not in df.columns:
        st.warning(f"[{plot_id}] Column `{x_col}` not found in CSV.")
        return None

    # Check that at least some y columns exist
    missing_y = [c for c in y_columns if c not in df.columns]
    if missing_y:
        st.warning(
            f"[{plot_id}] Column(s) `{'`, `'.join(missing_y)}` not found in CSV.  "
            f"Available: {', '.join(sorted(df.columns))}"
        )
        y_columns = [c for c in y_columns if c in df.columns]
        if not y_columns:
            return None

    # ── 4. X-values filter (rows only, no formatting) ────────────────
    df = _filter_x_values(df, x_cfg)
    if df.empty:
        st.warning(f"[{plot_id}] No data after applying filters.")
        return None

    # ── 5. Aggregation (on raw, unformatted data) ────────────────────
    agg_func: str | None = agg_cfg.get("func")
    agg_error_bars: str | None = agg_cfg.get("error_bars")
    transform: dict[str, Any] = plot_spec.get("transform", {})
    if agg_func:
        gk: list[str] = [x_col]
        if group_cfg and group_cfg.get("column") in df.columns:
            gk.append(group_cfg["column"])
        # Include normalize ref column in aggregation so it survives
        _agg_value_cols = list(y_columns)
        _norm_ref = transform.get("normalize_ref_column")
        if (
            _norm_ref
            and _norm_ref in df.columns
            and _norm_ref not in _agg_value_cols
        ):
            _agg_value_cols.append(_norm_ref)
        df = _aggregate_data(
            df,
            group_keys=gk,
            value_cols=_agg_value_cols,
            func=agg_func,
            error_bars=agg_error_bars,
        )

    # ── 6. X-axis formatting (categorical + rename, AFTER agg) ───────
    df = _format_x_axis(df, x_cfg)

    # ── 7. Normalize transform ───────────────────────────────────────
    if transform.get("normalize"):
        df = _normalize_stacked(
            df, transform.get("normalize_columns", y_columns)
        )
    if transform.get("normalize_distribution"):
        df = _normalize_distribution(
            df,
            transform.get("normalize_columns", y_columns),
            ref_column=transform.get("normalize_ref_column"),
        )

    # ── 8. Resolve traces ────────────────────────────────────────────
    multi_y = len(y_columns) > 1
    has_group = group_cfg is not None

    if multi_y and not has_group:
        rows: list[pd.DataFrame] = []
        for yc in y_columns:
            if yc not in df.columns:
                continue
            tmp = df[[x_col, yc]].copy()
            for suf in ("__err", "__err_plus", "__err_minus"):
                ec = f"{yc}{suf}"
                if ec in df.columns:
                    tmp[f"__y_value__{suf}"] = df[ec]
            tmp = tmp.rename(columns={yc: "__y_value__"})
            tmp["config"] = y_rename.get(yc, yc)
            rows.append(tmp)
        if not rows:
            st.warning(f"[{plot_id}] None of {y_columns} found in CSV.")
            return None
        plot_df = pd.concat(rows, ignore_index=True)
        y_plot_col = "__y_value__"
    elif multi_y and has_group:
        df = _resolve_group(df, group_cfg)
        rows = []
        for yc in y_columns:
            if yc not in df.columns:
                continue
            tmp = df[[x_col, yc, "config"]].copy()
            tmp = tmp.rename(columns={yc: "__y_value__"})
            tmp["config"] = tmp["config"] + " — " + y_rename.get(yc, yc)
            rows.append(tmp)
        if not rows:
            st.warning(f"[{plot_id}] None of {y_columns} found in CSV.")
            return None
        plot_df = pd.concat(rows, ignore_index=True)
        y_plot_col = "__y_value__"
    elif has_group:
        df = _resolve_group(df, group_cfg)
        y_plot_col = y_columns[0] if y_columns else "value"
        if y_plot_col not in df.columns:
            st.warning(f"[{plot_id}] Column `{y_plot_col}` not found.")
            return None
        plot_df = df
    else:
        y_plot_col = y_columns[0] if y_columns else "value"
        if y_plot_col not in df.columns:
            st.warning(f"[{plot_id}] Column `{y_plot_col}` not found.")
            return None
        if "config" not in df.columns:
            df["config"] = "all"
        plot_df = df

    # ── 9. Auto-dedup safety net ─────────────────────────────────────
    plot_df = _auto_dedup(plot_df, x_col, y_plot_col, plot_id)

    # ── 10. Render ───────────────────────────────────────────────────
    style_line_dash: str | dict[str, str] | None = plot_spec.get("line_dash")
    style_opacity: float | None = plot_spec.get("opacity")

    fig = go.Figure()
    builder(
        fig,
        plot_df,
        x_col,
        y_plot_col,
        line_dash=style_line_dash,
        opacity=style_opacity,
    )

    layout_kw: dict[str, Any] = {
        "title": title,
        "xaxis_title": x_label,
        "yaxis_title": y_label,
        "template": "plotly_white",
        "height": 600,
    }
    x_scale = plot_spec.get("x_scale")
    y_scale = plot_spec.get("y_scale")
    if x_scale:
        layout_kw["xaxis_type"] = x_scale
    if y_scale:
        layout_kw["yaxis_type"] = y_scale
    layout_kw.update(plot_spec.get("layout", {}))
    fig.update_layout(**layout_kw)

    render_custom_plotly_chart(fig, width="stretch", key=f"json_{plot_id}")

    note = plot_spec.get("note")
    if note:
        st.caption(f"ℹ️ {note}")
    if plot_spec.get("show_table", False):
        with st.expander(f"📊 Data table — {plot_id}"):
            st.dataframe(plot_df, width="stretch")

    return fig


# ── Manual mode ──────────────────────────────────────────────────────────────


def _run_manual_mode(combined: pd.DataFrame) -> None:
    # ── Column extraction from text ────────────────────────────────
    st.sidebar.header("Column Extraction")
    enable_extract = st.sidebar.checkbox(
        "Extract columns via regex", key="enable_extract"
    )
    if enable_extract:
        all_source_cols = list(combined.columns)
        default_src_idx = (
            all_source_cols.index("config")
            if "config" in all_source_cols
            else 0
        )
        extract_source: str = st.sidebar.selectbox(
            "Source column",
            options=all_source_cols,
            index=default_src_idx,
            key="extract_source",
        )
        extract_mode = st.sidebar.radio(
            "Extraction mode",
            ["Auto (Key:Value pairs)", "Custom regex"],
            key="extract_mode",
        )
        if extract_mode == "Auto (Key:Value pairs)":
            ec1, ec2 = st.sidebar.columns(2)
            sep = ec1.text_input(
                "Pair separator", value="|", key="extract_sep"
            )
            kv_sep = ec2.text_input(
                "Key:Value sep", value=":", key="extract_kv_sep"
            )
            extractions = [
                {
                    "source": extract_source,
                    "auto_kv": True,
                    "separator": sep,
                    "kv_separator": kv_sep,
                }
            ]
        else:
            pattern = st.sidebar.text_input(
                "Regex with named groups `(?P<Name>...)`",
                value=r"SNR:(?P<SNR>[^|]+)",
                key="extract_pattern",
            )
            extractions = [{"source": extract_source, "pattern": pattern}]

        pre_cols = set(combined.columns)
        combined = _extract_regex_columns(combined, extractions)
        new_cols = [c for c in combined.columns if c not in pre_cols]
        if new_cols:
            st.sidebar.success(f"Extracted: {', '.join(new_cols)}")
        else:
            st.sidebar.info("No new columns extracted.")

    # ── Group / trace column ───────────────────────────────────────
    st.sidebar.header("Group / Trace Column")
    all_available = list(combined.columns)
    default_group_idx = (
        all_available.index("config") if "config" in all_available else 0
    )
    group_col: str = st.sidebar.selectbox(
        "Group traces by",
        options=all_available,
        index=default_group_idx,
        key="group_col",
    )
    if group_col != "config":
        combined["config"] = combined[group_col].astype(str)

    # ── Config renaming ────────────────────────────────────────────
    original_configs = list(combined["config"].unique())

    st.sidebar.header("Config Renaming")
    rename_map: dict[str, str] = {}
    for cfg in original_configs:
        new_name = st.sidebar.text_input(
            f"Rename: {cfg}", value=cfg, key=f"rename_{cfg}"
        )
        rename_map[cfg] = new_name if new_name is not None else cfg
    combined["config"] = combined["config"].map(rename_map)

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

    selected_metrics: list[str] = st.sidebar.multiselect(
        "Y-Axis metric(s)", options=metric_options, default=metric_options[:1]
    )
    if not selected_metrics:
        st.info("Select at least one metric.")
        st.stop()

    st.sidebar.header("Aggregation")
    agg_func = st.sidebar.selectbox(
        "Aggregate function",
        options=[
            "none",
            "mean",
            "median",
            "sum",
            "count",
            "min",
            "max",
            "std",
        ],
    )
    agg_errors = st.sidebar.selectbox(
        "Error bars", options=["none", "std", "sem", "minmax"]
    )

    st.sidebar.header("Axis Labels")
    x_label = (
        st.sidebar.text_input("X-Axis label", value=x_col, key="x_axis_label")
        or x_col
    )
    y_label = (
        st.sidebar.text_input(
            "Y-Axis label",
            value=selected_metrics[0]
            if len(selected_metrics) == 1
            else "Value",
            key="y_axis_label",
        )
        or "Value"
    )

    st.sidebar.header("Axis Scale")
    _scale_opts = ["linear", "log"]
    x_scale_type: str = st.sidebar.selectbox(
        "X-Axis scale", options=_scale_opts, index=0, key="x_scale_type"
    )  # type: ignore[assignment]
    y_scale_type: str = st.sidebar.selectbox(
        "Y-Axis scale", options=_scale_opts, index=0, key="y_scale_type"
    )  # type: ignore[assignment]

    st.sidebar.header("Config Filter")
    all_configs = list(combined["config"].unique())
    selected_configs: list[str] = st.sidebar.multiselect(
        "Show configs",
        options=all_configs,
        default=all_configs,
    )
    if not selected_configs:
        st.info("Select at least one config to display.")
        st.stop()

    plot_df = combined[combined["config"].isin(selected_configs)].copy()
    if agg_func != "none":
        plot_df = _aggregate_data(
            plot_df,
            group_keys=[x_col, "config"],
            value_cols=selected_metrics,
            func=agg_func,
            error_bars=agg_errors if agg_errors != "none" else None,
        )

    # Auto-sort x-axis numerically when all values are numeric
    numeric_x = pd.to_numeric(plot_df[x_col], errors="coerce")
    if numeric_x.notna().all():
        plot_df[x_col] = numeric_x

    st.sidebar.header("Chart Type")
    chart_type: str = st.sidebar.selectbox(
        "Visualization", options=CHART_TYPES
    )  # type: ignore[assignment]

    st.sidebar.header("Style")
    manual_line_dash: str = st.sidebar.selectbox(
        "Line style",
        options=_DASH_STYLES,
        index=0,
        key="manual_line_dash",
    )  # type: ignore[assignment]
    manual_opacity: float = st.sidebar.slider(
        "Opacity",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        key="manual_opacity",
    )

    style_kw: dict[str, Any] = {}
    if manual_line_dash != "solid":
        style_kw["line_dash"] = manual_line_dash
    if manual_opacity < 1.0:
        style_kw["opacity"] = manual_opacity

    for idx, metric in enumerate(selected_metrics):
        fig = go.Figure()
        _BUILDERS[chart_type](fig, plot_df, x_col, metric, **style_kw)
        scale_kw: dict[str, Any] = {}
        if x_scale_type != "linear":
            scale_kw["xaxis_type"] = x_scale_type
        if y_scale_type != "linear":
            scale_kw["yaxis_type"] = y_scale_type
        fig.update_layout(
            title=f"{metric} vs {x_label}",
            xaxis_title=x_label,
            yaxis_title=y_label,
            template="plotly_white",
            height=600,
            **scale_kw,
        )
        render_custom_plotly_chart(fig, width="stretch", key=f"csv_plot_{idx}")

    st.subheader("Data Table")
    st.dataframe(plot_df, width="stretch")
    st.download_button(
        "Download combined CSV",
        plot_df.to_csv(index=False),
        "combined_results.csv",
        "text/csv",
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="CSV Plot Editor", layout="wide")
    st.title("CSV Plot Editor v3")
    st.markdown(
        "Upload CSV files exported from **Parameter Sweep Analyzer**, "
        "rename configs, pick a chart type, and render.  \n"
        "Upload one or more **JSON plot spec** files to auto-generate all "
        "plots ([format docs](csv_plot_editor_format.md))."
    )

    # 1. Upload CSVs
    uploaded_csvs = st.file_uploader(
        "Upload one or more sweep-result CSVs",
        type=["csv"],
        accept_multiple_files=True,
        key="csv_uploader",
    )

    # 2. Upload JSON spec(s)
    json_files = st.file_uploader(
        "Upload JSON plot spec(s) (optional — enables auto-plot mode)",
        type=["json"],
        accept_multiple_files=True,
        key="json_uploader",
    )

    if not uploaded_csvs:
        st.info("Upload at least one CSV to get started.")
        st.stop()

    # 3. Read & combine CSVs
    frames: list[pd.DataFrame] = []
    named_dfs: dict[str, pd.DataFrame] = {}
    for uf in uploaded_csvs:
        try:
            df = pd.read_csv(uf)
        except Exception as exc:
            st.error(f"Failed to read **{uf.name}**: {exc}")
            continue
        if "config" not in df.columns:
            st.warning(
                f"**{uf.name}** has no `config` column – adding filename."
            )
            df["config"] = uf.name.removesuffix(".csv")
        named_dfs[uf.name] = df
        frames.append(df)

    if not frames:
        st.error("No valid CSVs loaded.")
        st.stop()

    combined = pd.concat(frames, ignore_index=True)

    # 4. Route
    if json_files:
        all_plots: list[dict[str, Any]] = []
        for jf in json_files:
            try:
                spec = json.load(jf)
            except Exception as exc:
                st.error(f"Failed to parse **{jf.name}**: {exc}")
                continue
            plots_in = spec.get("plots", [])
            all_plots.extend(plots_in)
            st.info(f"📄 **{jf.name}**: {len(plots_in)} plot(s)")

        if not all_plots:
            st.warning("JSON spec(s) contain no `plots` entries.")
            st.stop()

        st.success(
            f"Auto-plot mode: rendering **{len(all_plots)}** plot(s) "
            f"from {len(json_files)} JSON file(s)."
        )

        rendered: list[tuple[str, go.Figure]] = []
        for ps in all_plots:
            st.divider()
            fig = _render_json_plot(ps, named_dfs, combined)
            if fig is not None:
                rendered.append((ps.get("id", f"plot_{len(rendered)}"), fig))

        # Save all
        if rendered:
            st.divider()
            st.subheader("💾 Save All Plots")

            # Store rendered plots in session state
            _RENDERED_PLOTS_KEY = "csv_plot_editor_rendered_plots"
            st.session_state[_RENDERED_PLOTS_KEY] = rendered

            # Progress placeholder
            progress_placeholder = st.empty()

            # Helper function to build ZIP with progress
            def _build_and_store(fmt: str, key_suffix: str):
                """Build ZIP with progress and store in session state."""
                figures = st.session_state.get(_RENDERED_PLOTS_KEY, rendered)
                with progress_placeholder.container():
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                try:

                    def update_progress(progress):
                        progress_bar.progress(progress)
                        progress_text.text(
                            f"Building {fmt.upper()}... {int(progress * 100)}%"
                        )

                    zip_data = _build_zip(
                        figures, fmt, progress_callback=update_progress
                    )
                    progress_bar.progress(1.0)
                    progress_text.text(f"✓ {fmt.upper()} ready!")
                    st.session_state[f"zip_data_{key_suffix}"] = zip_data
                    st.session_state[f"zip_ready_{key_suffix}"] = True
                finally:
                    time.sleep(0.5)
                    progress_placeholder.empty()

            # Build callbacks
            def _build_html():
                _build_and_store("html", "html")

            def _build_png():
                _build_and_store("png", "png")

            def _build_svg():
                _build_and_store("svg", "svg")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.button(
                    label=f"🏗️ Build HTML ({len(rendered)})",
                    on_click=_build_html,
                    key="build_html",
                )
                if st.session_state.get("zip_ready_html", False):
                    st.download_button(
                        label="📥 Download HTML",
                        data=st.session_state["zip_data_html"],
                        file_name="plots_html.zip",
                        mime="application/zip",
                        key="dl_all_html",
                    )
            with c2:
                try:
                    st.button(
                        label=f"🏗️ Build PNG ({len(rendered)})",
                        on_click=_build_png,
                        key="build_png",
                    )
                    if st.session_state.get("zip_ready_png", False):
                        st.download_button(
                            label="📥 Download PNG",
                            data=st.session_state["zip_data_png"],
                            file_name="plots_png.zip",
                            mime="application/zip",
                            key="dl_all_png",
                        )
                except Exception:
                    st.caption(
                        "PNG needs `kaleido`: `uv add 'kaleido>=1.0.0'`"
                    )
            with c3:
                try:
                    st.button(
                        label=f"🏗️ Build SVG ({len(rendered)})",
                        on_click=_build_svg,
                        key="build_svg",
                    )
                    if st.session_state.get("zip_ready_svg", False):
                        st.download_button(
                            label="📥 Download SVG",
                            data=st.session_state["zip_data_svg"],
                            file_name="plots_svg.zip",
                            mime="application/zip",
                            key="dl_all_svg",
                        )
                except Exception:
                    st.caption(
                        "SVG needs `kaleido`: `uv add 'kaleido>=1.0.0'`"
                    )

            with st.expander("Download individual plots"):
                for pid, fig in rendered:
                    ic1, ic2, ic3 = st.columns([3, 1, 1])
                    ic1.write(f"**{pid}**")
                    ic2.download_button(
                        "HTML",
                        data=lambda f=fig: _fig_to_html_str(f),
                        file_name=f"{pid}.html",
                        mime="text/html",
                        key=f"dl_i_html_{pid}",
                    )
                    try:
                        ic3.download_button(
                            "PNG",
                            data=lambda f=fig: _fig_to_png_bytes(f),
                            file_name=f"{pid}.png",
                            mime="image/png",
                            key=f"dl_i_png_{pid}",
                        )
                    except Exception:
                        ic3.caption("kaleido")
    else:
        _run_manual_mode(combined)


if __name__ == "__main__":
    main()
