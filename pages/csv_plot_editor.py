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
import math
import re
import time
import zipfile
from typing import Any, cast

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
    from utils import (
        # The real helper takes extra keyword arguments the fallback below
        # does not model; the fallback is only for running this page stand-alone.
        file_input,  # type: ignore[assignment]
        render_custom_plotly_chart,
    )
except ImportError:

    def render_custom_plotly_chart(
        fig: go.Figure, width: Any = "stretch", key: str | None = None
    ) -> None:
        st.plotly_chart(fig, width=width)

    def file_input(label: str, **kwargs):  # type: ignore[misc]
        kwargs.pop("default_dir", None)
        kwargs.pop("default_source", None)
        kwargs.pop("container", None)
        return st.file_uploader(label, **kwargs)


# ── Typing helpers ───────────────────────────────────────────────────────────
# pandas-stubs types ``frame[key]`` and ``pd.to_numeric`` as unions covering
# every overload (scalar, sub-frame, ExtensionArray...). Every call site below
# passes a single column name or a boolean mask, so narrow it once here rather
# than sprinkling ``# type: ignore`` over the module.


def _series(values: Any) -> pd.Series:
    """Narrow a single-column indexing result to a Series."""
    return cast("pd.Series", values)


def _frame(values: Any) -> pd.DataFrame:
    """Narrow a boolean-mask selection back to a DataFrame."""
    return cast("pd.DataFrame", values)


def _numeric(values: Any) -> pd.Series:
    """``pd.to_numeric(..., errors="coerce")`` narrowed to a Series."""
    return _series(pd.to_numeric(values, errors="coerce"))


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
                    _series(
                        df.groupby(existing_keys, observed=True)[col].min()
                    )
                    .reset_index()
                    .rename(columns={col: f"{col}__err_minus"})
                )
                mx = (
                    _series(
                        df.groupby(existing_keys, observed=True)[col].max()
                    )
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


def _finite_event_fixed_moments(
    k: int, common_fraction: float
) -> tuple[float, float]:
    """Expected correct intervals and compared slots for ``k`` merged events."""
    if k <= 0:
        return 0.0, 0.0
    c = float(np.clip(common_fraction, 0.0, 1.0))
    a = 0.5 * (1.0 - c)
    correct = 0.0
    if k >= 2:
        for offset_steps in range(k - 1):
            return_zero = 0.0
            for r in range(offset_steps // 2 + 1):
                return_zero += (
                    math.factorial(offset_steps)
                    / (
                        math.factorial(r) ** 2
                        * math.factorial(offset_steps - 2 * r)
                    )
                    * (a * a) ** r
                    * c ** (offset_steps - 2 * r)
                )
            correct += c * c * return_zero

    slots = 0.0
    for n_common in range(k + 1):
        for n_a_only in range(k - n_common + 1):
            n_b_only = k - n_common - n_a_only
            probability = (
                math.factorial(k)
                / (
                    math.factorial(n_common)
                    * math.factorial(n_a_only)
                    * math.factorial(n_b_only)
                )
                * c**n_common
                * a ** (n_a_only + n_b_only)
            )
            slots += probability * max(
                n_common + n_a_only - 1,
                n_common + n_b_only - 1,
                0,
            )
    return float(correct), float(slots)


def _compute_finite_event_model(
    df: pd.DataFrame, spec: dict[str, Any]
) -> pd.DataFrame:
    """Add finite merged-event BDR and optional joint-entropy output columns."""
    n_column = spec.get(
        "recognized_column", "Average Number of Recognized Rays"
    )
    pair_column = spec.get("pair_column", "Pair Percentage")
    if n_column not in df.columns or pair_column not in df.columns:
        raise KeyError(
            f"Required columns are missing: {n_column!r}, {pair_column!r}"
        )

    n_values = _series(_numeric(df[n_column]).fillna(0.0)).to_numpy()
    common_values = (
        _series(_numeric(df[pair_column]).fillna(0.0))
        .clip(0.0, 100.0)
        .to_numpy()
        / 100.0
    )
    union_values = np.divide(
        2.0 * n_values,
        1.0 + common_values,
        out=np.zeros_like(n_values, dtype=float),
        where=(1.0 + common_values) > 0,
    )
    expected_correct = np.zeros(len(df), dtype=float)
    expected_slots = np.zeros(len(df), dtype=float)
    for idx, (union_mean, common_fraction) in enumerate(
        zip(union_values, common_values, strict=True)
    ):
        lower = max(0, math.floor(union_mean))
        upper = lower + 1
        alpha = union_mean - lower
        c0, l0 = _finite_event_fixed_moments(lower, common_fraction)
        c1, l1 = _finite_event_fixed_moments(upper, common_fraction)
        expected_correct[idx] = (1.0 - alpha) * c0 + alpha * c1
        expected_slots[idx] = (1.0 - alpha) * l0 + alpha * l1

    correct_fraction = np.divide(
        expected_correct,
        expected_slots,
        out=np.zeros_like(expected_correct),
        where=expected_slots > 0,
    )
    df["n_components_cir"] = n_values
    df["c_common"] = common_values
    df["merged_events"] = union_values
    df["expected_correct_intervals"] = expected_correct
    df["expected_compared_slots"] = expected_slots
    df["p_correct_finite"] = correct_fraction
    df["bdr_finite_model"] = 50.0 * (1.0 - correct_fraction)
    # Pure positional-agreement percentage.  It must not by itself be called
    # the realized entropy potential because the recognized/true component
    # count ratio cancels from that normalization.
    df["positional_agreement_percent"] = 100.0 * correct_fraction

    entropy_order = np.maximum(n_values - 1.0, 0.0)
    joint_entropy = spec.get("joint_entropy")
    joint_entropy_by = spec.get("joint_entropy_by")
    entropy_values: np.ndarray | None = None
    if joint_entropy:
        table = np.asarray(joint_entropy, dtype=float)
        entropy_values = np.interp(
            entropy_order,
            np.arange(len(table), dtype=float),
            table,
        )
    elif joint_entropy_by:
        group_column = joint_entropy_by.get("column")
        tables = joint_entropy_by.get("values", {})
        if group_column not in df.columns:
            raise KeyError(
                f"Joint-entropy group column is missing: {group_column!r}"
            )
        entropy_values = np.zeros(len(df), dtype=float)
        groups = df[group_column].astype(str).to_numpy()
        for group_name, raw_table in tables.items():
            table = np.asarray(raw_table, dtype=float)
            mask = groups == str(group_name)
            entropy_values[mask] = np.interp(
                entropy_order[mask],
                np.arange(len(table), dtype=float),
                table,
            )
    if entropy_values is not None:
        df["joint_entropy_observable"] = entropy_values
        df["entropy_output_finite"] = correct_fraction * entropy_values

        # Estimate true components from the deepest saved Top-N row of every
        # configuration. Category percentages use the true-ray count as their
        # denominator, hence N_true ~= 100*N_rec/(P_TP+P_FP). Rays classified
        # as Lost (Time Res) retain energy in a merged peak but cannot provide
        # an independent temporal interval.
        depth_column = spec.get("depth_column", "Top-N Peaks")
        reference_group = spec.get("reference_group_column", "config")
        right_column = spec.get("right_column", "Percent Right (TP)")
        false_column = spec.get("false_column", "Percent False (FP)")
        time_res_column = spec.get(
            "time_res_column", "Percent Lost (Time Res)"
        )
        required_reference = [
            depth_column,
            reference_group,
            right_column,
            false_column,
            time_res_column,
        ]
        if all(column in df.columns for column in required_reference):
            depth = _numeric(df[depth_column])
            reference_indices = depth.groupby(
                _series(df[reference_group])
            ).idxmax()
            reference_rows = df.loc[reference_indices].copy()
            reference_rows["_n_reference"] = pd.to_numeric(
                reference_rows[n_column], errors="coerce"
            )
            reference_rows["_right_reference"] = pd.to_numeric(
                reference_rows[right_column], errors="coerce"
            )
            reference_rows["_false_reference"] = pd.to_numeric(
                reference_rows[false_column], errors="coerce"
            )
            reference_rows["_time_res_reference"] = pd.to_numeric(
                reference_rows[time_res_column], errors="coerce"
            )
            reference_rows = reference_rows.set_index(reference_group)
            group_values = df[reference_group]
            ref_n = group_values.map(reference_rows["_n_reference"]).to_numpy()
            ref_detected_share = group_values.map(
                reference_rows["_right_reference"]
                + reference_rows["_false_reference"]
            ).to_numpy()
            ref_time_res = group_values.map(
                reference_rows["_time_res_reference"]
            ).to_numpy()
            estimated_true = np.divide(
                100.0 * ref_n,
                ref_detected_share,
                out=np.zeros_like(ref_n, dtype=float),
                where=ref_detected_share > 0,
            )
            estimated_resolvable = estimated_true * (
                1.0 - np.clip(ref_time_res, 0.0, 100.0) / 100.0
            )
            potential_order = np.maximum(estimated_resolvable - 1.0, 0.0)

            potential_entropy = np.zeros(len(df), dtype=float)
            if joint_entropy:
                potential_table = np.asarray(joint_entropy, dtype=float)
                potential_entropy = np.interp(
                    potential_order,
                    np.arange(len(potential_table), dtype=float),
                    potential_table,
                )
                max_measured_order = float(len(potential_table) - 1)
            else:
                # ``entropy_values`` can only be populated above by one of
                # these two specs. Help the type checker retain that invariant
                # across the intervening dataframe code.
                assert joint_entropy_by is not None
                group_column = joint_entropy_by.get("column")
                tables = joint_entropy_by.get("values", {})
                groups = df[group_column].astype(str).to_numpy()
                max_orders: list[int] = []
                for group_name, raw_table in tables.items():
                    potential_table = np.asarray(raw_table, dtype=float)
                    mask = groups == str(group_name)
                    potential_entropy[mask] = np.interp(
                        potential_order[mask],
                        np.arange(len(potential_table), dtype=float),
                        potential_table,
                    )
                    max_orders.append(len(potential_table) - 1)
                max_measured_order = (
                    float(min(max_orders)) if max_orders else 0.0
                )

            realized_percent = np.divide(
                100.0 * correct_fraction * entropy_values,
                potential_entropy,
                out=np.zeros_like(entropy_values, dtype=float),
                where=potential_entropy > 0,
            )
            df["estimated_true_components"] = estimated_true
            df["estimated_time_resolvable_components"] = estimated_resolvable
            df["potential_entropy_order"] = potential_order
            df["potential_entropy_order_used"] = np.minimum(
                potential_order, max_measured_order
            )
            df["potential_joint_entropy"] = potential_entropy
            df["entropy_potential_realized_percent"] = np.clip(
                realized_percent, 0.0, 100.0
            )
    return df


def _compute_time_resolution_energy(
    df: pd.DataFrame, spec: dict[str, Any]
) -> pd.DataFrame:
    """Add energy completeness with unresolved-ray power assigned to TP peaks.

    ``Lost (Time Res)`` denotes a true ray close enough to a detected peak to
    be unresolved.  Its power is therefore represented by that merged peak
    and should not be counted as missing energy.  The existing TP percentage
    supplies the total-signal normalization, so no source CSV recalculation is
    required.
    """
    percent_column = spec.get(
        "percent_column", "Percent of Total Signal Power in TP"
    )
    tp_power_column = spec.get("tp_power_column", "TP Power in Bin")
    merged_power_column = spec.get(
        "merged_power_column", "Lost (Time Res) Power in Bin"
    )
    required = [percent_column, tp_power_column, merged_power_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Required energy columns are missing: {missing!r}")

    tp_percent = _numeric(df[percent_column]).to_numpy()
    tp_power = _numeric(df[tp_power_column]).to_numpy()
    merged_power = _numeric(df[merged_power_column]).to_numpy()
    corrected = np.divide(
        tp_percent * (tp_power + merged_power),
        tp_power,
        out=np.full_like(tp_percent, np.nan, dtype=float),
        where=tp_power > 0,
    )
    df["energy_completeness_time_res_adjusted"] = np.clip(
        corrected, 0.0, 100.0
    )
    return df


def _compute_columns(
    df: pd.DataFrame, computed: list[dict[str, Any]]
) -> pd.DataFrame:
    """Add computed columns via ``pd.eval``."""
    for spec in computed:
        if spec.get("model") == "finite_event_positional":
            try:
                df = _compute_finite_event_model(df, spec)
            except Exception as exc:
                st.warning(f"Finite-event positional model failed: {exc}")
            continue
        if spec.get("model") == "time_resolution_energy":
            try:
                df = _compute_time_resolution_energy(df, spec)
            except Exception as exc:
                st.warning(f"Time-resolution energy correction failed: {exc}")
            continue
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
                    numeric = _numeric(df[col])
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
    group_column: str | None = None,
) -> pd.DataFrame:
    """Normalize Y columns as a per-row distribution (%).

    When *ref_column* is provided, each row's Y values are divided by that
    row's *ref_column* value (per-bin normalization).  Otherwise each column
    is divided by its own sum independently.

    When *group_column* is provided, normalization is performed independently
    within each group so that grouped plots are comparable.
    """
    existing = [c for c in y_columns if c in df.columns]
    if not existing:
        return df
    df = df.copy()

    if group_column and group_column in df.columns:
        for idx in df.groupby(group_column).groups.values():
            if ref_column and ref_column in df.columns:
                total = float(df.loc[idx, ref_column].sum())
                if total > 0:
                    for c in existing:
                        df.loc[idx, c] = df.loc[idx, c] / total * 100.0
            else:
                for c in existing:
                    col_sum = float(df.loc[idx, c].sum())
                    if col_sum > 0:
                        df.loc[idx, c] = df.loc[idx, c] / col_sum * 100.0
    elif ref_column and ref_column in df.columns:
        total = float(_series(df[ref_column]).sum())
        if total > 0:
            for c in existing:
                df[c] = df[c] / total * 100.0
    else:
        for c in existing:
            col_sum = float(_series(df[c]).sum())
            if col_sum > 0:
                df[c] = df[c] / col_sum * 100.0
    return df


def _normalize_group_peak(
    df: pd.DataFrame,
    y_columns: list[str],
    group_column: str | None = None,
) -> pd.DataFrame:
    """Scale each group so its peak Y value equals 100.

    This makes groups with slightly different absolute magnitudes (e.g. due to
    noise realisation differences) directly comparable by shape.
    """
    existing = [c for c in y_columns if c in df.columns]
    if not existing:
        return df
    df = df.copy()
    if group_column and group_column in df.columns:
        for idx in df.groupby(group_column).groups.values():
            peak = df.loc[idx, existing].max().max()
            if peak > 0:
                for c in existing:
                    df.loc[idx, c] = df.loc[idx, c] / peak * 100.0
    else:
        peak = df[existing].max().max()
        if peak > 0:
            for c in existing:
                df[c] = df[c] / peak * 100.0
    return df


def _apply_group_diff(
    df: pd.DataFrame,
    y_columns: list[str],
    x_col: str,
    group_column: str | None = None,
    scale: float = 1.0,
    drop_first: bool = False,
) -> pd.DataFrame:
    """Replace each Y column by its discrete difference along sorted *x_col*.

    Computed independently within each group (sorted by x).  The first x-point
    of every group keeps its original (undifferenced) value as the baseline.
    Useful to turn a cumulative curve (e.g. mean recognised components vs N)
    into its per-step increment (e.g. share of CIRs that gain an N-th
    component).  ``scale`` multiplies the result (use ``100`` for percent).
    """
    existing = [c for c in y_columns if c in df.columns]
    if not existing:
        return df
    df = df.copy()

    if group_column and group_column in df.columns:
        groups = list(df.groupby(group_column, observed=True).groups.values())
    else:
        groups = [df.index]

    first_rows: list[Any] = []
    for idx in groups:
        # Order this group's rows by x, then take the discrete difference,
        # keeping the first point as its original (baseline) value.
        order = df.loc[idx].sort_values(x_col).index
        if len(order):
            first_rows.append(order[0])
        for c in existing:
            vals = df.loc[order, c]
            d = vals.diff()
            if len(d):
                d.iloc[0] = vals.iloc[0]
            df.loc[order, c] = d.to_numpy() * scale
    if drop_first and first_rows:
        # The baseline point is only needed to difference the second point;
        # drop it so the curve shows true increments (e.g. a proper ≤100% share).
        df = df.drop(index=first_rows)
    return df


# ── Chart builders ───────────────────────────────────────────────────────────

CHART_TYPES: list[str] = [
    "Line",
    "Bar",
    "Grouped Bar",
    "Stacked Bar",
    "Stacked Area",
    "Heatmap",
    "3D Surface",
]

_CHART_TYPE_MAP: dict[str, str] = {
    "line": "Line",
    "bar": "Bar",
    "grouped_bar": "Grouped Bar",
    "stacked_bar": "Stacked Bar",
    "stacked_area": "Stacked Area",
    "heatmap": "Heatmap",
    "surface3d": "3D Surface",
    "surface_3d": "3D Surface",
}


def _get_error_arrays(df: pd.DataFrame, y_col: str) -> dict[str, Any] | None:
    sym_err = f"{y_col}__err"
    asym_plus = f"{y_col}__err_plus"
    asym_minus = f"{y_col}__err_minus"
    if sym_err in df.columns:
        return {"type": "data", "array": df[sym_err].tolist(), "visible": True}
    if asym_plus in df.columns and asym_minus in df.columns:
        return {
            "type": "data",
            "symmetric": False,
            "array": df[asym_plus].tolist(),
            "arrayminus": df[asym_minus].tolist(),
            "visible": True,
        }
    return None


_DEFAULT_COLORWAY: list[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _hex_to_rgba(color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` (or an ``rgb(...)`` string) to an ``rgba(...)`` string."""
    c = color.strip()
    if c.startswith("#") and len(c) == 7:
        r, g, b = (int(c[i : i + 2], 16) for i in (1, 3, 5))
        return f"rgba({r}, {g}, {b}, {alpha})"
    if c.startswith("rgb(") and c.endswith(")"):
        return f"rgba({c[4:-1]}, {alpha})"
    if c.startswith("rgba("):
        return c
    # Fallback: a neutral translucent grey
    return f"rgba(127, 127, 127, {alpha})"


def _err_bounds(df: pd.DataFrame, y_col: str) -> tuple[Any, Any] | None:
    """Return (upper, lower) Series for a confidence band, or None if no error data."""
    y = df[y_col]
    sym = f"{y_col}__err"
    plus = f"{y_col}__err_plus"
    minus = f"{y_col}__err_minus"
    if sym in df.columns:
        return y + df[sym], y - df[sym]
    if plus in df.columns and minus in df.columns:
        return y + df[plus], y - df[minus]
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
    error_band: bool = False,
    colorway: list[str] | None = None,
) -> None:
    palette = colorway or _DEFAULT_COLORWAY
    for idx, cfg in enumerate(df["config"].unique()):
        sub = _frame(df[df["config"] == cfg]).sort_values(x_col)
        dash = _resolve_dash(cfg, idx, line_dash)
        color = palette[idx % len(palette)]

        bounds = _err_bounds(sub, y_col) if error_band else None
        if bounds is not None:
            upper, lower = bounds
            # Upper edge (invisible), then lower edge filled up to it.
            fig.add_trace(
                go.Scatter(
                    x=sub[x_col],
                    y=upper,
                    mode="lines",
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{cfg} +",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=sub[x_col],
                    y=lower,
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=_hex_to_rgba(color, 0.18),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{cfg} -",
                )
            )

        trace_kw: dict[str, Any] = {
            "x": sub[x_col],
            "y": sub[y_col],
            "mode": "lines+markers",
            "name": str(cfg),
            "line": {"dash": dash},
        }
        if error_band:
            # Band already shows the spread; keep the line clean.
            trace_kw["line"]["color"] = color
        else:
            trace_kw["error_y"] = _get_error_arrays(sub, y_col)
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
        sub = _frame(df[df["config"] == cfg]).sort_values(x_col)
        trace_kw: dict[str, Any] = {
            "x": sub[x_col],
            "y": sub[y_col],
            "name": str(cfg),
            "error_y": _get_error_arrays(sub, y_col),
        }
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
        sub = _frame(df[df["config"] == cfg]).sort_values(x_col)
        trace_kw: dict[str, Any] = {
            "x": sub[x_col],
            "y": sub[y_col],
            "name": str(cfg),
            "error_y": _get_error_arrays(sub, y_col),
        }
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
        sub = _frame(df[df["config"] == cfg]).sort_values(x_col)
        trace_kw: dict[str, Any] = {
            "x": sub[x_col],
            "y": sub[y_col],
            "name": str(cfg),
        }
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
        sub = _frame(df[df["config"] == cfg]).sort_values(x_col)
        dash = _resolve_dash(cfg, idx, line_dash)
        trace_kw: dict[str, Any] = {
            "x": sub[x_col],
            "y": sub[y_col],
            "mode": "lines",
            "stackgroup": "one",
            "name": str(cfg),
            "line": {"dash": dash},
        }
        if opacity is not None:
            trace_kw["opacity"] = opacity
        fig.add_trace(go.Scatter(**trace_kw))


def _natural_sort_key(value: Any) -> tuple[int, float, str]:
    """Sort key that orders by an embedded number when present, else by text.

    Keeps ``N = 2`` before ``N = 10`` and ``-10`` before ``-5`` rather than the
    lexicographic order ``pivot_table`` would otherwise impose.
    """
    s = str(value)
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace("−", "-"))
    if m:
        return (0, float(m.group()), s)
    return (1, 0.0, s)


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
    piv = piv.reindex(
        index=sorted(piv.index, key=_natural_sort_key),
        columns=sorted(piv.columns, key=_natural_sort_key),
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


def _build_surface3d(
    fig: go.Figure,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    surface_axis_col: str | None = None,
    line_dash: str | dict[str, str] | None = None,
    opacity: float | None = None,
) -> None:
    if surface_axis_col is None or surface_axis_col not in df.columns:
        return

    for cfg in df["config"].unique():
        sub = df[df["config"] == cfg]
        piv = sub.pivot_table(
            index=surface_axis_col,
            columns=x_col,
            values=y_col,
            aggfunc="mean",
            observed=True,
        ).sort_index()
        if piv.empty:
            continue
        trace_kw: dict[str, Any] = {
            "z": piv.to_numpy(),
            "x": list(piv.columns),
            "y": list(piv.index),
            "name": str(cfg),
            "showscale": True,
            "colorbar": {"title": str(cfg)},
            "contours": {
                "z": {
                    "show": True,
                    "usecolormap": True,
                    "highlightcolor": "#333333",
                    "project_z": True,
                }
            },
        }
        if opacity is not None:
            trace_kw["opacity"] = opacity
        fig.add_trace(go.Surface(**trace_kw))


_BUILDERS: dict[str, Any] = {
    "Line": _build_line,
    "Bar": _build_bar,
    "Grouped Bar": _build_grouped_bar,
    "Stacked Bar": _build_stacked_bar,
    "Stacked Area": _build_stacked_area,
    "Heatmap": _build_heatmap,
    "3D Surface": _build_surface3d,
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
            numeric_col = _numeric(df[col])
            mask = mask | (numeric_col == nv)
        except (ValueError, TypeError):
            pass
    return _frame(df[mask]).copy()


def _format_x_axis(df: pd.DataFrame, x_cfg: dict[str, Any]) -> pd.DataFrame:
    """Step 5: Apply categorical ordering and rename AFTER aggregation."""
    col: str = x_cfg["column"]
    if x_cfg.get("values"):
        vals = x_cfg["values"]
        str_vals = [str(v) for v in vals]
        # Convert to string first to ensure consistency
        df[col] = df[col].astype(str)
        df[col] = pd.Categorical(df[col], categories=str_vals, ordered=True)
    else:
        # Auto-sort numerically when all x values are numeric
        numeric = _numeric(df[col])
        if numeric.notna().all():
            df[col] = numeric
    if x_cfg.get("rename"):
        rmap: dict[str, str] = x_cfg["rename"]
        df[col] = df[col].astype(str).map(lambda v, _r=rmap: _r.get(v, v))
        if x_cfg.get("values"):
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
    if group_cfg.get("values"):
        vals = group_cfg["values"]
        str_vals = [str(v) for v in vals]
        mask = _series(df[col]).astype(str).isin(str_vals)
        df = _frame(df[mask]).copy()
        df[col] = pd.Categorical(
            df[col].astype(str), categories=str_vals, ordered=True
        )
    if group_cfg.get("rename"):
        rmap: dict[str, str] = group_cfg["rename"]
        df[col] = df[col].astype(str).map(lambda v, _r=rmap: _r.get(v, v))
        if group_cfg.get("values"):
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
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    plot_id: str,
    extra_key_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Step 8: If multiple rows share the same (config, x) pair, collapse them
    via mean.  Prevents zigzag lines and stacked anomalies.
    """
    if "config" not in df.columns or x_col not in df.columns:
        return df

    group_keys = ["config", x_col]
    for col in extra_key_cols or []:
        if col in df.columns and col not in group_keys:
            group_keys.append(col)

    dup_count = df.groupby(group_keys, observed=True).size()
    if (dup_count > 1).any():
        max_dups = int(_series(dup_count).max())
        st.caption(
            f"⚠️ [{plot_id}] Up to {max_dups} rows per plot key — "
            f'auto-averaging.  Add `"aggregate"` to spec to control this.'
        )
        # Identify all numeric columns to aggregate
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        value_cols = [c for c in numeric_cols if c not in group_keys]
        if value_cols:
            agg_dict: dict[str, str] = {c: "mean" for c in value_cols}
            # Keep first value for non-numeric columns
            non_numeric = [
                c
                for c in df.columns
                if c not in numeric_cols and c not in group_keys
            ]
            for c in non_numeric:
                agg_dict[c] = "first"
            df = (
                df.groupby(group_keys, observed=True)
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
                numeric_col = _numeric(col_series)
                matched = matched | (numeric_col == val)
            matched = matched | (col_series.astype(str) == str(val))
        df = _frame(df[matched])
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
    if isinstance(source, list):
        missing = [s for s in source if s not in all_dfs]
        if missing:
            st.warning(
                f"Source CSV(s) not uploaded: {', '.join(missing)} — "
                "plot uses only the available ones."
            )
        parts = [all_dfs[s] for s in source if s in all_dfs]
        df = pd.concat(parts, ignore_index=True) if parts else combined.copy()
    else:
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
    surface_axis_cfg: dict[str, Any] | None = plot_spec.get(
        "surface_y"
    ) or plot_spec.get("axis_y")
    y_cfg: dict[str, Any] = plot_spec["y"]
    group_cfg: dict[str, Any] | None = plot_spec.get("group")
    agg_cfg: dict[str, Any] = plot_spec.get("aggregate", {})
    x_col: str = x_cfg["column"]
    surface_axis_col: str | None = (
        surface_axis_cfg.get("column") if surface_axis_cfg else None
    )
    x_label: str = x_cfg.get("label", x_col)
    surface_axis_label: str | None = (
        surface_axis_cfg.get("label", surface_axis_col)
        if surface_axis_cfg
        else None
    )
    y_columns: list[str] = y_cfg.get("columns", [])
    y_label: str = y_cfg.get("label", y_columns[0] if y_columns else "Value")
    y_rename: dict[str, str] = y_cfg.get("rename", {})

    if x_col not in df.columns:
        st.warning(f"[{plot_id}] Column `{x_col}` not found in CSV.")
        return None
    if chart_type == "3D Surface":
        if not surface_axis_cfg or not surface_axis_col:
            st.warning(
                f"[{plot_id}] 3D Surface requires `surface_y` "
                f"or `axis_y` with a `column` field."
            )
            return None
        if surface_axis_col not in df.columns:
            st.warning(
                f"[{plot_id}] Column `{surface_axis_col}` not found in CSV."
            )
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
    if surface_axis_cfg:
        df = _filter_x_values(df, surface_axis_cfg)
    if df.empty:
        st.warning(f"[{plot_id}] No data after applying filters.")
        return None

    # ── 5. Aggregation (on raw, unformatted data) ────────────────────
    agg_func: str | None = agg_cfg.get("func")
    agg_error_bars: str | None = agg_cfg.get("error_bars")
    transform: dict[str, Any] = plot_spec.get("transform", {})
    if agg_func:
        gk: list[str] = [x_col]
        if surface_axis_col and surface_axis_col in df.columns:
            gk.append(surface_axis_col)
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
    if surface_axis_cfg:
        df = _format_x_axis(df, surface_axis_cfg)

    # ── 7. Normalize transform ───────────────────────────────────────
    if transform.get("normalize"):
        df = _normalize_stacked(
            df, transform.get("normalize_columns", y_columns)
        )
    if transform.get("normalize_distribution"):
        _grp_col = (
            group_cfg["column"]
            if group_cfg and group_cfg.get("column") in df.columns
            else None
        )
        df = _normalize_distribution(
            df,
            transform.get("normalize_columns", y_columns),
            ref_column=transform.get("normalize_ref_column"),
            group_column=_grp_col,
        )
    if transform.get("normalize_group_peak"):
        _grp_col = (
            group_cfg["column"]
            if group_cfg and group_cfg.get("column") in df.columns
            else None
        )
        df = _normalize_group_peak(
            df,
            transform.get("normalize_columns", y_columns),
            group_column=_grp_col,
        )
    if transform.get("diff"):
        _grp_col = (
            group_cfg["column"]
            if group_cfg and group_cfg.get("column") in df.columns
            else None
        )
        df = _apply_group_diff(
            df,
            transform.get("normalize_columns", y_columns),
            x_col,
            group_column=_grp_col,
            scale=float(transform.get("diff_scale", 1.0)),
            drop_first=bool(transform.get("diff_drop_first", False)),
        )

    # ── 8. Resolve traces ────────────────────────────────────────────
    multi_y = len(y_columns) > 1
    has_group = group_cfg is not None

    if multi_y and not has_group:
        rows: list[pd.DataFrame] = []
        for yc in y_columns:
            if yc not in df.columns:
                continue
            tmp = _frame(df[[x_col, yc]]).copy()
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
            tmp = _frame(df[[x_col, yc, "config"]]).copy()
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
    plot_df = _auto_dedup(
        plot_df,
        x_col,
        y_plot_col,
        plot_id,
        extra_key_cols=[surface_axis_col] if surface_axis_col else None,
    )

    # ── 10. Render ───────────────────────────────────────────────────
    style_line_dash: str | dict[str, str] | None = plot_spec.get("line_dash")
    style_opacity: float | None = plot_spec.get("opacity")

    fig = go.Figure()
    if chart_type == "3D Surface":
        builder(
            fig,
            plot_df,
            x_col,
            y_plot_col,
            surface_axis_col=surface_axis_col,
            line_dash=style_line_dash,
            opacity=style_opacity,
        )
    elif chart_type == "Line":
        builder(
            fig,
            plot_df,
            x_col,
            y_plot_col,
            line_dash=style_line_dash,
            opacity=style_opacity,
            error_band=bool(plot_spec.get("error_band", False)),
            colorway=plot_spec.get("layout", {}).get("colorway"),
        )
    else:
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
    if chart_type == "3D Surface":
        fig.update_layout(
            scene={
                "xaxis_title": x_label,
                "yaxis_title": surface_axis_label or surface_axis_col,
                "zaxis_title": y_label,
            }
        )

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
        extract_source: str = (
            st.sidebar.selectbox(
                "Source column",
                options=all_source_cols,
                index=default_src_idx,
                key="extract_source",
            )
            or ""
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
    group_col: str = (
        st.sidebar.selectbox(
            "Group traces by",
            options=all_available,
            index=default_group_idx,
            key="group_col",
        )
        or "config"
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
    combined["config"] = _series(combined["config"]).map(rename_map)

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

    st.sidebar.header("Chart Type")
    chart_type: str = st.sidebar.selectbox(
        "Visualization", options=CHART_TYPES
    )  # type: ignore[assignment]

    surface_axis_col: str | None = None
    if chart_type == "3D Surface":
        surface_axis_options = [c for c in all_cols if c != x_col]
        default_surface_idx = 0
        if "SNR" in surface_axis_options:
            default_surface_idx = surface_axis_options.index("SNR")
        surface_axis_col = st.sidebar.selectbox(
            "Surface Y-Axis column",
            options=surface_axis_options,
            index=default_surface_idx,
            key="surface_axis_col",
        )

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
        group_keys = [x_col, "config"]
        if surface_axis_col and surface_axis_col in plot_df.columns:
            group_keys.insert(1, surface_axis_col)
        plot_df = _aggregate_data(
            _frame(plot_df),
            group_keys=group_keys,
            value_cols=selected_metrics,
            func=agg_func,
            error_bars=agg_errors if agg_errors != "none" else None,
        )

    # Auto-sort x-axis numerically when all values are numeric
    numeric_x = _numeric(plot_df[x_col])
    if numeric_x.notna().all():
        plot_df[x_col] = numeric_x
    if surface_axis_col:
        numeric_surface_axis = pd.to_numeric(
            plot_df[surface_axis_col], errors="coerce"
        )
        if numeric_surface_axis.notna().all():
            plot_df[surface_axis_col] = numeric_surface_axis

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
        if chart_type == "3D Surface":
            _BUILDERS[chart_type](
                fig,
                plot_df,
                x_col,
                metric,
                surface_axis_col=surface_axis_col,
                **style_kw,
            )
        else:
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
        if chart_type == "3D Surface":
            fig.update_layout(
                scene={
                    "xaxis_title": x_label,
                    "yaxis_title": surface_axis_col,
                    "zaxis_title": y_label,
                }
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

    # 1. CSVs (upload or pick from a server folder)
    uploaded_csvs = file_input(
        "One or more sweep-result CSVs",
        type=["csv"],
        accept_multiple_files=True,
        key="csv_uploader",
        default_dir="output_data",
    )

    # 2. JSON spec(s)
    json_files = file_input(
        "JSON plot spec(s) (optional — enables auto-plot mode)",
        type=["json"],
        accept_multiple_files=True,
        key="json_uploader",
        default_dir="configs",
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
