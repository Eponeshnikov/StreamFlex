# CSV Plot Editor — JSON Specification Reference

Generate publication-ready plots from CSV data by describing them in JSON.

## Quick Start

1. Export one or more CSV files (each must have a `config` column, or the filename is used).
2. Write a JSON spec file describing your plots.
3. Upload both to the CSV Plot Editor page — all plots render automatically.

```json
{
  "plots": [
    {
      "id": "my_plot",
      "title": "Accuracy vs SNR",
      "chart_type": "line",
      "x": {"column": "SNR", "label": "SNR (dB)"},
      "y": {"columns": ["Accuracy"], "label": "Accuracy"},
      "group": {"column": "Model"}
    }
  ]
}
```

---

## Plot Object

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Unique identifier (used as widget key and filename on export). |
| `title` | string | yes | Chart title. |
| `chart_type` | string | yes | One of: `line`, `bar`, `grouped_bar`, `stacked_bar`, `stacked_area`, `heatmap`. |
| `source` | string | no | CSV filename to use. Omit to combine all uploaded CSVs. |
| `extract_columns` | array | no | Extract new columns from text via regex or Key:Value splitting. |
| `computed_columns` | array | no | Derive columns using pandas expressions. |
| `filters` | object | no | Keep only rows matching allowed values. |
| `x` | object | yes | X-axis configuration. |
| `y` | object | yes | Y-axis configuration. |
| `group` | object | no | Split data into separate traces. |
| `aggregate` | object | no | Collapse rows sharing the same (x, group) pair. |
| `transform` | object | no | Post-aggregation transforms (normalize, scale). See [Transform](#transform-transform). |
| `x_scale` | string | no | X-axis scale type: `"linear"` (default) or `"log"`. |
| `y_scale` | string | no | Y-axis scale type: `"linear"` (default) or `"log"`. |
| `layout` | object | no | Plotly layout overrides (passed directly to `fig.update_layout`). |
| `line_dash` | string or object | no | Line style for line/area charts. See [Line Dash](#line-dash-line_dash). |
| `opacity` | number | no | Trace opacity, `0.0` (transparent) to `1.0` (opaque). Applies to all chart types. |
| `show_table` | bool | no | Show an expandable data table under the plot. |
| `note` | string | no | Caption displayed below the chart. |

---

## Processing Pipeline

Each plot is processed in this order:

```
CSV data
  │
  ├─  1. extract_columns   Extract new columns from text (regex / auto-KV)
  ├─  2. computed_columns   Derive columns via expressions
  ├─  3. filters            Drop non-matching rows
  ├─  4. x.values filter    Keep only requested x-values (no rename yet)
  ├─  5. aggregate          Collapse rows (mean/sum/…) + error bars
  ├─  6. x format           Apply categorical ordering + rename (after aggregation)
  ├─  7. transform          Normalize stacks, distributions, peak-scale groups
  ├─  8. group → config     Split into Plotly traces
  ├─  9. auto-dedup         Average any remaining (config, x) duplicates
  └─ 10. render             Build Plotly figure
```

Steps 4 and 6 are deliberately split so that aggregation (step 5) operates on raw values, not renamed/categorical strings.

If duplicates remain after step 8, auto-dedup averages them and displays a warning.

---

## Column Extraction (`extract_columns`)

Extract new columns from a text column that contains structured data. Useful when a single column encodes multiple parameters (e.g. `Config 1 | SNR:-10 | Scene:florence | Model:RT`).

Two modes are available:

### Auto Key:Value

Splits on a pair separator (`|`) then a key-value separator (`:`) to discover all pairs automatically.

```json
"extract_columns": [
  {"source": "config", "auto_kv": true}
]
```

From `Config 1 | SNR:-10 | Scene:florence | Model:RT` this creates columns `SNR`, `Scene`, `Model` (the first segment `Config 1` has no `:` and is skipped). Numeric values are auto-converted.

| Field | Type | Default | Description |
|---|---|---|---|
| `source` | string | `"config"` | Column to extract from. |
| `auto_kv` | bool | — | Must be `true` to enable this mode. |
| `separator` | string | `"\|"` | Delimiter between pairs. |
| `kv_separator` | string | `":"` | Delimiter between key and value. |
| `keys` | array | all | Keep only these keys (omit to keep all discovered keys). |

```json
"extract_columns": [
  {
    "source": "config",
    "auto_kv": true,
    "separator": "|",
    "kv_separator": ":",
    "keys": ["SNR", "Model"]
  }
]
```

### Regex with Named Groups

Use `(?P<Name>...)` capture groups. Each group becomes a column.

```json
"extract_columns": [
  {"source": "config", "pattern": "SNR:(?P<SNR>[^|]+).*Model:(?P<Model>[^|]+)"}
]
```

| Field | Type | Default | Description |
|---|---|---|---|
| `source` | string | `"config"` | Column to extract from. |
| `pattern` | string | — | Regex with one or more `(?P<Name>...)` named groups. |

---

## Computed Columns (`computed_columns`)

Derive new columns using `pd.eval` expressions. Runs after extraction and before filters, so extracted and computed columns can be used everywhere downstream.

```json
"computed_columns": [
  {"name": "total_loss", "expr": "`Loss A` + `Loss B` + `Loss C`"},
  {"name": "f1_pct", "expr": "`F1 Score` * 100"}
]
```

Column names containing spaces or special characters must be wrapped in backticks inside `expr`.

---

## Filters (`filters`)

Keep only rows where the column value is in the allowed list. Type coercion is automatic: `[10]` matches both integer `10` and string `"10"`.

```json
"filters": {
  "SNR": [10, 20],
  "Scene": ["munich", "florence"]
}
```

---

## X-Axis (`x`)

| Field | Type | Required | Description |
|---|---|---|---|
| `column` | string | yes | CSV column name. |
| `label` | string | no | Display label (defaults to `column`). |
| `values` | array | no | Subset and ordering of X-values. Only these are shown, in this order. |
| `rename` | object | no | `{"raw_value": "display_name"}` mapping for tick labels. |

```json
"x": {
  "column": "SNR",
  "label": "SNR (dB)",
  "values": [-10, -5, 0, 5, 10, 20],
  "rename": {"-10": "-10 dB", "20": "20 dB"}
}
```

---

## Y-Axis (`y`)

| Field | Type | Required | Description |
|---|---|---|---|
| `columns` | array | yes | One or more CSV column names. |
| `label` | string | no | Y-axis display label. |
| `rename` | object | no | `{"column_name": "legend_name"}` for multi-column traces. |

```json
"y": {
  "columns": ["Accuracy", "F1 Score"],
  "label": "Score",
  "rename": {"Accuracy": "Acc", "F1 Score": "F1"}
}
```

---

## Group (`group`)

Split data into separate traces by a column's values.

| Field | Type | Required | Description |
|---|---|---|---|
| `column` | string | yes | Column to split on. |
| `values` | array | no | Subset and ordering of group values. |
| `rename` | object | no | `{"raw_value": "display_name"}` for legend entries. |

```json
"group": {
  "column": "Model",
  "values": ["RT", "tdl"],
  "rename": {"RT": "Ray Tracing", "tdl": "TDL"}
}
```

### Trace Resolution

| Y columns | Group present? | Result |
|---|---|---|
| 1 | no | 1 trace labelled "all" |
| 1 | yes | 1 trace per group value |
| N > 1 | no | 1 trace per Y column |
| N > 1 | yes | N × G traces (group — y_column) |

---

## Aggregation (`aggregate`)

Collapse multiple rows that share the same (x, group) combination into a single point.

| Field | Type | Required | Description |
|---|---|---|---|
| `func` | string | yes | `mean`, `median`, `sum`, `count`, `min`, `max`, `std`, `first`, `last`. |
| `error_bars` | string | no | `std`, `sem`, `minmax`, `q25_q75`. Omit or `"none"` to disable. |

Error bar types:
- `std` — symmetric ±1 standard deviation
- `sem` — symmetric ±SEM (std / √n)
- `minmax` — asymmetric bars from actual min/max values
- `q25_q75` — asymmetric bars from 25th/75th percentiles

```json
"aggregate": {"func": "mean", "error_bars": "std"}
```

---

## Transform (`transform`)

Post-aggregation transformations applied at step 7. Multiple transforms can be combined — they run in this order:

1. `normalize` — row-wise normalization (stacked → 100%)
2. `normalize_distribution` — column-wise distribution normalization
3. `normalize_group_peak` — per-group peak scaling

| Field | Type | Description |
|---|---|---|
| `normalize` | bool | Rescale Y columns so each row sums to 100%. Use for stacked charts. |
| `normalize_distribution` | bool | Normalize Y columns as a distribution (%). See below. |
| `normalize_ref_column` | string | Reference column for distribution normalization. Each column is divided by this column's total. |
| `normalize_columns` | array | Which Y columns to normalize (defaults to all `y.columns`). |
| `normalize_group_peak` | bool | Scale each group so its peak Y value equals 100%. Makes groups directly comparable by shape. |

### `normalize` (row-wise, for stacked charts)

Each row's Y values are divided by their sum, then multiplied by 100. This makes stacked bar / stacked area charts show proportions instead of raw values.

```json
"transform": {"normalize": true}
```

### `normalize_distribution` (column-wise)

Converts absolute values into a percentage distribution across bins/rows.

- **Without** `normalize_ref_column`: each Y column is independently divided by its own sum.
- **With** `normalize_ref_column`: all Y columns are divided by the ref column's total sum.

When a `group` column is present, normalization is performed **per group** — each group's values sum to 100% independently.

```json
"transform": {
  "normalize_distribution": true,
  "normalize_ref_column": "True Ray Power in Bin"
}
```

### `normalize_group_peak` (peak scaling)

After other normalizations, scales each group independently so its maximum Y value equals 100. Useful when groups have slightly different absolute magnitudes (e.g. due to noise or realization differences) and you want to compare their shapes directly.

When no `group` column is present, the entire dataset is scaled by its global peak.

```json
"transform": {
  "normalize_distribution": true,
  "normalize_ref_column": "True Ray Power in Bin",
  "normalize_group_peak": true
}
```

---

## Axis Scale (`x_scale`, `y_scale`)

Set axis scale type at the plot level. Maps to Plotly's `xaxis_type` / `yaxis_type`.

| Value | Description |
|---|---|
| `"linear"` | Linear scale (default). |
| `"log"` | Logarithmic scale. |

```json
"x_scale": "linear",
"y_scale": "log"
```

---

## Line Dash (`line_dash`)

Set the line style for `line` and `stacked_area` charts. Ignored by bar and heatmap chart types.

Available styles: `solid`, `dot`, `dash`, `longdash`, `dashdot`, `longdashdot`.

As a **string** — applies the same dash to every trace:

```json
"line_dash": "dash"
```

As an **object** — map config/trace names to individual styles:

```json
"line_dash": {
  "Ray Tracing": "solid",
  "TDL": "dot",
  "CDL": "dashdot"
}
```

Unmapped traces default to `"solid"`.

---

## Opacity (`opacity`)

Trace opacity from `0.0` (fully transparent) to `1.0` (fully opaque). Applies to all chart types.

```json
"opacity": 0.7
```

---

## Layout Overrides (`layout`)

Flat dictionary passed directly to `fig.update_layout()`:

```json
"layout": {
  "height": 700,
  "yaxis_range": [0.5, 1.0],
  "bargap": 0.15,
  "legend_title_text": "Model",
  "colorway": ["#1f77b4", "#ff7f0e", "#2ca02c"]
}
```

Default template is `plotly_white` with height 600.

---

## Full Example

Given a CSV where the `config` column looks like:
```
Config 1 | SNR:-10 | Scene:florence | Logic:Logic 3 | Model:RT
```

```json
{
  "plots": [
    {
      "id": "accuracy_vs_snr",
      "title": "Accuracy vs SNR by Model",
      "chart_type": "line",
      "extract_columns": [
        {"source": "config", "auto_kv": true}
      ],
      "x": {"column": "SNR", "label": "SNR (dB)"},
      "y": {"columns": ["Accuracy"], "label": "Accuracy"},
      "group": {
        "column": "Model",
        "values": ["RT", "tdl"],
        "rename": {"RT": "Ray Tracing", "tdl": "TDL"}
      },
      "aggregate": {"func": "mean", "error_bars": "std"},
      "note": "Averaged across scenes. Error bars = ±1σ."
    },
    {
      "id": "f1_by_scene",
      "title": "F1 Score by Scene",
      "chart_type": "grouped_bar",
      "extract_columns": [
        {"source": "config", "auto_kv": true, "keys": ["SNR", "Scene", "Model"]}
      ],
      "filters": {"SNR": [10]},
      "x": {
        "column": "Scene",
        "label": "Scene",
        "values": ["munich", "florence"],
        "rename": {"munich": "Munich", "florence": "Florence"}
      },
      "y": {"columns": ["F1 Score"], "label": "F1 Score"},
      "group": {"column": "Model"},
      "show_table": true
    },
    {
      "id": "heatmap_snr_model",
      "title": "Accuracy Heatmap: SNR vs Model",
      "chart_type": "heatmap",
      "extract_columns": [
        {"source": "config", "auto_kv": true}
      ],
      "x": {"column": "SNR", "label": "SNR (dB)"},
      "y": {"columns": ["Accuracy"], "label": "Accuracy"},
      "group": {"column": "Model"},
      "aggregate": {"func": "mean"}
    },
    {
      "id": "multi_metric",
      "title": "Accuracy & F1 vs SNR (RT only)",
      "chart_type": "line",
      "extract_columns": [
        {"source": "config", "pattern": "SNR:(?P<SNR>[^|]+).*Model:(?P<Model>[^|]+)"}
      ],
      "filters": {"Model": ["RT"]},
      "x": {"column": "SNR", "label": "SNR (dB)"},
      "y": {
        "columns": ["Accuracy", "F1 Score"],
        "label": "Score",
        "rename": {"Accuracy": "Acc", "F1 Score": "F1"}
      },
      "aggregate": {"func": "mean"}
    },
    {
      "id": "power_distribution",
      "title": "Power Distribution by Bin (RT, SNR=-10)",
      "chart_type": "stacked_area",
      "filters": {"config": ["All Scenes SNR=-10 | Logic:Logic 3 | Model:RT"]},
      "x": {"column": "SNR_Point", "label": "Per-ray SNR (dB)"},
      "y": {
        "columns": ["TP Power in Bin", "Lost (FN) Power in Bin", "Lost (Time Res) Power in Bin"],
        "label": "Share of total power (%)"
      },
      "aggregate": {"func": "mean"},
      "transform": {
        "normalize_distribution": true,
        "normalize_ref_column": "True Ray Power in Bin"
      }
    },
    {
      "id": "power_comparison_grouped",
      "title": "Power Comparison Across Configs",
      "chart_type": "line",
      "x": {"column": "SNR_Point", "label": "Per-ray SNR (dB)"},
      "y": {
        "columns": ["True Ray Power in Bin", "TP Power in Bin"],
        "label": "Share of group peak (%)"
      },
      "group": {"column": "config"},
      "aggregate": {"func": "mean"},
      "transform": {
        "normalize_distribution": true,
        "normalize_ref_column": "True Ray Power in Bin",
        "normalize_group_peak": true
      },
      "y_scale": "log",
      "line_dash": {"RT": "solid", "TDL": "dash"}
    },
    {
      "id": "computed_example",
      "title": "Acc-F1 Gap vs SNR",
      "chart_type": "line",
      "extract_columns": [
        {"source": "config", "auto_kv": true}
      ],
      "computed_columns": [
        {"name": "gap", "expr": "Accuracy - `F1 Score`"}
      ],
      "x": {"column": "SNR", "label": "SNR (dB)"},
      "y": {"columns": ["gap"], "label": "Accuracy − F1"},
      "group": {"column": "Model"},
      "aggregate": {"func": "mean"}
    }
  ]
}
```

---

## Guide for Generating Specs

1. **Inspect the CSV columns** — identify parameter columns (X, filter, group) and metric columns (Y).
2. **Check if text columns encode multiple parameters** — if so, add `extract_columns` with `auto_kv` or a regex pattern. Extracted columns can then be used in `x`, `group`, `filters`, and `computed_columns`.
3. **Determine if aggregation is needed** — if multiple rows exist per (x, group) pair, add `aggregate`. Use `mean` for most metrics, `sum` for counts, `median` for outlier robustness.
4. **Add error bars** when aggregating 3+ items — `std` or `sem` for line charts, skip for stacked charts.
5. **Use `computed_columns`** for derived metrics not in the CSV.
6. **Use `transform.normalize`** for stacked charts where layers should sum to 100%.
7. **Use `transform.normalize_distribution`** to convert absolute power/count values into a percentage distribution across bins. Add `normalize_ref_column` when all Y columns should be divided by a common reference column's total.
8. **Use `transform.normalize_group_peak`** when comparing groups that differ in absolute magnitude but you want to compare their shapes (peaks aligned at 100%).
9. **Use `x_scale` / `y_scale`** to switch axes to logarithmic scale.
10. **Filter values must match the CSV exactly** — check casing, whitespace, and units.
11. **Always include `id`** — use descriptive slugs like `"fig01_acc_by_scene"`.
