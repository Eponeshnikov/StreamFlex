"""Human-readable configuration names, values, summaries and lineage;
Plotly style config loading.
"""

import json
import os

import numpy as np
import streamlit as st

# --- UTILITY FUNCTIONS ---

_CONFIG_ACRONYMS = {
    "adc": "ADC",
    "awgn": "AWGN",
    "ber": "BER",
    "cfo": "CFO",
    "cir": "CIR",
    "dc": "DC",
    "dt": "Sample Interval",
    "iq": "IQ",
    "los": "LOS",
    "mimo": "MIMO",
    "nlos": "NLOS",
    "ofdm": "OFDM",
    "qam": "QAM",
    "rx": "RX",
    "snr": "SNR",
    "tx": "TX",
}

_PLUGIN_LABELS = {
    "cir_generator": "Channel Model",
    "code_generator": "Code",
    "gold_codes": "Gold Code",
    "iq_generator": "IQ Modulation",
    "kasami_codes": "Kasami Code",
    "m_sequences": "M-Sequence",
    "optimal_receiver": "Receiver",
    "pulse_shaping": "Pulse Shaping",
    "signal_channelizer": "Propagation",
}


def humanize_config_name(name: object) -> str:
    """Convert an internal snake-case identifier to a UI label."""
    text_value = str(name or "Parameter").replace(".", " ").replace("_", " ")
    words = []
    for word in text_value.split():
        lower = word.lower()
        words.append(_CONFIG_ACRONYMS.get(lower, word.capitalize()))
    return " ".join(words)


def _format_si(value: float, unit: str) -> str:
    scales = (
        (1e9, "G"),
        (1e6, "M"),
        (1e3, "k"),
        (1.0, ""),
        (1e-3, "m"),
        (1e-6, "µ"),
        (1e-9, "n"),
        (1e-12, "p"),
    )
    magnitude = abs(value)
    for scale, prefix in scales:
        if magnitude >= scale or scale == 1e-12:
            return f"{value / scale:.6g} {prefix}{unit}"
    return f"{value:.6g} {unit}"


def format_config_value(key: object, value: object) -> str:
    """Compact, unit-aware formatting for configuration values."""
    key_l = str(key).lower()
    if value is None:
        return "Not set"
    if isinstance(value, (bool, np.bool_)):
        return "Enabled" if value else "Disabled"
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value.item()
        elif value.size <= 8:
            value = value.tolist()
        else:
            return f"Array · shape {' × '.join(map(str, value.shape))} · {value.dtype}"
    if isinstance(value, dict):
        if set(value) >= {"re", "im"}:
            re_shape = np.shape(value["re"])
            return f"Complex array · shape {' × '.join(map(str, re_shape)) or 'scalar'}"
        return ", ".join(
            f"{humanize_config_name(k)}: {format_config_value(k, v)}"
            for k, v in value.items()
        )
    if isinstance(value, (list, tuple)):
        if len(value) > 8:
            return f"{type(value).__name__.title()} · {len(value)} values"
        return " × ".join(format_config_value(key, v) for v in value)
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}".replace(",", " ")
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return str(number)
        if "snr" in key_l:
            return f"{number:g} dB"
        if "power_dbm" in key_l or key_l.endswith("dbm"):
            return f"{number:g} dBm"
        if any(
            token in key_l
            for token in (
                "frequency",
                "symbol_rate",
                "sample_rate",
                "bandwidth",
            )
        ):
            return _format_si(number, "Hz")
        if key_l == "dt" or "delay" in key_l:
            return _format_si(number, "s")
        return f"{number:.6g}"
    text_value = str(value).replace("_", " ")
    return " ".join(
        _CONFIG_ACRONYMS.get(word.lower(), word.capitalize())
        for word in text_value.split()
    )


def config_lineage(config_info: object) -> list[dict]:
    """Return upstream-to-downstream config stages from nested source_info."""
    stages: list[dict] = []
    seen: set[int] = set()

    def visit(node):
        if isinstance(node, (list, tuple)):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict) or id(node) in seen:
            return
        seen.add(id(node))
        source = node.get("source_info")
        if source:
            visit(source)
        if "parameters" in node or "plugin_key" in node:
            stages.append(node)

    visit(config_info)
    return stages


def format_config_summary(
    config_info: object, index: int | None = None
) -> str:
    """Build a readable, stable selectbox label for a result configuration."""
    stages = config_lineage(config_info)
    current = (
        stages[-1]
        if stages
        else (config_info if isinstance(config_info, dict) else {})
    )
    prefix = f"Result {index + 1}" if index is not None else "Result"
    plugin = _PLUGIN_LABELS.get(
        current.get("plugin_key", ""),
        humanize_config_name(current.get("plugin_key", "Configuration")),
    )
    priority = (
        "channel_backend",
        "model_type",
        "model",
        "modulation",
        "filter_type",
        "symbol_rate",
        "snr",
        "samples_per_symbol",
        "delay_spread",
    )
    details = []
    for key in priority:
        for stage in reversed(stages or [current]):
            params = (
                stage.get("parameters", {}) if isinstance(stage, dict) else {}
            )
            if key in params and params[key] is not None:
                details.append(
                    f"{humanize_config_name(key)}: {format_config_value(key, params[key])}"
                )
                break
        if len(details) == 4:
            break
    return " · ".join([prefix, plugin, *details])


def render_config_lineage(
    config_info: object, *, expanded: bool = False
) -> None:
    """Render every current and upstream parameter as a readable table."""
    stages = config_lineage(config_info)
    if not stages:
        return
    rows = []
    for stage in stages:
        plugin_key = stage.get("plugin_key", "Configuration")
        stage_label = _PLUGIN_LABELS.get(
            plugin_key, humanize_config_name(plugin_key)
        )
        stage_id = stage.get("id")
        if stage_id is not None:
            stage_label = f"{stage_label} #{stage_id}"
        for key, value in (stage.get("parameters") or {}).items():
            rows.append(
                {
                    "Stage": stage_label,
                    "Parameter": humanize_config_name(key),
                    "Value": format_config_value(key, value),
                }
            )
    if rows:
        with st.expander("Configuration details", expanded=expanded):
            st.dataframe(rows, width="stretch", hide_index=True)


def unflatten_dict(d: dict) -> dict:
    """Converts a flat dictionary with dot-separated keys to a nested dictionary."""
    result = {}
    for key, value in d.items():
        if key.startswith("layout."):
            key = key.replace("layout.", "", 1)
        parts = key.split(".")
        nested_dict = result
        for part in parts[:-1]:
            nested_dict = nested_dict.setdefault(part, {})
        nested_dict[parts[-1]] = value
    return result


@st.cache_data
def find_plotly_configs(config_dir: str = "configs/plotly") -> list[str]:
    """Finds all .json configuration files in the specified directory."""
    if not os.path.isdir(config_dir):
        return []
    return [
        os.path.splitext(f)[0]
        for f in os.listdir(config_dir)
        if f.endswith(".json") and os.path.isfile(os.path.join(config_dir, f))
    ]


@st.cache_data
def load_plot_config(config_name: str) -> dict:
    """Loads a specific plot configuration file."""
    config_path = os.path.join("configs", "plotly", f"{config_name}.json")
    if not os.path.exists(config_path):
        st.error(f"Configuration file not found at: `{config_path}`")
        return {"light": {}, "dark": {}}
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.error(
            f"Could not parse `{config_path}`. Please ensure it is a valid JSON file."
        )
        return {"light": {}, "dark": {}}
