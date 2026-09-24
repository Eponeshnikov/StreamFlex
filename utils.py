import ast
import hashlib

# custom_plotter.py
import io
import json
import numbers
import os
import pickle
import subprocess
import sys
import threading
import traceback
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Literal, cast, overload

import drjit as dr
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from loguru import logger
from sionna.rt import (
    MeshRadioMap,
    PlanarRadioMap,
)  # Import Sionna classes needed for type checking and functionality
from streamlit.elements.lib.layout_utils import Width

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


# --- FILE INPUT WIDGET (upload OR scan a server folder) ---


class LocalFile(io.BytesIO):
    """A file read from disk that mimics Streamlit's ``UploadedFile``.

    It is a ``BytesIO`` subclass exposing ``.name``, ``.size`` and ``.type``,
    so it is a drop-in for the objects returned by ``st.file_uploader`` and can
    be passed directly to ``pd.read_csv``, ``json.load``, ``pickle.load`` etc.
    """

    def __init__(self, path: str):
        with open(path, "rb") as fh:
            data = fh.read()
        super().__init__(data)
        self.name = os.path.basename(path)
        self.path = path
        self.size = len(data)
        self.type = ""

    def getvalue(self) -> bytes:
        return self.getbuffer().tobytes()

    # Value-based identity so the object is stable across Streamlit reruns
    # (e.g. the plugin ``create_widget`` state-diff and caching keys).
    def __eq__(self, other) -> bool:
        return isinstance(other, LocalFile) and other.path == self.path

    def __hash__(self) -> int:
        return hash(self.path)

    def __reduce__(self):
        return (self.__class__, (self.path,))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"LocalFile(name={self.name!r}, path={self.path!r})"


def _scan_folder(folder: str, exts: list[str] | None) -> list[str]:
    """Return sorted file paths directly inside *folder* matching the extensions.

    Only the top level of *folder* is scanned (no recursion). A missing folder
    is created so the expected data location always exists for the user to fill.
    """
    if not folder:
        return []
    # Make sure the data directory exists; create it if it doesn't.
    os.makedirs(folder, exist_ok=True)
    out = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        if (
            exts is None
            or os.path.splitext(path)[1].lower().lstrip(".") in exts
        ):
            out.append(path)
    return sorted(out)


@overload
def file_input(
    label: str,
    type: str | list[str] | None = ...,
    accept_multiple_files: Literal[False] = ...,
    key: str | None = ...,
    help: str | None = ...,
    *,
    default_dir: str = ...,
    default_source: str = ...,
    container=...,
) -> "LocalFile | None": ...


@overload
def file_input(
    label: str,
    type: str | list[str] | None,
    accept_multiple_files: Literal[True],
    key: str | None = ...,
    help: str | None = ...,
    *,
    default_dir: str = ...,
    default_source: str = ...,
    container=...,
) -> "list[LocalFile]": ...


def file_input(
    label: str,
    type: str | list[str] | None = None,
    accept_multiple_files: bool = False,
    key: str | None = None,
    help: str | None = None,
    *,
    default_dir: str = ".",
    default_source: str = "folder",
    container=None,
) -> "LocalFile | list[LocalFile] | None":
    """Drop-in replacement for ``st.file_uploader`` with a server-folder mode.

    A toggle switches between two sources:

    * **Server folder** (default): a text input chooses a directory to scan and
      a select/multiselect lists the matching files. Designed for running on a
      remote server where uploading over a slow connection is impractical.
    * **Browser upload**: the standard ``st.file_uploader``.

    Return value matches ``st.file_uploader``: a single file-like object (or
    ``None``) when ``accept_multiple_files`` is ``False``, otherwise a list.
    Folder selections are returned as :class:`LocalFile` instances.

    Args:
        label: Label shown on the file selection widget.
        type: Allowed extension(s) (string or list), e.g. ``"json"`` or
            ``["pickle", "pkl"]``. ``None`` allows any file.
        accept_multiple_files: Allow selecting more than one file.
        key: Unique key prefix (REQUIRED) for widget state isolation.
        help: Tooltip for the selection widget.
        default_dir: Folder pre-filled in the scan path input.
        default_source: ``"folder"`` (default) or ``"upload"``.
        container: Streamlit container to render into (e.g. ``st.sidebar``).
            Defaults to the top-level ``st``.
    """
    if key is None:
        raise ValueError(
            "file_input requires a unique 'key' to isolate widget state."
        )
    target = container if container is not None else st

    # Normalise the type filter to a list of bare lowercase extensions.
    exts: list[str] | None = None
    if type is not None:
        type_list = [type] if isinstance(type, str) else list(type)
        exts = [t.lower().lstrip(".") for t in type_list]

    use_upload = target.toggle(
        "📤 Upload from browser instead",
        value=(default_source == "upload"),
        key=f"{key}__source_is_upload",
        help="Off: pick files already present in a folder on the server. "
        "On: upload from your local machine (slow over a poor connection).",
    )

    if use_upload:
        return cast(
            "LocalFile | list[LocalFile] | None",
            target.file_uploader(
                label,
                type=type,
                accept_multiple_files=accept_multiple_files,
                key=f"{key}__uploader",
                help=help,
            ),
        )

    # --- Server-folder mode ---
    folder = target.text_input(
        "📁 Server folder to scan",
        value=default_dir,
        key=f"{key}__folder",
        help="Path on the server to search for files (this folder only).",
    )

    matches = _scan_folder(folder, exts)
    if not matches:
        hint = f" matching {exts}" if exts else ""
        target.info(f"No files{hint} found in `{folder or '∅'}`.")
        return [] if accept_multiple_files else None

    def _rel(path: str) -> str:
        try:
            return os.path.relpath(path, folder)
        except ValueError:
            return path

    if accept_multiple_files:
        chosen = target.multiselect(
            label,
            options=matches,
            format_func=_rel,
            key=f"{key}__multiselect",
            help=help,
        )
        return [LocalFile(p) for p in chosen]

    chosen = target.selectbox(
        label,
        options=matches,
        format_func=_rel,
        index=None,
        placeholder="Select a file from the folder…",
        key=f"{key}__selectbox",
        help=help,
    )
    return LocalFile(chosen) if chosen else None


# --- MAIN RENDERING FUNCTION ---


def render_custom_plotly_chart(
    fig: go.Figure,
    width: Width = "stretch",
    key: str | None = None,
):
    """
    Renders a Plotly chart with an in-app toggle to enable advanced styling controls.

    By default, a standard Streamlit chart is shown. A toggle switch allows the user
    to access custom theme selection and interactive styling options.

    Args:
        fig (go.Figure): The Plotly figure object to render.
        width (str, optional): Chart width - "stretch" or "content". Defaults to "stretch".
        key (str): A unique key for the component. This is REQUIRED if you are rendering
                more than one chart on the page to prevent widget state collisions.
    """
    if key is None:
        raise ValueError(
            "The 'key' parameter is required to ensure unique widget IDs. "
            "Please provide a unique string for each chart you render."
        )
    _custom_plotly_chart_fragment(fig, width, key)


@st.fragment
def _custom_plotly_chart_fragment(
    fig: go.Figure, width: Width, key: str
) -> None:
    """The body of :func:`render_custom_plotly_chart`, as a fragment.

    Every control in here (the styling toggle, the options popover) only
    restyles this one chart, so it reruns only this chart instead of the
    page or plugin around it -- which, on the analysis pages, means
    re-rendering every other figure as well.
    """
    # --- Main toggle to switch between standard and custom modes ---
    custom_mode_key = f"{key}_enable_custom_mode"
    if custom_mode_key not in st.session_state:
        st.session_state[custom_mode_key] = False  # Default to off

    # Create columns for toggle and save button
    _toggle_col, _save_col = st.columns([1, 4])

    # with toggle_col:
    st.toggle(
        "Enable Custom Styling",
        key=custom_mode_key,
        help="Toggle to show advanced styling options and apply custom themes.",
    )

    # with save_col:
    # Pickled on click, not on every rerun: a scene or animation figure runs
    # to tens of megabytes, and a download needs no rerun at all.
    st.download_button(
        label="💾 Save Fig",
        data=lambda: pickle.dumps(fig),
        file_name=f"{key}_figure.pickle",
        mime="application/octet-stream",
        key=f"{key}_save_pickle_btn",
        help="Save the current figure as a pickle file",
        width="stretch",
        on_click="ignore",
    )

    # --- RENDER LOGIC ---
    # If custom mode is OFF, display a standard chart
    if not st.session_state[custom_mode_key]:
        st.plotly_chart(
            fig,
            width=width,
            theme="streamlit",  # Use streamlit's default theme
            key=f"{key}_default_chart",
        )
        return

    # --- If custom mode is ON, display the advanced controls ---
    else:
        # 1. FIND AND SELECT CONFIGURATION
        # ---------------------------------
        available_configs = find_plotly_configs()
        if not available_configs:
            st.error(
                "No Plotly configuration files found in `configs/plotly/` directory."
            )
            st.info(
                "To use custom styles, please create a theme using a generator app and save it as a .json file in that folder."
            )
            st.plotly_chart(fig, width=width)
            return

        # Create UI for selecting config and chart options
        top_cols = st.columns([3, 1])
        with top_cols[0]:
            selected_config_name = st.selectbox(
                "Select Chart Style",
                options=available_configs,
                key=f"{key}_config_select",
                label_visibility="collapsed",
            )

        # Load the selected configuration
        plot_configs = load_plot_config(selected_config_name)

        # 2. INITIALIZE SESSION STATE AND DETECT THEME
        # ------------------------------------------
        try:
            # st.context is deprecated, st.get_option is the modern way
            current_theme_type = st.context.theme.type
        except AttributeError:
            current_theme_type = "light"  # Fallback for older versions

        # Define unique keys for all widgets
        match_theme_key = f"{key}_match_app_theme"
        show_legend_key = f"{key}_show_legend"
        export_format_key = f"{key}_export_format"
        export_scale_key = f"{key}_export_scale"
        use_st_theme_key = f"{key}_use_st_theme"
        show_border_key = f"{key}_show_border"

        # Set defaults in session state if they don't exist
        if match_theme_key not in st.session_state:
            st.session_state[match_theme_key] = True
        if show_legend_key not in st.session_state:
            theme_for_default = (
                current_theme_type
                if st.session_state[match_theme_key]
                else "light"
            )
            st.session_state[show_legend_key] = plot_configs.get(
                theme_for_default, {}
            ).get("layout.showlegend", True)
        if export_format_key not in st.session_state:
            st.session_state[export_format_key] = "svg"
        if export_scale_key not in st.session_state:
            st.session_state[export_scale_key] = 2
        if use_st_theme_key not in st.session_state:
            st.session_state[use_st_theme_key] = False
        if show_border_key not in st.session_state:
            st.session_state[show_border_key] = True

        # 3. DEFINE UI CONTROLS IN A POPOVER
        # ------------------------------------
        with top_cols[1], st.popover("⚙️ Options"):
            st.markdown("**General**")
            st.checkbox(
                "Match App Theme",
                key=match_theme_key,
                help="Automatically switch between light/dark themes based on the app's theme.",
            )
            st.toggle(
                "Show Legend",
                key=show_legend_key,
                help="Show or hide the plot legend.",
            )
            st.toggle(
                "Use Streamlit Theme",
                key=use_st_theme_key,
                help="Override custom styles with Streamlit's native theme.",
            )
            st.checkbox("Show Container Border", key=show_border_key)

            st.markdown("**Image Export**")
            st.selectbox(
                "Format",
                options=["svg", "png", "jpeg", "webp"],
                key=export_format_key,
            )
            st.number_input(
                "Scale (multiplier)",
                min_value=1,
                max_value=10,
                step=1,
                key=export_scale_key,
            )

            with st.expander("View Current Style Config"):
                theme_to_display = (
                    current_theme_type
                    if st.session_state[match_theme_key]
                    else "light"
                )
                st.json(plot_configs.get(theme_to_display, {}))

        # 4. APPLY STYLES AND RENDER
        # ----------------------------
        fig_to_render = deepcopy(fig)

        # Determine which theme (light/dark) to use
        active_style_dict = plot_configs.get(
            current_theme_type
            if st.session_state[match_theme_key]
            else "light",
            {},
        )

        # Override style with interactive controls
        active_style_dict["layout.showlegend"] = st.session_state[
            show_legend_key
        ]

        # Unflatten and apply the style dictionary
        if active_style_dict:
            nested_style = unflatten_dict(active_style_dict)
            fig_to_render.update_layout(nested_style)

        chart_config = {
            "toImageButtonOptions": {
                "format": st.session_state[export_format_key],
                "scale": st.session_state[export_scale_key],
            }
        }

        chart_theme_param = (
            "streamlit" if st.session_state[use_st_theme_key] else None
        )

        with st.container(border=st.session_state[show_border_key]):
            st.plotly_chart(
                fig_to_render,
                width=width,
                config=chart_config,
                theme=chart_theme_param,
                key=f"{key}_custom_chart",  # Use a unique key for the custom chart
            )


def generate_unique_filename(plugin_name, data, *args, **kwargs):
    """
    Generate a unique filename by combining the plugin_name with a hash of the serialized data.

    Parameters:
        plugin_name (str): The name of the plugin.
        data: The data to be serialized and hashed.

    Returns:
        str: A unique filename in the format "plugin_name_hash.pickle".
    """
    combined_data = (plugin_name, data, args, kwargs)
    serialized_data = pickle.dumps(combined_data)
    hash_object = hashlib.sha256(serialized_data)
    filename = f"{plugin_name}^" + hash_object.hexdigest()
    return filename


def save_to_pickle(data, filename, folder="cache"):
    """
    Save data to a pickle file and return the file path.

    Parameters:
        data: The data to be saved.
        filename (str): The name of the pickle file.
        folder (str): The folder where the pickle file will be saved.

    Returns:
        str: The full path to the saved pickle file.
    """
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    # Create the full file path
    file_path = os.path.join(folder, filename)  # type: ignore

    # Save the data to the pickle file
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    return file_path


def read_data(
    data, save_flag, shape=None, dtype: np.dtype | type = np.complex64
):
    if save_flag:
        if data.endswith(".pickle"):
            with open(data, "rb") as file:
                readed_data = pickle.load(file)
        elif data.endswith(".bin"):
            readed_data = np.memmap(
                data,
                dtype=dtype,
                mode="r",
                shape=shape,
            )
        else:
            raise ValueError(
                f"Unsupported file extension for '{data}'; expected .pickle or .bin"
            )
    else:
        readed_data = data  # 7D np.ndarray
    return readed_data


# Helper function to safely parse values, especially for None and numbers
def safe_literal_eval(value_str, expected_type=None, allow_none=False):
    """
    Safely evaluate a string literal, handling None and basic types.

    Parameters:
        value_str (str): The string to be evaluated. It can represent a literal value like a number, string, or None.
        expected_type (str, optional): The expected type of the evaluated value.
            Can be "int", "float", or "str". If provided, the function will enforce type checking.
        allow_none (bool, optional): Whether to allow the string "None" to be evaluated as None.
            If False, a ValueError will be raised if "None" is encountered.

    Returns:
        The evaluated value, which can be an int, float, str, list, or None, depending on the input.

    Raises:
        ValueError: If the input string cannot be evaluated, or if the evaluated value does not match the expected type,
                or if "None" is encountered but `allow_none` is False.
    """
    # self.logger.debug(f"Attempting safe_literal_eval on '{value_str}' (expected: {expected_type}, allow_none: {allow_none})") # Cannot log here as it's a global function
    try:
        # Handle direct None string
        if isinstance(value_str, str) and value_str.strip().lower() == "none":
            if allow_none:
                # self.logger.debug("Evaluated 'None' string as None.")
                return None
            else:
                # self.logger.warning(f"Disallowed 'None' string encountered for value '{value_str}'.")
                raise ValueError("None is not allowed for this parameter.")

        # Evaluate other literals using ast.literal_eval, which safely evaluates strings to Python literals
        val = ast.literal_eval(value_str)
        # self.logger.debug(f"ast.literal_eval result: {val} (type: {type(val)})")

        # Type checking for single values. A ``list`` result means the string
        # was a list literal, which is validated element-wise by the caller.
        if (
            expected_type == "int"
            and not isinstance(val, int)
            and not isinstance(val, list)
        ):
            raise TypeError(f"Expected an integer, got {type(val)}")
        # numbers.Number lets an int stand in for a float
        if (
            expected_type == "float"
            and not isinstance(val, numbers.Number)
            and not isinstance(val, list)
        ):
            raise TypeError(f"Expected a float, got {type(val)}")
        if (
            expected_type == "str"
            and not isinstance(val, str)
            and not isinstance(val, list)
        ):
            raise TypeError(f"Expected a string, got {type(val)}")

        # Check for None if not allowed (after evaluation)
        if val is None and not allow_none:
            # self.logger.warning(f"Disallowed None value encountered after evaluation for '{value_str}'.")
            raise ValueError("None is not allowed for this parameter.")

        # self.logger.debug(f"Successfully evaluated '{value_str}' to: {val}")
        return val
    except (ValueError, SyntaxError, TypeError) as e:
        # self.logger.error(f"Evaluation failed for '{value_str}': {e}")
        raise ValueError(f"Invalid input format '{value_str}': {e}")


def calculate_eta(total_work, completed_work, time_elapsed) -> float | str:
    """
    Calculate the estimated time remaining (eta) in seconds.

    Parameters:
        total_work (float or int): Total amount of work to be done (e.g., bytes, tasks).
        completed_work (float or int): Amount of work already completed.
        time_elapsed (float): Time elapsed so far in seconds.

    Returns:
        float: Estimated time remaining in seconds, or 0.0 if work is complete.
        None: If eta cannot be estimated due to insufficient data.

    Raises:
        ValueError: If total_work is not positive, or if completed_work or time_elapsed is negative.
    """
    # Input validation
    if total_work <= 0:
        raise ValueError("total_work must be positive")
    if completed_work < 0:
        raise ValueError("completed_work cannot be negative")
    if time_elapsed < 0:
        raise ValueError("time_elapsed cannot be negative")

    # If work is complete or overdone, no time remains
    if completed_work >= total_work:
        return 0.0

    # If no work is done or no time has elapsed, eta cannot be estimated
    if completed_work == 0 or time_elapsed == 0:
        return "Estimated..."

    # Calculate eta: time_elapsed * (remaining_work / completed_work)
    return time_elapsed * (total_work - completed_work) / completed_work


def get_colored_logs(lines=100, log_dir="logs"):
    """
    Retrieve logs and add color based on log level.

    This function reads the most recent log file from the specified directory,
    retrieves the specified number of lines, and applies HTML color formatting
    based on the log level of each line.
    Parameters:
    -----------
    lines : int, optional
        The number of lines to retrieve from the log files. Default is 100.
    log_dir : str, optional
        The directory where log files are stored. Default is 'logs'.

    Returns:
    --------
    str
        A string containing the colored log lines in HTML format.
        If the logs directory is not found, returns an error message in red.
        If no log files are available, returns a warning message in yellow.
        If an error occurs during processing, returns an error message in red.
    """
    try:
        if not os.path.exists(log_dir):
            return "<span style='color: red'>Logs directory not found</span>"

        log_files = [
            os.path.join(log_dir, f)
            for f in os.listdir(log_dir)
            if f.endswith(".log")
        ]
        if not log_files:
            return "<span style='color: yellow'>No log files available</span>"

        latest_file = max(log_files, key=os.path.getmtime)

        with open(latest_file, "r", encoding="utf-8") as f:
            content = f.readlines()[-lines:]
            colored_lines = []
            for line in content:
                # Add color based on log level
                if "ERROR" in line:
                    colored_lines.append(
                        f"<span style='color: #ff4b4b'>{line}</span>"
                    )
                elif "WARNING" in line:
                    colored_lines.append(
                        f"<span style='color: #faca2b'>{line}</span>"
                    )
                elif "INFO" in line:
                    # Changed INFO to white
                    colored_lines.append(
                        f"<span style='color: #FFFFFF'>{line}</span>"
                    )
                elif "DEBUG" in line:
                    # Changed DEBUG to light blue
                    colored_lines.append(
                        f"<span style='color: #4DCFFF'>{line}</span>"
                    )
                else:
                    colored_lines.append(
                        f"<span style='color: white'>{line}</span>"
                    )
            return "".join(colored_lines)
    except Exception as e:
        return f"<span style='color: red'>Error reading logs: {e!s}</span>"


def logger_init(log_dir="logs"):
    """
    Initialize and configure the logger for both file and console output.

    This function sets up logging to both a file and the console. It creates
    a log directory if it doesn't exist, configures file logging with rotation
    and retention policies, and sets up console logging with color output.

    Parameters:
    -----------
    log_dir : str, optional
        The directory where log files will be stored. Default is 'logs'.

    Returns:
    --------
    None
    """
    # Create logs directory if not exists
    os.makedirs(log_dir, exist_ok=True)

    logger.remove()

    # Configure file logging
    logger.add(
        f"{log_dir}"
        + "/{time:YYYY-MM-DD}.log",  # Now in logs folder with date pattern
        rotation="00:00",
        retention="1 week",
        level="DEBUG",
        enqueue=True,
        compression="zip",  # Optional: compress rotated files
    )

    logger.add(
        sys.stderr,
        level="DEBUG",
        colorize=True,
    )


CACHE_FOLDER = ".cache"


def generate_cache_filename(cache_dir, func, *args, **kwargs):
    """
    Generate a unique cache filename based on the function name, arguments, and keyword arguments.

    This function takes a function object, positional arguments, and keyword arguments as input.
    It combines the function name, arguments, and keyword arguments into a tuple, serializes the tuple,
    and computes the SHA-256 hash of the serialized data. The hash is then used to generate a unique
    filename with a '.pickle' extension. The filename is returned as a string.

    ``cache_dir`` is only joined onto the result, never hashed, so the same
    call keys to the same filename on whichever disk the cache lives.

    Parameters
    ----------
    cache_dir (str): Directory the cache file belongs in.
    func (function): The function object for which the cache filename is being generated.
    *args (tuple): Positional arguments passed to the function.
    **kwargs (dict): Keyword arguments passed to the function.

    Returns:
    str: A unique cache filename based on the function name, arguments, and keyword arguments.
    """
    combined_data = (func.__name__, args, kwargs)
    serialized_data = pickle.dumps(combined_data)
    hash_object = hashlib.sha256(serialized_data)
    filename = f"{func.__name__}^" + hash_object.hexdigest() + ".pickle"
    return os.path.join(cache_dir, filename)


def parquet_signature(path: str) -> tuple:
    """Sizes and mtimes of a parquet file, or of a folder of part files.

    A cheap cache key that changes when the data does: a rebuilt dataset or
    a re-saved run keys differently without anyone clearing a cache.
    """
    if os.path.isdir(path):
        entries = sorted(
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.endswith(".parquet")
        )
    else:
        entries = [path]
    return tuple(
        (entry, os.stat(entry).st_size, os.stat(entry).st_mtime_ns)
        for entry in entries
    )


@st.cache_resource(max_entries=2, ttl="30m", show_spinner="Reading parquet…")
def _read_parquet_cached(path: str, signature: tuple):
    del signature  # only part of the cache key
    import polars as pl

    return pl.read_parquet(path)


def read_parquet_cached(path: str):
    """``pl.read_parquet(path)``, read once per file version, not per rerun.

    For pages that load a whole table and then slice it by widgets: every
    interaction used to re-read the file, which for a ResultsSaver run is
    gigabytes. ``cache_resource`` returns the same frame without a copy
    (polars frames are immutable), keyed on :func:`parquet_signature`; the
    TTL lets an idle page give the memory back.
    """
    return _read_parquet_cached(path, parquet_signature(path))


def cache_result(reset=False, cache_dir=None):
    """
    A decorator function that caches the results of a function and stores them in a cache file.

    Parameters
    ----------
    reset (bool): If True, the cache file will be deleted and the function will be executed again.
                If False (default), the function will attempt to load the result from the cache file.
    cache_dir (str): Where the cache files live. None resolves at call time
                from the scratch root (see :func:`resolve_cache_dir`), so a
                plugin can hand its own choice down to a loky worker through
                the per-config payload.

    Returns:
    function: The decorated function, which will either return the cached result or execute the function
            and store the result in the cache file.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            folder = cache_dir or resolve_cache_dir()
            cache_file = generate_cache_filename(folder, func, *args, **kwargs)
            if not reset:
                try:
                    with open(cache_file, "rb") as file:
                        cached_data = pickle.load(file)
                    cached_result = cached_data
                    return cached_result
                except (OSError, pickle.PickleError, EOFError):
                    pass
            result = func(*args, **kwargs)
            cached_data = result
            os.makedirs(folder, exist_ok=True)
            with open(cache_file, "wb") as file:
                pickle.dump(cached_data, file)

            return result

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# OUTPUT STORAGE ROOTS
# ---------------------------------------------------------------------------
# Every plugin that spills results to disk writes under a *root*, which used to
# be the repo-local ``output_data/`` unconditionally. The box also carries a
# second SSD, so the root is a per-plugin choice now. Only the root moves: the
# ``plugins/<plugin>/<timestamp>/`` layout underneath is untouched, so a run
# reads back the same way wherever it landed.

PROJECT_STORAGE_ROOT = "output_data"
DATA_DISK_MOUNT = "/mnt/data"
DATA_DISK_STORAGE_ROOT = os.path.join(
    DATA_DISK_MOUNT, "streamflex", "output_data"
)

# Label -> root. Insertion order is the order of the segmented control.
STORAGE_ROOTS: "OrderedDict[str, str]" = OrderedDict(
    (
        ("Project", PROJECT_STORAGE_ROOT),
        ("Data disk", DATA_DISK_STORAGE_ROOT),
    )
)
DEFAULT_STORAGE_LABEL = "Project"
# joblib (loky) workers have no Streamlit session. Plugins that cannot thread
# the root through their payload export it here before dispatching, the same
# trick STREAMFLEX_TIMESTAMP already uses for the run folder.
STORAGE_ROOT_ENV = "STREAMFLEX_STORAGE_ROOT"

# Scratch is a *separate* choice from results: the memmap spill (``.tmp/``)
# and the pickle cache (``.cache/``) are write-heavy and disposable, so it is
# reasonable to park them on the big disk while results stay in the project —
# or the other way round.
PROJECT_SCRATCH_ROOT = "."
DATA_DISK_SCRATCH_ROOT = os.path.join(DATA_DISK_MOUNT, "streamflex")
SCRATCH_ROOTS: "OrderedDict[str, str]" = OrderedDict(
    (
        ("Project", PROJECT_SCRATCH_ROOT),
        ("Data disk", DATA_DISK_SCRATCH_ROOT),
    )
)
DEFAULT_SCRATCH_LABEL = "Project"
SCRATCH_ROOT_ENV = "STREAMFLEX_SCRATCH_ROOT"
TMP_FOLDER = ".tmp"


def _root_status(root: str, project_root: str) -> tuple[bool, str]:
    """Shared usability check behind the storage and scratch roots.

    Anything off the project root lives on the data disk, which counts as
    usable only when something is actually mounted at :data:`DATA_DISK_MOUNT`.
    An unmounted mount point is an ordinary empty directory on the system
    drive, so files written into it would quietly fill ``/`` instead of the
    SSD.
    """
    if root == project_root:
        return True, ""
    if not os.path.ismount(DATA_DISK_MOUNT):
        return False, f"`{DATA_DISK_MOUNT}` is not mounted"
    # Probe the deepest part of the root that exists rather than creating it:
    # this runs on every rerun, and the writers already build their own
    # folders when they actually write.
    probe = os.path.abspath(root)
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    if not os.access(probe, os.W_OK):
        return False, f"`{probe}` is not writable by this user"
    return True, ""


def storage_root_status(label: str) -> tuple[bool, str]:
    """Report whether ``label``'s results root can be written to right now.

    Returns
    -------
    tuple[bool, str]
        ``(usable, reason)``; ``reason`` is empty when usable.
    """
    if label not in STORAGE_ROOTS:
        return False, f"unknown storage `{label}`"
    return _root_status(STORAGE_ROOTS[label], PROJECT_STORAGE_ROOT)


def scratch_root_status(label: str) -> tuple[bool, str]:
    """Report whether ``label``'s scratch root can be written to right now."""
    if label not in SCRATCH_ROOTS:
        return False, f"unknown scratch location `{label}`"
    return _root_status(SCRATCH_ROOTS[label], PROJECT_SCRATCH_ROOT)


def resolve_storage_root(label: str | None = None) -> str:
    """Resolve a storage label to a root path.

    ``None`` means "whatever the parent process chose" — read from
    :data:`STORAGE_ROOT_ENV`, which is how a loky worker recovers it. An
    unusable root falls back to the project root rather than writing into an
    unmounted stub.
    """
    if label is None:
        return os.environ.get(STORAGE_ROOT_ENV) or PROJECT_STORAGE_ROOT
    usable, _ = storage_root_status(label)
    return STORAGE_ROOTS[label] if usable else PROJECT_STORAGE_ROOT


def resolve_scratch_root(label: str | None = None) -> str:
    """Resolve a scratch label to a root path (see :func:`resolve_storage_root`)."""
    if label is None:
        return os.environ.get(SCRATCH_ROOT_ENV) or PROJECT_SCRATCH_ROOT
    usable, _ = scratch_root_status(label)
    return SCRATCH_ROOTS[label] if usable else PROJECT_SCRATCH_ROOT


def tmp_dir_for(scratch_root: str) -> str:
    """Memmap spill directory under ``scratch_root``."""
    return os.path.normpath(os.path.join(scratch_root, TMP_FOLDER))


def cache_dir_for(scratch_root: str) -> str:
    """``cache_result`` pickle cache under ``scratch_root``."""
    return os.path.normpath(os.path.join(scratch_root, CACHE_FOLDER))


def resolve_cache_dir(label: str | None = None) -> str:
    """Cache directory for a scratch label (default: what this process chose)."""
    return cache_dir_for(resolve_scratch_root(label))


def _usable_scratch_roots() -> list[str]:
    return [
        root
        for label, root in SCRATCH_ROOTS.items()
        if scratch_root_status(label)[0]
    ]


def all_tmp_dirs() -> list[str]:
    """Every ``.tmp`` across the usable scratch roots.

    The app's cache panel reports and clears through this: with the choice
    made per plugin, one session's scratch can be spread over both disks, and
    a single hard-coded folder would both under-report and under-delete.
    """
    return [tmp_dir_for(root) for root in _usable_scratch_roots()]


def all_cache_dirs() -> list[str]:
    """Every ``.cache`` across the usable scratch roots."""
    return [cache_dir_for(root) for root in _usable_scratch_roots()]


def rebase_storage_path(path: str, root: str) -> str:
    """Move ``path`` under ``root`` when it sits under some known root.

    Used by the path text inputs: switching the storage control should carry
    the pending output path over instead of stranding it on the old disk. A
    path outside every known root is user-chosen and left alone.
    """
    if not path:
        return path
    normalized = os.path.normpath(path)
    for known in STORAGE_ROOTS.values():
        known = os.path.normpath(known)
        if normalized == known:
            return root
        if normalized.startswith(known + os.sep):
            return os.path.join(root, os.path.relpath(normalized, known))
    return path


def _root_control(
    plugin,
    widget_manager,
    *,
    roots,
    default_label: str,
    status_fn,
    resolve_fn,
    fallback_root: str,
    widget_name: str,
    container,
    label: str,
    help_text: str | None,
    help_prefix: str,
) -> str:
    """Shared body of the storage and scratch segmented controls."""
    target = st if container is None else container
    if help_text is None:
        help_text = help_prefix + ", ".join(
            f"{name} = `{path}`" for name, path in roots.items()
        )
    choice = plugin.create_widget(
        widget_manager=widget_manager,
        widget_type=target.segmented_control,
        widget_name=widget_name,
        value_param="default",
        default_value=default_label,
        args=(label, list(roots)),
        kwargs={"help": help_text},
    )
    # A single-select segmented control returns None once the user deselects.
    if choice not in roots:
        choice = default_label
    usable, reason = status_fn(choice)
    if not usable:
        target.warning(
            f"`{choice}` is unavailable ({reason}) — using "
            f"`{fallback_root}` instead.",
            icon="⚠️",
        )
    return resolve_fn(choice)


def storage_root_control(
    plugin,
    widget_manager,
    *,
    widget_name: str = "storage_root",
    container=None,
    label: str = "Results storage",
    help_text: str | None = None,
) -> str:
    """Segmented control choosing where this plugin writes its results.

    Goes through ``plugin.create_widget`` so the choice persists across reruns
    and travels in snapshots. Returns the resolved root path, already fallen
    back to the project root if the chosen disk is not available.
    """
    return _root_control(
        plugin,
        widget_manager,
        roots=STORAGE_ROOTS,
        default_label=DEFAULT_STORAGE_LABEL,
        status_fn=storage_root_status,
        resolve_fn=resolve_storage_root,
        fallback_root=PROJECT_STORAGE_ROOT,
        widget_name=widget_name,
        container=container,
        label=label,
        help_text=help_text,
        help_prefix="Where this plugin writes results: ",
    )


def scratch_root_control(
    plugin,
    widget_manager,
    *,
    widget_name: str = "scratch_root",
    container=None,
    label: str = "Cache & scratch",
    help_text: str | None = None,
) -> str:
    """Segmented control choosing where ``.tmp``/``.cache`` live.

    Deliberately separate from :func:`storage_root_control`: scratch is
    write-heavy and disposable, so parking it on the big disk while results
    stay in the project (or the reverse) is a reasonable thing to want.

    Returns the resolved scratch *root* — pass it through
    :func:`tmp_dir_for` / :func:`cache_dir_for` for the actual folders.
    """
    return _root_control(
        plugin,
        widget_manager,
        roots=SCRATCH_ROOTS,
        default_label=DEFAULT_SCRATCH_LABEL,
        status_fn=scratch_root_status,
        resolve_fn=resolve_scratch_root,
        fallback_root=PROJECT_SCRATCH_ROOT,
        widget_name=widget_name,
        container=container,
        label=label,
        help_text=help_text,
        help_prefix=(
            "Where this plugin's memmap spill and pickle cache live: "
        ),
    )


def get_object_color(obj):
    """
    Extract color/texture information from a scene object.
    Returns RGB color as hex string.
    """
    try:
        if hasattr(obj, "radio_material") and obj.radio_material is not None:
            mat = obj.radio_material
            if hasattr(mat, "color"):
                color = mat.color
                rgb = np.array(color)
                return f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"
    except Exception:
        pass
    return "lightgray"


def radio_map_to_numpy(radio_map, metric, tx_idx, db_scale):
    """
    Convert radiomap data to numpy array with proper scaling

    :param radio_map: RadioMap instance
    :param metric: Metric to extract ('path_gain', 'rss', 'sinr')
    :param tx_idx: Transmitter index (None for max over all TXs)
    :param db_scale: Whether to convert to dB scale
    :return: numpy array of radiomap values
    """
    if metric == "path_gain":
        data = radio_map.path_gain
    elif metric == "rss":
        data = radio_map.rss
    elif metric == "sinr":
        data = radio_map.sinr
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if tx_idx is not None:
        data = data[tx_idx]
    else:
        data = dr.max(data, axis=0)

    data_np = data.numpy()

    if db_scale:
        with np.errstate(divide="ignore"):  # Ignore log10 of zero
            if metric == "rss":
                # Convert W to dBm
                data_np = 10 * np.log10(data_np * 1000)
            else:
                # Convert linear to dB
                data_np = 10 * np.log10(data_np)

    return data_np


# Max radiomap cells per axis sent to Plotly. A go.Surface / Mesh3d with
# millions of cells (small cell_size + large area) serializes a huge JSON
# payload and hangs/crashes the browser's WebGL renderer. We decimate the
# *display* grid above this cap; the full-resolution map is still used for
# RX sampling and metrics — only the on-screen trace is coarsened.
MAX_RADIOMAP_CELLS_PER_AXIS = 400


def add_planar_radiomap_to_figure(
    fig,
    radio_map,
    metric="path_gain",
    tx_idx=None,
    db_scale=True,
    vmin=None,
    vmax=None,
    colorscale="Viridis",
    show_colorbar=True,
    opacity=0.8,
):
    """
    Add a planar radiomap to a Plotly figure.
    """
    if not isinstance(radio_map, PlanarRadioMap):
        raise TypeError("This function only works with PlanarRadioMap")

    data_np = radio_map_to_numpy(radio_map, metric, tx_idx, db_scale)
    num_cells_y, num_cells_x = data_np.shape

    # Decimate the display grid so the browser can render it (see the
    # MAX_RADIOMAP_CELLS_PER_AXIS note). Strided slicing keeps the full
    # spatial extent; only the on-screen resolution is reduced.
    step_y = max(1, int(np.ceil(num_cells_y / MAX_RADIOMAP_CELLS_PER_AXIS)))
    step_x = max(1, int(np.ceil(num_cells_x / MAX_RADIOMAP_CELLS_PER_AXIS)))
    if step_x > 1 or step_y > 1:
        data_np = data_np[::step_y, ::step_x]
        num_cells_y, num_cells_x = data_np.shape

    try:
        bbox = radio_map.measurement_surface.bbox()
        x_coords = np.linspace(bbox.min.x, bbox.max.x, num_cells_x)
        y_coords = np.linspace(bbox.min.y, bbox.max.y, num_cells_y)
        x, y = np.meshgrid(x_coords, y_coords)
        z_val = bbox.center().z
        z = np.full_like(x, z_val)
    except Exception as e:
        st.error(f"Could not reconstruct radiomap grid from bbox: {e}")
        return

    finite_data = data_np[np.isfinite(data_np)]
    if vmin is None:
        vmin = np.min(finite_data) if finite_data.size > 0 else 0
    if vmax is None:
        vmax = np.max(finite_data) if finite_data.size > 0 else 1

    if db_scale:
        colorbar_title = (
            f"{metric.upper()} [dBm]"
            if metric == "rss"
            else f"{metric.upper()} [dB]"
        )
    else:
        colorbar_title = metric.upper()
    custom_data_stacked = np.stack([data_np.T], axis=-1)

    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=data_np,
            customdata=custom_data_stacked,
            cmin=vmin,
            cmax=vmax,
            colorscale=colorscale,
            opacity=opacity,
            showscale=show_colorbar,
            name=f"RadioMap ({metric})",
            hovertemplate=(
                f"{metric}: "
                + "%{customdata[0]:.2f}<br>X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m<br><extra></extra>"
            ),
            colorbar={"title": colorbar_title, "x": 1.02}
            if show_colorbar
            else None,
            showlegend=True,
            legendgroup="Radiomap",
            hoverinfo="name",
        )
    )


def add_mesh_radiomap_to_figure(
    fig,
    radio_map,
    metric="path_gain",
    tx_idx=None,
    db_scale=True,
    vmin=None,
    vmax=None,
    colorscale="Viridis",
    show_colorbar=True,
    opacity=0.8,
):
    """
    Add a mesh-based radiomap to a Plotly figure
    """
    if not isinstance(radio_map, MeshRadioMap):
        raise TypeError("This function only works with MeshRadioMap")

    data_np = radio_map_to_numpy(radio_map, metric, tx_idx, db_scale)
    mesh = radio_map.measurement_surface
    vertices = mesh.vertex_positions_buffer().numpy()
    faces = mesh.faces_buffer().numpy()

    x, y, z = vertices[0::3], vertices[1::3], vertices[2::3]
    i, j, k = faces[0::3], faces[1::3], faces[2::3]

    # Add slight visual offset to avoid z-fighting with the actual ground object
    z = z + 0.05

    finite_data = data_np[np.isfinite(data_np)]
    if vmin is None:
        vmin = np.min(finite_data) if finite_data.size > 0 else 0
    if vmax is None:
        vmax = np.max(finite_data) if finite_data.size > 0 else 1

    if db_scale:
        colorbar_title = (
            f"{metric.upper()} [dBm]"
            if metric == "rss"
            else f"{metric.upper()} [dB]"
        )
    else:
        colorbar_title = metric.upper()

    # Map per-triangle data to per-vertex data for smooth shading.
    # Vectorized scatter-add (np.add.at handles repeated vertex indices)
    # replaces a Python double loop that was O(faces) and stalled on fine
    # meshes.
    num_vertices = len(x)
    vertex_values = np.zeros(num_vertices)
    vertex_counts = np.zeros(num_vertices)

    for idx in (i, j, k):
        np.add.at(vertex_values, idx, data_np)
        np.add.at(vertex_counts, idx, 1)

    vertex_values = np.divide(
        vertex_values,
        vertex_counts,
        out=np.full_like(vertex_values, np.nan),
        where=vertex_counts > 0,
    )

    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            intensity=vertex_values,
            cmin=vmin,
            cmax=vmax,
            colorscale=colorscale,
            opacity=opacity,
            showscale=show_colorbar,
            name=f"RadioMap ({metric})",
            hovertemplate=(
                f"{metric}: %{{intensity:.2f}}<br>"
                "X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m<br>"
                "<extra></extra>"
            ),
            colorbar={"title": colorbar_title, "x": 1.02}
            if show_colorbar
            else None,
            showlegend=True,
            legendgroup="Radiomap",
            hoverinfo="name",
        )
    )


def _weld_mesh(vertices, faces):
    """Losslessly merge exactly coincident vertices for Plotly.

    Sionna meshes often repeat a vertex for every adjacent triangle.  Plotly
    only needs one copy plus the remapped indices, so exact welding reduces
    the websocket payload without moving a point or removing a triangle.
    Unreferenced vertices are omitted as well.
    """
    points = np.asarray(vertices).reshape(-1, 3)
    face_indices = np.asarray(faces).reshape(-1)
    if face_indices.size == 0:
        empty_points = points[:0]
        empty_indices = np.empty((0, 3), dtype=np.uint32)
        return (
            empty_points[:, 0],
            empty_points[:, 1],
            empty_points[:, 2],
            empty_indices[:, 0],
            empty_indices[:, 1],
            empty_indices[:, 2],
        )

    # Select referenced points before np.unique so stale/unreferenced buffer
    # entries do not get serialized. Equality is exact: there is deliberately
    # no rounding, quantization, decimation, or degenerate-face filtering.
    referenced = points[face_indices.astype(np.int64, copy=False)]
    unique, inverse = np.unique(referenced, axis=0, return_inverse=True)
    remapped = inverse.astype(np.uint32, copy=False).reshape(-1, 3)
    return (
        unique[:, 0],
        unique[:, 1],
        unique[:, 2],
        remapped[:, 0],
        remapped[:, 1],
        remapped[:, 2],
    )


# Public alias: the welder is useful to any renderer, not just the one
# below.
def weld_mesh(vertices, faces):
    """Losslessly merge coincident vertices; see :func:`_weld_mesh`."""
    return _weld_mesh(vertices, faces)


@st.cache_resource(show_spinner=False, ttl=300, refresh_mode="background")
def load_scene_cached(scene_path: str, merge_shapes: bool = True):
    """Load a Sionna scene, picking up on-disk rebuilds without a stall.

    Keyed on the path alone, so every page shares one loaded copy. The
    ttl exists for the case that used to need a manual cache clear: a
    scene rebuilt by the OSM builder while the app is open. With
    ``refresh_mode="background"`` the expired entry is still served
    immediately and the reload happens off the script thread, so nobody
    waits for it -- note this only ever helps an entry that already
    exists; the first load of a scene is still synchronous.
    """
    from sionna.rt import load_scene

    return load_scene(scene_path, merge_shapes=merge_shapes)


# Coarse mesh categories, by the naming the OSM scene builder writes
# (`building_<id>`, `roof_<id>`, `surface_landuse`, `tree_crowns`, ...).
# A scene that names its shapes some other way -- Sionna's own munich or
# etoile, an indoor box -- lands entirely in "other", which is harmless:
# the layer filter then simply has one entry.
SCENE_LAYERS = (
    "buildings",
    "terrain",
    "roads",
    "surfaces",
    "trees",
    "windows",
    "vehicles",
    "other",
)


def scene_layer_of(name: str) -> str:
    """Coarse category of one scene shape."""
    low = str(name).lower()
    if low.startswith(("building_", "roof_")):
        return "buildings"
    if low.startswith("window"):
        return "windows"
    if low.startswith("tree"):
        return "trees"
    if low.startswith(("road", "bridge")):
        return "roads"
    # Parked cars are inserted into a loaded scene rather than written into
    # it (pages/osm_scene/cars.py), always as one merged shape.
    if low.startswith(("parked_cars", "car_")):
        return "vehicles"
    if low.startswith("surface"):
        return "surfaces"
    if low.startswith(("terrain", "ground")):
        return "terrain"
    return "other"


def scene_content_key(scene) -> str:
    """Cheap, stable identity of a loaded scene: shape names + face counts.

    Used as the cache key for derived geometry. Mitsuba reports face
    counts without touching the buffers, so this costs nothing even on a
    2000-shape city, and a rebuilt scene keys differently by itself --
    no ttl or manual invalidation involved.
    """
    parts = []
    for name, obj in scene.objects.items():
        try:
            parts.append((name, int(obj.mi_mesh.face_count())))
        except Exception:
            parts.append((name, -1))
    parts.sort()
    return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()


def scene_layer_counts(scene) -> dict:
    """`{layer: triangles}` for the layers actually present in the scene."""
    counts: dict[str, int] = {}
    for name, obj in scene.objects.items():
        try:
            n = int(obj.mi_mesh.face_count())
        except Exception:
            continue
        layer = scene_layer_of(name)
        counts[layer] = counts.get(layer, 0) + n
    return counts


def _object_material(obj):
    """`(material name, colour)` of one scene object."""
    try:
        name = str(getattr(obj.radio_material, "name", "") or "material")
    except Exception:
        name = "material"
    return name, get_object_color(obj)


def _build_scene_mesh_parts(scene, hidden_layers):
    """Welded scene geometry, one entry per radio material.

    Returns ``([(material, colour, x, y, z, i, j, k)], info)``. Three
    things this does that a per-object dump does not:

    * **Grouped by radio material**, not by shape. Plotly pays a fixed
      cost per trace and an LOD3 city has ~2000 shapes; grouping brings
      that to a dozen and makes a legend meaningful, since the colour
      *is* the material.
    * **Layer filtering.** On an OSM city the land-cover surfaces alone
      are two thirds of the triangles and none of them help place a
      radio device. Layers come from shape names, so this only bites on
      a scene loaded with ``merge_shapes=False`` -- merging throws the
      names away.
    * **Welding only.** Sionna repeats a vertex per adjacent triangle,
      so welding is a free ~25% cut with nothing moved. There is
      deliberately no decimation: snapping vertices to a lattice does
      shrink the payload, but walls collapse into each other and the
      scene falls apart visually long before it gets small.

    Pure numpy and Mitsuba buffer reads, no Streamlit calls: it has to
    be safe to run on a worker thread (see :func:`scene_mesh_parts`).
    """
    hidden = set(hidden_layers or ())
    groups: dict = {}
    for name, obj in scene.objects.items():
        if scene_layer_of(name) in hidden:
            continue
        try:
            mesh = obj.mi_mesh
            vertices = mesh.vertex_positions_buffer().numpy().reshape(-1, 3)
            faces = mesh.faces_buffer().numpy().reshape(-1, 3)
        except Exception:
            continue
        if not len(faces):
            continue
        groups.setdefault(_object_material(obj), []).append((vertices, faces))

    parts = []
    triangles = 0
    vertices_full = 0
    vertices_welded = 0
    for (mat_name, color), group in sorted(groups.items()):
        offset = 0
        verts, tris = [], []
        for vertices, faces in group:
            verts.append(vertices)
            tris.append(faces + offset)
            offset += len(vertices)
        x, y, z, i, j, k = _weld_mesh(np.vstack(verts), np.vstack(tris))
        if len(i) == 0:
            continue
        triangles += len(i)
        vertices_welded += len(x)
        vertices_full += sum(len(v) for v, _ in group)
        parts.append(
            (
                mat_name,
                color,
                x.astype(np.float32),
                y.astype(np.float32),
                z.astype(np.float32),
                i.astype(np.uint32),
                j.astype(np.uint32),
                k.astype(np.uint32),
            )
        )
    info = {
        "geometry_reduction": "exact_vertex_welding",
        "triangles": triangles,
        "vertices": vertices_welded,
        "vertices_full": vertices_full,
        "traces": len(parts),
        "hidden_layers": sorted(hidden),
    }
    return parts, info


# How many (scene, hidden-layer) geometries to keep. Each runs to tens of
# megabytes of numpy, so this is a memory budget, not a speed knob.
SCENE_MESH_CACHE_ENTRIES = 6


@st.cache_resource(show_spinner=False)
def _scene_mesh_store():
    """Executor + results shared by every session on this server.

    Not `st.cache_resource` on the build itself: the worker thread has no
    ScriptRunContext, and reaching into Streamlit's cache from off the
    script thread is not something to rely on. The store is a plain dict
    under a lock, and the cache decorator here only makes it a singleton.
    """
    return {
        "pool": ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="scene-mesh"
        ),
        "done": OrderedDict(),
        "running": {},
        "lock": threading.Lock(),
    }


def scene_mesh_parts(scene, hidden_layers=(), background=False):
    """Welded geometry per radio material, optionally built off-thread.

    With ``background=True`` the first call submits the build to a worker
    and returns ``None``; the caller shows a placeholder and comes back.
    Welding an LOD3 city takes ~3 s, and that is 3 s during which the
    script thread would otherwise be unable to serve anything else --
    the 2D maps, the sidebar, a TX click. The work is pure numpy and
    Mitsuba buffer reads, so it releases the GIL often enough for the
    page to stay usable (worst observed stall ~0.3 s).

    What this does **not** do is make the figure arrive any sooner: the
    payload still goes over the websocket synchronously once it exists.
    It takes the build off the critical path, not the transfer.
    """
    key = (scene_content_key(scene), tuple(sorted(hidden_layers or ())))
    store = _scene_mesh_store()
    with store["lock"]:
        if key in store["done"]:
            store["done"].move_to_end(key)
            return store["done"][key]
        pending = store["running"].get(key)
        if pending is None and background:
            pending = store["pool"].submit(
                _build_scene_mesh_parts, scene, key[1]
            )
            store["running"][key] = pending

    try:
        if pending is not None:
            if background and not pending.done():
                return None
            result = pending.result()
        else:
            result = _build_scene_mesh_parts(scene, key[1])
    except Exception:
        # Drop the failed job so the next rerun retries instead of
        # re-raising the same stale future forever.
        with store["lock"]:
            store["running"].pop(key, None)
        raise

    with store["lock"]:
        store["running"].pop(key, None)
        store["done"][key] = result
        while len(store["done"]) > SCENE_MESH_CACHE_ENTRIES:
            store["done"].popitem(last=False)
    return result


def scene_mesh_traces(
    scene,
    hidden_layers=(),
    opacity: float = 0.5,
    material_legend: bool = False,
    background: bool = False,
):
    """``Mesh3d`` traces for a scene backdrop, one per radio material.

    Returns ``(traces, info)``, or ``None`` when ``background=True`` and
    the geometry is still being welded. The geometry is cached by
    :func:`scene_mesh_parts`; only the lightweight trace wrappers are
    rebuilt per call, which is why opacity and the legend flag are
    applied here and not baked into that cache.
    """
    built = scene_mesh_parts(scene, hidden_layers, background=background)
    if built is None:
        return None
    parts, info = built
    traces = [
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            opacity=float(opacity),
            color=color,
            name=mat_name,
            showlegend=bool(material_legend),
            legendgroup="materials",
            legendgrouptitle_text="Materials",
            hoverinfo="name",
        )
        for mat_name, color, x, y, z, i, j, k in parts
    ]
    return traces, info


def scene_display_controls(
    scene,
    widget_key: str = "scene",
    container=None,
    default_hidden=("surfaces", "windows", "trees"),
):
    """Shared "what to draw" controls for any 3D scene figure.

    Returns ``(hidden_layers, material_legend)``.

    The layer selection is keyed per scene: a keyed widget outlives the
    scene it was set on, and the layers of the next scene are different
    ones -- carrying the choice over means either a Streamlit error on a
    missing option or, worse, silently keeping an empty selection and
    never applying the defaults to the new scene. The legend toggle is a
    user preference and does carry over.
    """
    host = container if container is not None else st
    counts = scene_layer_counts(scene)
    options = [layer for layer in SCENE_LAYERS if layer in counts]
    fingerprint = scene_content_key(scene)[:12]
    hide_key = f"{widget_key}_{fingerprint}_hidden_layers"
    legend_key = f"{widget_key}_material_legend"

    if len(options) <= 1:
        # `merge_shapes=True` replaces every shape name with `no-name-N`,
        # so there is nothing left to sort into layers. Say so instead of
        # offering a filter with one bucket in it.
        host.caption(
            "Layer filter needs a scene loaded with `merge_shapes=False` "
            "— this one has no per-shape names left."
        )
        hidden: list = []
    else:
        if hide_key not in st.session_state:
            st.session_state[hide_key] = [
                layer for layer in default_hidden if layer in options
            ]
        hidden = host.multiselect(
            "Hide layers",
            options,
            key=hide_key,
            format_func=lambda layer: f"{layer} ({counts.get(layer, 0):,})",
            help=(
                "Plotly re-sends every triangle on every rerun, so hiding "
                "the bulky layers is what makes a big scene usable."
            ),
        )
    legend = host.checkbox(
        "Material legend",
        key=legend_key,
        value=st.session_state.get(legend_key, False),
        help="Legend by radio material; entries toggle the mesh on the plot.",
    )
    return tuple(hidden or ()), bool(legend)


#: Sionna's own path palette, exactly
#: (sionna.rt.constants: LOS/SPECULAR/DIFFUSE/REFRACTION/DIFFRACTION_COLOR).
#: Module level because the 3D scene view is not the only thing that colours
#: an interaction type: the ray-analysis charts key off the same table, so a
#: ray drawn light-blue in the scene is a light-blue bar there too.
SIONNA_PATH_COLORS = {
    "los": "rgb(128,128,128)",  # (0.5, 0.5, 0.5)
    "specular": "rgb(153,153,255)",  # (0.6, 0.6, 1.0)
    "diffuse": "rgb(153,255,153)",  # (0.6, 1.0, 0.6)
    "refraction": "rgb(255,153,153)",  # (1.0, 0.6, 0.6)
    "diffraction": "rgb(153,0,153)",  # (0.6, 0.0, 0.6)
}


def render_sionna_scene_plotly(
    scene,
    paths=None,
    show_paths=True,
    show_legend=True,
    show_objects=True,
    building_opacity=0.5,
    selected_tx_names=None,
    selected_rx_names=None,
    global_path_opacity=None,
    specular_opacity=None,
    diffuse_opacity=None,
    refraction_opacity=None,
    diffraction_opacity=None,
    radio_map=None,
    rm_metric="path_gain",
    rm_tx=None,
    rm_db_scale=True,
    rm_vmin=None,
    rm_vmax=None,
    rm_colorscale="Viridis",
    rm_show_colorbar=True,
    rm_opacity=0.8,
    color_paths_by_segment=True,
    opacity_from_power=False,
    show_segment_toggle=True,
    widget_key="scene",
    hidden_layers=None,
    material_legend=False,
    show_layer_controls=False,
    controls_container=None,
    background=False,
    power_time_index=None,
) -> go.Figure:
    """
    Render a Sionna scene using Plotly in Streamlit.

    The scene backdrop comes from :func:`scene_mesh_traces`: welded
    losslessly, grouped by radio material, and filtered by layer. This is
    the one place scene geometry is turned into Plotly traces -- every
    page and plugin that draws a 3D scene goes through here so they share
    the caching, the layer filter and the material legend.

    ``hidden_layers`` names layers to leave out (see :data:`SCENE_LAYERS`).
    Pass ``show_layer_controls=True`` to have the multiselect and the
    legend toggle rendered here instead, into ``controls_container``.

    With ``background=True`` the scene mesh is welded on a worker thread:
    the figure comes back without a backdrop and with
    ``meta["pending"] = True`` until it is ready, so the caller can draw
    everything else and poll instead of blocking the script.
    """
    fig = go.Figure()

    path_colors = dict(SIONNA_PATH_COLORS)
    path_widths = {"los": 4}

    if show_objects:
        if show_layer_controls:
            hidden_layers, material_legend = scene_display_controls(
                scene, widget_key=widget_key, container=controls_container
            )
        built = scene_mesh_traces(
            scene,
            hidden_layers=hidden_layers,
            opacity=building_opacity,
            material_legend=material_legend,
            background=background,
        )
        if built is None:
            fig.update_layout(meta={"pending": True})
        else:
            traces, info = built
            for trace in traces:
                fig.add_trace(trace)
            fig.update_layout(meta={**info, "pending": False})

    if radio_map is not None:
        try:
            if isinstance(radio_map, PlanarRadioMap):
                add_planar_radiomap_to_figure(
                    fig,
                    radio_map,
                    rm_metric,
                    rm_tx,
                    rm_db_scale,
                    rm_vmin,
                    rm_vmax,
                    rm_colorscale,
                    rm_show_colorbar,
                    rm_opacity,
                )
            elif isinstance(radio_map, MeshRadioMap):
                add_mesh_radiomap_to_figure(
                    fig,
                    radio_map,
                    rm_metric,
                    rm_tx,
                    rm_db_scale,
                    rm_vmin,
                    rm_vmax,
                    rm_colorscale,
                    rm_show_colorbar,
                    rm_opacity,
                )
        except Exception as e:
            st.error(f"Error rendering radiomap: {e!s}")
            st.code(traceback.format_exc())

    tx_names_to_render = (
        list(scene.transmitters.keys())
        if selected_tx_names is None else selected_tx_names
    )
    for tx_name in tx_names_to_render:
        if tx_name in scene.transmitters:
            tx = scene.transmitters[tx_name]
            pos = np.asarray(tx.position.numpy()).flatten()
            fig.add_trace(
                go.Scatter3d(
                    x=[float(pos[0])],
                    y=[float(pos[1])],
                    z=[float(pos[2])],
                    mode="markers",
                    marker={"size": 8, "color": "red", "symbol": "circle"},
                    name=f"TX: {tx_name}",
                    showlegend=show_legend,
                    legendgroup="transmitters",
                    legendgrouptitle_text="Transmitters",
                )
            )

    rx_names_to_render = (
        list(scene.receivers.keys())
        if selected_rx_names is None else selected_rx_names
    )
    for rx_name in rx_names_to_render:
        if rx_name in scene.receivers:
            rx = scene.receivers[rx_name]
            pos = np.asarray(rx.position.numpy()).flatten()
            fig.add_trace(
                go.Scatter3d(
                    x=[float(pos[0])],
                    y=[float(pos[1])],
                    z=[float(pos[2])],
                    mode="markers",
                    marker={"size": 8, "color": "green", "symbol": "circle"},
                    name=f"RX: {rx_name}",
                    showlegend=show_legend,
                    legendgroup="receivers",
                    legendgrouptitle_text="Receivers",
                )
            )

    if show_paths and paths is not None:
        # The per-segment coloring toggle lives here so every caller exposes it
        # consistently without re-implementing the widget. Callers rendering
        # more than one figure must pass distinct ``widget_key`` values.
        if show_segment_toggle:
            try:
                color_paths_by_segment = st.checkbox(
                    "Per-segment path colors (Sionna style)",
                    value=color_paths_by_segment,
                    key=f"{widget_key}_color_paths_by_segment",
                    help=(
                        "Color each path segment individually like Sionna: the "
                        "incident segment (TX -> first interaction) is gray "
                        "(LoS), each later segment takes its interaction color. "
                        "Off = whole path colored by its first interaction type."
                    ),
                )
            except Exception:
                pass

        add_paths_to_figure(
            fig,
            scene,
            paths,
            path_colors,
            path_widths,
            show_legend,
            selected_tx_names,
            selected_rx_names,
            global_path_opacity,
            specular_opacity,
            diffuse_opacity,
            refraction_opacity,
            diffraction_opacity,
            color_paths_by_segment,
            opacity_from_power,
            power_time_index,
        )

    fig.update_layout(
        scene={
            "xaxis_title": "X (m)",
            "yaxis_title": "Y (m)",
            "zaxis_title": "Z (m)",
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.5}},
        },
        title="Sionna Scene Visualization",
        showlegend=show_legend,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
        height=700,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
    )
    return fig


def _path_power_db(paths, time_index=None):
    """Per-ray received power in dB, indexed ``[rx, tx, path]``.

    ``Paths.a`` is a (real, imag) pair shaped
    ``[rx, rx_ant, tx, tx_ant, path]`` (+ a time axis when the run has one),
    or ``[rx, tx, path]`` for a synthetic array. Antennas are summed (the
    ray's power at the receiver, not per element). When a time step is
    selected, use that step so the 3D view agrees with the CIR plot;
    otherwise average over time. Returns ``None`` when the
    solver did not produce amplitudes — the caller then falls back to flat
    opacity rather than failing the render.
    """
    try:
        a = paths.a
        if isinstance(a, (tuple, list)):
            if len(a) != 2:
                return None
            real, imag = a
            arr = np.asarray(real.numpy()) + 1j * np.asarray(imag.numpy())
        else:
            arr = np.asarray(a.numpy() if hasattr(a, "numpy") else a)
    except Exception:
        return None
    power = np.abs(arr) ** 2
    if power.ndim == 6:  # [rx, rx_ant, tx, tx_ant, path, time]
        power = (
            power[..., min(max(int(time_index), 0), power.shape[5] - 1)]
            if time_index is not None else power.mean(axis=5)
        )
    if power.ndim == 5:  # [rx, rx_ant, tx, tx_ant, path]
        power = power.sum(axis=(1, 3))
    elif power.ndim == 4:  # synthetic array + time
        power = (
            power[..., min(max(int(time_index), 0), power.shape[3] - 1)]
            if time_index is not None else power.mean(axis=3)
        )
    if power.ndim != 3:
        return None
    return 10.0 * np.log10(np.maximum(power, 1e-30))


#: Opacity levels the power mapping is snapped to. Every ray drawn at the
#: same level shares one trace, so this is what bounds the trace count when
#: brightness varies per ray — 16 steps are more than the eye resolves
#: through a transparent line.
POWER_OPACITY_LEVELS = 16

#: Minimum opacity fraction for weak rays, so every valid ray remains in the
#: scene even when its received amplitude is far below the strongest one.
MIN_POWER_OPACITY_FRACTION = 0.02


def add_paths_to_figure(
    fig,
    scene,
    paths,
    path_colors,
    path_widths,
    show_legend,
    selected_tx_names=None,
    selected_rx_names=None,
    global_path_opacity=None,
    specular_opacity=None,
    diffuse_opacity=None,
    refraction_opacity=None,
    diffraction_opacity=None,
    color_paths_by_segment=True,
    opacity_from_power=False,
    power_time_index=None,
):
    """
    Add propagation paths to the Plotly figure.

    Every drawn segment goes into a **batched** trace: one Scatter3d per
    (interaction type, opacity, width), with the polylines separated by
    ``None``. Plotly pays a fixed cost per trace both in the payload and in
    every redraw, and a city solve returns tens of thousands of rays — one
    trace each (or one per *segment* in Sionna-style colouring) is what makes
    the view unusable long before the geometry does. Batching preserves every
    valid ray in either brightness mode; only its opacity and width change.

    color_paths_by_segment:
        If True (default), each path is split into segments colored
        individually like Sionna's own renderer: the incident segment
        (TX -> first interaction) is gray (LoS color) and every subsequent
        segment takes the color of the interaction it leaves. If False, the
        whole path takes the color of its first interaction.

    opacity_from_power:
        Scale each ray's opacity by its own received power (``|a|^2``,
        summed over antennas), so the picture ranks rays instead of drawing
        a 10 dB ray and a 150 dB weaker one identically. The per-type
        sliders stay the ceiling: the strongest ray on screen gets exactly
        the opacity they set. Power is normalized separately for each
        selected TX/RX link. Every valid ray stays in the scene. Opacities
        and widths use bounded levels so the trace count stays manageable.
    """
    opacity_map = {
        "specular": specular_opacity,
        "diffuse": diffuse_opacity,
        "refraction": refraction_opacity,
        "diffraction": diffraction_opacity,
    }

    def _segment_opacity(seg_type, is_true_los, path_primary_type):
        # A genuine LoS ray keeps its legacy full opacity with power mode off.
        # In power mode the global slider is a ceiling for every ray, LoS
        # included. Gray incident segments of reflected paths follow their
        # interaction type as before.
        if is_true_los:
            if opacity_from_power and global_path_opacity is not None:
                return global_path_opacity
            return 1.0
        if global_path_opacity is not None:
            return global_path_opacity
        # In per-type mode the gray incident segment has no slider of its own,
        # so it follows its path's primary interaction opacity.
        eff_type = seg_type if seg_type != "los" else path_primary_type
        configured = opacity_map.get(eff_type)
        return 1.0 if configured is None else configured

    # (type, opacity, width) -> coordinate lists. Insertion order is what
    # decides which group carries a type's legend entry.
    batches = {}

    def _emit(seg_type, opacity, width, points):
        batch = batches.get((seg_type, opacity, width))
        if batch is None:
            batch = {"x": [], "y": [], "z": []}
            batches[(seg_type, opacity, width)] = batch
        for axis, values in zip(("x", "y", "z"), zip(*points)):
            # Millimetres. Plotly ships coordinates as JSON text and a city
            # solve is ~200k of them per redraw; the full float repr spends
            # half those bytes on digits below the wavelength.
            batch[axis].extend(round(v, 3) for v in values)
            batch[axis].append(None)  # break the polyline before the next one

    try:
        vertices_np = paths.vertices.numpy()
        interactions_np = paths.interactions.numpy()
        valid_np = paths.valid.numpy()
        source_positions_np = np.stack(
            [p.numpy() for p in paths.sources], axis=1
        )
        target_positions_np = np.stack(
            [p.numpy() for p in paths.targets], axis=1
        )

        is_synthetic = paths.synthetic_array

        if is_synthetic:
            max_depth, num_rx, num_tx, num_paths, _ = vertices_np.shape
        else:
            (max_depth, num_rx, _, num_tx, _, num_paths, _) = vertices_np.shape

        tx_names = list(scene.transmitters.keys())
        rx_names = list(scene.receivers.keys())
        drawn_pairs = [
            (rx_idx, tx_idx)
            for rx_idx in range(num_rx)
            for tx_idx in range(num_tx)
            if not (
                (selected_tx_names is not None and tx_names[tx_idx] not in selected_tx_names)
                or (
                    selected_rx_names is not None
                    and rx_names[rx_idx] not in selected_rx_names
                )
            )
        ]

        # The CIR plots magnitude, so opacity follows the ray's amplitude
        # relative to the strongest path of this TX/RX pair. All valid paths
        # remain in the scene, including those far below the peak.
        power_db = (
            _path_power_db(paths, power_time_index)
            if opacity_from_power else None
        )
        power_limits = {}
        if power_db is not None:
            valid3 = valid_np[:, 0, :, 0, :] if valid_np.ndim == 5 else valid_np
            for rx_idx, tx_idx in drawn_pairs:
                values = power_db[rx_idx, tx_idx][valid3[rx_idx, tx_idx] > 0]
                values = values[np.isfinite(values)]
                if values.size:
                    power_limits[rx_idx, tx_idx] = float(values.max())

        def _power_scale(rx_idx, tx_idx, path_idx):
            """Return opacity fraction and width level for one ray."""
            peak = power_limits.get((rx_idx, tx_idx))
            if power_db is None or peak is None:
                # `power_limits` is only ever filled when the powers exist, so
                # the peak lookup already covers this — but the closure reads
                # `power_db` past the guard that built it, and a reader (or a
                # type checker) has no way to know that from here.
                return 1.0, None
            value = float(power_db[rx_idx, tx_idx, path_idx])
            if not np.isfinite(value):
                return MIN_POWER_OPACITY_FRACTION, 0.0
            amplitude_ratio = 10.0 ** (min(value - peak, 0.0) / 20.0)
            level = round(amplitude_ratio * POWER_OPACITY_LEVELS) / POWER_OPACITY_LEVELS
            scaled = MIN_POWER_OPACITY_FRACTION + (
                1.0 - MIN_POWER_OPACITY_FRACTION
            ) * level
            return scaled, level

        for rx_idx, tx_idx in drawn_pairs:
            tx_name = tx_names[tx_idx]
            rx_name = rx_names[rx_idx]

            for path_idx in range(num_paths):
                is_valid = (
                    valid_np[rx_idx, 0, tx_idx, 0, path_idx]
                    if not is_synthetic
                    else valid_np[rx_idx, tx_idx, path_idx]
                )
                if not is_valid:
                    continue

                if is_synthetic:
                    # Use the same authoritative Scene coordinates as the
                    # TX/RX markers. This is essential after motion-aware
                    # recomputation: cached/converted Paths endpoint
                    # buffers can otherwise visually diverge from devices.
                    source_pos = np.asarray(
                        scene.transmitters[tx_name].position.numpy()
                    ).reshape(-1)[:3]
                    target_pos = np.asarray(
                        scene.receivers[rx_name].position.numpy()
                    ).reshape(-1)[:3]
                else:
                    source_pos = source_positions_np[
                        tx_idx * scene.tx_array.array_size
                    ]
                    target_pos = target_positions_np[
                        rx_idx * scene.rx_array.array_size
                    ]

                # Build the polyline source -> interaction vertices -> target.
                #
                # Interaction vertices are selected using the `interactions`
                # array (InteractionType.NONE == 0 marks "no interaction"),
                # NOT by testing the vertex coordinates. A genuine interaction
                # point can legitimately lie at the origin (0,0,0), and depths
                # past the last interaction carry leftover non-zero garbage,
                # so a value-based test both drops real vertices and keeps
                # spurious ones. This mirrors Sionna's own renderer
                # (sionna.rt.preview.Previewer.plot_paths), which breaks at
                # the first NONE interaction.
                path_type = "los"
                path_coords = [source_pos]
                vertex_types = []  # interaction type at each vertex
                for depth in range(max_depth):
                    interaction = int(
                        interactions_np[
                            depth, rx_idx, 0, tx_idx, 0, path_idx
                        ]
                        if not is_synthetic
                        else interactions_np[depth, rx_idx, tx_idx, path_idx]
                    )
                    if interaction == 0:  # InteractionType.NONE
                        break
                    type_name = get_path_type_name(interaction)
                    if path_type == "los":
                        path_type = type_name
                    vertex = (
                        vertices_np[depth, rx_idx, 0, tx_idx, 0, path_idx]
                        if not is_synthetic
                        else vertices_np[depth, rx_idx, tx_idx, path_idx]
                    )
                    path_coords.append(vertex)
                    vertex_types.append(type_name)
                path_coords.append(target_pos)

                coords = [np.asarray(p).tolist() for p in path_coords]
                # Only a true LoS ray (TX -> RX with no interactions) is
                # drawn thick. Gray *incident* segments of reflected paths
                # use the normal width, like every other interaction.
                is_true_los = len(vertex_types) == 0
                scale, power_level = _power_scale(rx_idx, tx_idx, path_idx)
                if power_level is None:
                    width = path_widths.get("los", 4) if is_true_los else 1
                elif is_true_los:
                    width = max(path_widths.get("los", 4), 2 + round(6 * power_level))
                else:
                    width = 1 + round(5 * power_level)

                if not color_paths_by_segment:
                    # One polyline per path, colored by its first interaction.
                    _emit(
                        path_type,
                        _segment_opacity(path_type, is_true_los, path_type)
                        * scale,
                        width,
                        coords,
                    )
                else:
                    # One polyline per segment, colored like Sionna: the
                    # incident segment (TX -> first vertex) is LoS-gray,
                    # each later segment takes the type of its start vertex.
                    seg_types = ["los"] + vertex_types
                    for j, seg_type in enumerate(seg_types):
                        _emit(
                            seg_type,
                            _segment_opacity(seg_type, is_true_los, path_type)
                            * scale,
                            width,
                            coords[j : j + 2],
                        )

        added_to_legend = set()
        for (seg_type, opacity, width), batch in batches.items():
            show_in_legend = show_legend and seg_type not in added_to_legend
            if show_in_legend:
                added_to_legend.add(seg_type)
            fig.add_trace(
                go.Scatter3d(
                    x=batch["x"],
                    y=batch["y"],
                    z=batch["z"],
                    mode="lines",
                    line={
                        "color": path_colors.get(seg_type, "gray"),
                        "width": width,
                    },
                    opacity=opacity,
                    name=seg_type.replace("_", " ").title(),
                    showlegend=show_in_legend,
                    legendgroup=f"paths_{seg_type}",
                    # Hover would run a pick pass over every vertex of a
                    # batch on each mouse move, and all it could report is
                    # the type — which the legend already says.
                    hoverinfo="skip",
                )
            )
    except Exception as e:
        st.error(f"Could not render paths: {e!s}")
        st.code(traceback.format_exc())


def get_path_type_name(interaction_type):
    """
    Convert interaction type constant to path type name.
    """
    return {
        1: "specular",
        2: "diffuse",
        8: "diffraction",
        4: "refraction",
    }.get(interaction_type, "los")


def find_duplex_metadata(value, _seen=None):
    """Find propagated duplex metadata in a result/config source chain."""
    if _seen is None:
        _seen = set()
    if id(value) in _seen:
        return None
    _seen.add(id(value))
    if isinstance(value, dict):
        direct = value.get("duplex_metadata")
        if isinstance(direct, dict) and direct.get("side_order"):
            return direct
        for key in ("results", "parameters", "config_info", "source_info"):
            if key in value:
                found = find_duplex_metadata(value[key], _seen)
                if found:
                    return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = find_duplex_metadata(item, _seen)
            if found:
                return found
    return None


def duplex_batch_labels(value, batch_size):
    """Label first/second batch halves without changing batch indices."""
    metadata = find_duplex_metadata(value) or {}
    return direction_labels_from_metadata(metadata, batch_size)


def duplex_batch_pair(value, batch_size, selected_batch=0):
    """Return the matching forward/reverse batch indices for a duplex result.

    Duplex data stores all forward batches first and the corresponding reverse
    batches in the second half. For non-duplex data this returns only the
    selected batch, preserving the legacy visualizer behaviour.
    """
    metadata = find_duplex_metadata(value) or {}
    batch_size = int(batch_size)
    selected_batch = max(0, min(int(selected_batch), batch_size - 1))
    side_order = metadata.get("side_order") or []
    base = int(metadata.get("base_batch_size") or 0)
    if len(side_order) == 2 and not base and batch_size % 2 == 0:
        base = batch_size // 2
    if len(side_order) != 2 or base <= 0 or base * 2 != batch_size:
        return [(selected_batch, f"Batch {selected_batch}")]
    local = selected_batch % base
    return [
        (local, "Base → Mobile"),
        (base + local, "Mobile → Base"),
    ]


# Ceilings for the smooth-animation mode. Every frame carries a full copy of
# the traces, and nothing about the axis bounds them: a trajectory run has one
# time step per RX point, so 3000 points became 3000 frames and the Signal
# Channelizer pushed a single 361 MB delta on 2026-09-07. The client could not
# drain it, the websocket dropped, and the disconnect killed the OptiReceiver
# stage that was running behind it (see `session_guard`). Frames are the only
# lever here: the samples inside one frame are the waveform itself, and
# thinning those is what makes a pulse or a peak stop being where it is.
MAX_ANIMATION_FRAMES = 200
MAX_ANIMATION_PAYLOAD_BYTES = 24 * 1024**2


def decimate_animation_indices(start, count, frame_bytes, ui):
    """Stride an animation's time indices down to a sendable frame count.

    Public because the frame cap is not specific to the time axis: the CIR
    generator animates a *trajectory RX* axis the same way, and it is the
    same 3000 frames.
    """
    indices = list(range(start, start + count))
    allowed = MAX_ANIMATION_FRAMES
    if frame_bytes:
        allowed = min(
            allowed, max(1, int(MAX_ANIMATION_PAYLOAD_BYTES // frame_bytes))
        )
    if len(indices) <= allowed:
        return indices

    step = int(np.ceil(len(indices) / allowed))
    strided = indices[::step]
    ui.caption(
        f"Animation shows every {step}th step ({len(strided)} of {count} "
        "frames) — the full set would not survive the websocket. Switch to "
        "the slider for an exact step."
    )
    return strided


def visualization_time_control(
    plugin_instance,
    widget_manager,
    result,
    num_time_steps,
    *,
    key_prefix,
    container=None,
    frame_bytes=None,
):
    """Shared Streamlit/smooth-Plotly time control for result visualizers.

    Returns ``(selected_time_index, animation_indices)``. The second item is
    ``None`` for the Streamlit slider and a list of absolute time indices for
    client-side Plotly animation. Trajectory metadata narrows both modes to the
    selected trajectory segment.

    ``frame_bytes`` is the caller's estimate of what one animation frame costs
    on the wire (trace samples times their item size, once per trace). When
    given, the frame count is capped by payload rather than by the flat
    ``MAX_ANIMATION_FRAMES``, and the returned indices are strided over the
    full span — the animation still covers the whole trajectory, at a coarser
    step. The slider mode is never decimated: it sends one frame.
    """
    ui = container if container is not None else st
    count_total = max(1, int(num_time_steps))
    start, count = 0, count_total
    segments = ((find_axis_metadata(result) or {}).get("time") or {}).get(
        "segments"
    ) or []
    valid_segments = []
    for segment in segments:
        seg_start = max(0, int(segment.get("start", 0)))
        seg_count = max(0, int(segment.get("count", 0)))
        seg_count = min(seg_count, count_total - seg_start)
        if seg_start < count_total and seg_count > 0:
            valid_segments.append((seg_start, seg_count, segment))
    if valid_segments:
        labels = [
            (
                f"Trajectory {seg.get('trajectory_id', i)} "
                f"(TX {seg.get('tx_index', '?')}, {seg_count} pts)"
            )
            for i, (_, seg_count, seg) in enumerate(valid_segments)
        ]
        selected = plugin_instance.create_widget(
            widget_manager=widget_manager,
            widget_type=ui.selectbox,
            widget_name=f"{key_prefix}_trajectory",
            default_value=0,
            value_param="index",
            args=("Select trajectory", labels),
            value_serializer=lambda value: labels.index(value),
            value_deserializer=lambda index: index,
            rerun_scope="fragment",
        )
        selected_index = labels.index(selected) if selected in labels else 0
        start, count, _ = valid_segments[selected_index]
    if count <= 1:
        return start, None

    modes = ["Slider (Streamlit)", "Smooth animation (Plotly)"]
    mode = plugin_instance.create_widget(
        widget_manager=widget_manager,
        widget_type=ui.radio,
        widget_name=f"{key_prefix}_mode",
        default_value=1,
        value_param="index",
        args=("Time control", modes),
        value_serializer=lambda value: modes.index(value),
        value_deserializer=lambda index: index,
        kwargs={
            "horizontal": True,
            "help": (
                "Streamlit slider redraws after release. Smooth animation "
                "switches Plotly frames directly in the browser."
            ),
        },
        rerun_scope="fragment",
    )
    if str(mode).startswith("Smooth"):
        return start, decimate_animation_indices(
            start, count, frame_bytes, ui
        )
    local_index = int(
        plugin_instance.create_widget(
            widget_manager=widget_manager,
            widget_type=ui.slider,
            widget_name=f"{key_prefix}_slider_{start}_{count}",
            default_value=0,
            value_param="value",
            args=("Time step", 0, count - 1),
            kwargs={"step": 1},
            rerun_scope="fragment",
        )
    )
    return start + local_index, None


def add_plotly_frame_slider(
    fig, labels, *, prefix="Time step: ", redraw=False
):
    """Attach browser-side frame scrubbing and play/pause controls.

    ``redraw=False`` is the cheap path: Plotly updates the traces in place and
    skips a full replot. It is also the *only* path that works for plain
    ``scatter`` data and nothing else — a frame that changes the layout (a
    title, a subplot annotation) or that carries WebGL traces (``scattergl``)
    is simply not repainted, so the chart sits on frame 0 while the slider
    moves. Pass ``redraw=True`` for those.
    """
    frame_options = {"duration": 0, "redraw": bool(redraw)}
    play_options = {"duration": 120, "redraw": bool(redraw)}
    steps = [
        {
            "method": "animate",
            "label": str(label),
            "args": [
                [str(index)],
                {
                    "mode": "immediate",
                    "frame": dict(frame_options),
                    "transition": {"duration": 0},
                },
            ],
        }
        for index, label in enumerate(labels)
    ]
    fig.update_layout(
        sliders=[
            {
                "active": 0,
                "x": 0.14,
                "y": 0.0,
                "len": 0.86,
                "pad": {"t": 50, "b": 10},
                "currentvalue": {"prefix": prefix, "visible": True},
                "steps": steps,
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "showactive": False,
                "x": 0.0,
                "y": 0.0,
                "pad": {"t": 50, "r": 8},
                "buttons": [
                    {
                        "label": "▶",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": dict(play_options),
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "⏸",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": dict(frame_options),
                            },
                        ],
                    },
                ],
            }
        ],
    )
    return fig


def direction_labels_from_metadata(metadata, batch_size):
    """Per-batch-index duplex direction labels from a ``duplex_metadata`` dict.

    Returns a list of length ``batch_size``. When the metadata describes a
    two-sided duplex channel (``side_order`` of length 2), the first
    ``base_batch_size`` indices are the forward (base-station → mobile) side and
    the rest the reverse (mobile → base-station) side; otherwise every index is
    a plain ``Batch i``.
    """
    metadata = metadata or {}
    batch_size = int(batch_size)
    side_order = metadata.get("side_order") or []
    base = int(metadata.get("base_batch_size") or 0)
    if len(side_order) == 2 and not base and batch_size % 2 == 0:
        base = batch_size // 2
    if len(side_order) != 2 or base * 2 != batch_size:
        return [f"Batch {i}" for i in range(batch_size)]
    first = (
        "Base → Mobile"
        if str(side_order[0]).startswith("forward")
        else "First side"
    )
    second = (
        "Mobile → Base"
        if str(side_order[1]).startswith("reverse")
        else "Second side"
    )
    return [
        first if base == 1 else f"{first} — batch {i}" for i in range(base)
    ] + [second if base == 1 else f"{second} — batch {i}" for i in range(base)]


# ---------------------------------------------------------------------------
# Axis metadata — a stage-agnostic descriptor of a 7-dim CIR/signal tensor.
#
# The pipeline tensors share the leading-5-axis layout
# ``[batch, rx, rx_ant, tx, tx_ant, ...]``. ``batch`` encodes the duplex
# direction (TX→RX / RX→TX), and — for trajectory JSONs where the RX walk was
# moved onto the time axis — the time axis is a concatenation of several
# per-TX walks. ``axis_metadata`` records both so any visualizer (in a plugin
# or a page) can split the time axis back into individual trajectories, filter
# by the owning TX, and know which direction a batch index represents. It rides
# through the pipeline via the standard ``config_info``/``source_info`` chain
# and is dim-agnostic (plain JSON-able dict), so ResultsSaver serializes it too.
# ---------------------------------------------------------------------------
AXIS_METADATA_VERSION = 1
CIR_AXIS_NAMES = ["batch", "rx", "rx_ant", "tx", "tx_ant", "path", "time"]


def build_axis_metadata(
    *,
    batch_size,
    duplex_metadata=None,
    trajectory_segments=None,
    num_time_steps=None,
    tx_index=None,
    dt=None,
    axes=None,
):
    """Assemble the standardized ``axis_metadata`` dict (see module note).

    ``trajectory_segments`` is the per-combo, RX-list-rebased segment list
    (each with ``rx_start_index`` / ``n_points`` / ``tx_index`` / ``dt`` / ...)
    stored by the CIR generator after the RX→time swap. Its segments are mapped
    onto the time axis as ``time_segments``.
    """
    batch_size = int(batch_size)
    directions = direction_labels_from_metadata(
        duplex_metadata or {}, batch_size
    )
    time_segments = []
    for i, seg in enumerate(trajectory_segments or []):
        try:
            start = int(seg.get("rx_start_index", 0))
            count = int(seg.get("n_points", 0))
        except (TypeError, ValueError):
            continue
        time_segments.append(
            {
                "trajectory_id": i,
                "tx_index": seg.get("tx_index"),
                "start": start,
                "count": count,
                "dt": seg.get("dt"),
                "seed": seg.get("seed"),
                "mode": seg.get("mode"),
                "v_min": seg.get("v_min"),
                "v_max": seg.get("v_max"),
            }
        )
    meta = {
        "version": AXIS_METADATA_VERSION,
        "axes": list(axes) if axes else list(CIR_AXIS_NAMES),
        "batch": {
            "kind": "duplex_direction",
            "directions": directions,
            "side_order": (duplex_metadata or {}).get("side_order") or [],
            "base_batch_size": (duplex_metadata or {}).get("base_batch_size"),
        },
        "time": {
            "kind": "trajectory" if time_segments else "static",
            "dt": dt,
            "num_time_steps": (
                int(num_time_steps) if num_time_steps is not None else None
            ),
            "segments": time_segments,
        },
        "tx_index": tx_index,
    }
    return meta


def find_axis_metadata(value, _seen=None):
    """Locate a propagated ``axis_metadata`` dict in a result/config chain."""
    if _seen is None:
        _seen = set()
    if id(value) in _seen:
        return None
    _seen.add(id(value))
    if isinstance(value, dict):
        direct = value.get("axis_metadata")
        if isinstance(direct, dict) and "axes" in direct:
            return direct
        for key in ("results", "parameters", "config_info", "source_info"):
            if key in value:
                found = find_axis_metadata(value[key], _seen)
                if found:
                    return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = find_axis_metadata(item, _seen)
            if found:
                return found
    return None


def axis_metadata_time_selector(
    axis_metadata,
    num_time_steps,
    *,
    key_prefix,
    container=None,
    tx_filter=None,
):
    """Render trajectory / time-point selectors for a time-axis-bearing tensor.

    Splits the concatenated time axis into the walks recorded in
    ``axis_metadata['time']['segments']`` and lets the user pick a trajectory
    (optionally filtered to one TX) and a point within it. Returns
    ``(time_index, info)`` where ``info`` carries the resolved trajectory id,
    owning ``tx_index`` and the segment's ``(start, count)`` for callers that
    want to slice the whole walk.

    Falls back to a plain time-step slider when no trajectory metadata exists,
    so it is safe to call unconditionally.
    """
    ui = container if container is not None else st
    num_time_steps = int(num_time_steps)
    segments = ((axis_metadata or {}).get("time") or {}).get("segments") or []
    if tx_filter is not None:
        segments = [
            s for s in segments if s.get("tx_index") in (None, tx_filter)
        ]
    if not segments or num_time_steps <= 1:
        t_idx = 0
        if num_time_steps > 1:
            t_idx = ui.slider(
                "Time step",
                0,
                num_time_steps - 1,
                0,
                key=f"{key_prefix}_tstep",
            )
        return int(t_idx), {
            "trajectory_id": None,
            "tx_index": None,
            "start": 0,
            "count": num_time_steps,
        }
    labels = []
    for s in segments:
        tx = s.get("tx_index")
        tx_txt = f"TX {tx}" if tx is not None else "TX ?"
        labels.append(
            f"Trajectory {s['trajectory_id']} ({tx_txt}, "
            f"{s.get('count', 0)} pts)"
        )
    sel = ui.selectbox(
        "Select trajectory",
        list(range(len(segments))),
        format_func=lambda i: labels[i],
        key=f"{key_prefix}_traj",
    )
    seg = segments[int(sel)]
    start = int(seg.get("start", 0))
    count = max(1, int(seg.get("count", 1)))
    end = min(start + count, num_time_steps)
    local = 0
    if end - start > 1:
        local = ui.slider(
            "Point in trajectory",
            0,
            end - start - 1,
            0,
            key=f"{key_prefix}_pt",
        )
    return start + int(local), {
        "trajectory_id": seg.get("trajectory_id"),
        "tx_index": seg.get("tx_index"),
        "start": start,
        "count": end - start,
    }


# ---------------------------------------------------------------------------
# Memory-aware N-dimensional chunk planner.
#
# The heavy stages (SignalChannelizer delay-and-sum, OptiReceiver matched
# filter / peak finder) used to chunk only the batch and time axes. With
# trajectory JSONs the other axes (rx, tx, ...) can also be large, so we split
# the tensor across *all* requested axes into blocks sized to a fraction of the
# free RAM/VRAM (divided across parallel workers). ``free_memory_bytes`` picks
# the right pool for the device; ``plan_axis_chunks`` returns a per-axis chunk
# size; ``iter_nd_blocks`` walks the resulting blocks as tuples of slices.
# ---------------------------------------------------------------------------
def free_memory_bytes(device="cpu"):
    """Free memory (bytes) for ``device`` — VRAM for cuda/gpu, else host RAM.

    ``auto`` resolves the same way the heavy blocks pick their compute
    device: CUDA when available, host otherwise. Budgeting host RAM while
    computing on the GPU sizes blocks far past VRAM and OOMs.
    """
    dev = str(device).lower()
    if "cuda" in dev or "gpu" in dev or "auto" in dev:
        try:
            import torch

            if torch.cuda.is_available():
                free, _total = torch.cuda.mem_get_info()
                return int(free)
        except Exception:
            pass
    return available_memory_bytes()


def _cgroup_v2_available_bytes(
    cgroup_root="/sys/fs/cgroup", proc_cgroup="/proc/self/cgroup"
):
    """Return the remaining memory in this process' cgroup v2 hierarchy.

    A systemd scope can impose ``MemoryMax`` below the host's available RAM.
    Check the current cgroup and all its parents because a parent slice may be
    the limiting one. ``None`` means that no finite readable limit was found.
    """
    try:
        with open(proc_cgroup, encoding="utf-8") as fh:
            rel = next(
                line.rstrip("\n").split("::", 1)[1]
                for line in fh
                if "::" in line
            )
    except (OSError, StopIteration):
        return None

    current = os.path.normpath(os.path.join(cgroup_root, rel.lstrip("/")))
    root = os.path.normpath(cgroup_root)
    remaining = []
    while current == root or current.startswith(root + os.sep):
        try:
            with open(
                os.path.join(current, "memory.max"), encoding="utf-8"
            ) as fh:
                limit_text = fh.read().strip()
            if limit_text != "max":
                with open(
                    os.path.join(current, "memory.current"), encoding="utf-8"
                ) as fh:
                    used = int(fh.read().strip())
                remaining.append(max(0, int(limit_text) - used))
        except (OSError, ValueError):
            pass
        if current == root:
            break
        current = os.path.dirname(current)
    return min(remaining) if remaining else None


def available_memory_bytes():
    """Memory currently available to this process, including cgroup limits."""
    try:
        import psutil

        host_available = int(psutil.virtual_memory().available)
    except Exception:
        host_available = 4 * 1024**3
    cgroup_available = _cgroup_v2_available_bytes()
    if cgroup_available is not None:
        return min(host_available, cgroup_available)
    return host_available


def release_gpu_memory():
    """Return this process' cached VRAM to the driver.

    Two caching allocators are in play and freeing one does nothing for the
    other: torch's, which the heavy plugins allocate through, and Dr.Jit's,
    which Sionna RT allocates through.

    Only the torch half actually comes back. ``dr.flush_malloc_cache()`` is
    the documented API and is called here, but measured on Dr.Jit 1.3.1 +
    CUDA it returns nothing to the driver: allocate and free three buffers
    and the process still holds every byte, so a finished ray-tracing run
    stays resident for the life of the Streamlit process (7.1 GiB of a
    24 GiB card, on the 2026-09-07 trajectory run). Keep the call — it is
    correct, cheap, and covers host-pinned blocks and future versions — but
    do not count on it for VRAM. The way to not lose the card to Sionna RT
    is to allocate less of it in the first place.

    Every call is best-effort: a CPU-only box, a driver that is already
    gone, or a Dr.Jit built without the CUDA backend must not take a plugin
    run down on the way out. Returns the bytes handed back, or 0 when there
    is no GPU to ask.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        before, _total = torch.cuda.mem_get_info()
    except Exception:
        return 0
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        import drjit as dr

        dr.flush_malloc_cache()
    except Exception:
        pass
    try:
        after, _total = torch.cuda.mem_get_info()
        return max(0, int(after) - int(before))
    except Exception:
        return 0


def release_worker_pool():
    """Shut the idle joblib/loky pool down instead of waiting it out.

    joblib keeps its executor warm for reuse, and every worker holds a CUDA
    context of its own — ~420 MiB, so an ``n_jobs=16`` stage parks ~6.5 GiB
    of pure overhead on the card. The pool *is* reaped on joblib's idle
    timeout, but that timeout is minutes, which is exactly when the next
    stage of the chain starts and finds the card full.

    Reads loky's module-level singleton rather than calling
    ``get_reusable_executor()``: the public accessor *starts* a pool when
    none exists, which would spawn the very workers this is meant to drop.
    Returns True when a pool was actually shut down.
    """
    try:
        from joblib.externals.loky import reusable_executor

        executor = getattr(reusable_executor, "_executor", None)
        if executor is None:
            return False
        executor.shutdown(wait=True)
        return True
    except Exception:
        return False


def gpu_memory_held_bytes():
    """VRAM this process holds, or None when it cannot be read.

    Torch only knows about its own allocator, and the memory that matters
    here is Dr.Jit's — which torch reports as zero and cannot free. So ask
    the driver for the per-process figure instead. Best-effort: no GPU, no
    ``nvidia-smi``, or a container without visibility all return None rather
    than raising in a ``finally``.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    me = os.getpid()
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == me:
            try:
                return int(float(parts[1])) * 1024**2
            except ValueError:
                return None
    return 0


GPU_JOB_CACHE_ENTRIES = 4


@st.cache_resource(show_spinner=False)
def _gpu_job_store():
    """Results of work sent to a child process, shared per server.

    Same shape as ``_scene_mesh_store``: a plain dict under a lock, with the
    cache decorator only making it a singleton. The executor itself is
    loky's — the two heavy plugins already drive it under Streamlit, and it
    solves the spawn/``__main__`` problem that bare
    ``multiprocessing`` would re-run the page over.
    """
    return {
        "done": OrderedDict(),
        "running": {},
        "warm": set(),
        "lock": threading.Lock(),
    }


@overload
def submit_gpu_job[T](
    key: str,
    fn: Callable[..., T],
    *args: Any,
    background: Literal[False],
    warm_group: str | None = ...,
    **kwargs: Any,
) -> T: ...


@overload
def submit_gpu_job[T](
    key: str,
    fn: Callable[..., T],
    *args: Any,
    background: bool = ...,
    warm_group: str | None = ...,
    **kwargs: Any,
) -> T | None: ...


def submit_gpu_job(
    key, fn, *args, background=True, warm_group=None, **kwargs
):
    """Run ``fn`` in a child process and reclaim its VRAM when it is done.

    This is the only way to get GPU memory back from Sionna RT: Dr.Jit's
    allocator never returns device memory to the driver (see
    ``release_gpu_memory``), so the process that ray-traces has to be a
    process that exits. Measured on an LOD3 city, spawning the child,
    importing ``sionna.rt`` and loading the scene costs 0.8 s total, against
    minutes for the solve it replaces.

    ``submit`` returns in ~1 ms, so with ``background=True`` the script
    thread is never blocked: the first call returns ``None`` and the caller
    draws a placeholder and polls, the way ``scene_mesh_parts`` does. Set
    ``background=False`` to wait for the result inline (tests, workers).

    ``fn`` must be importable by qualified name in a fresh interpreter, so
    it belongs in a module like ``pages/rt_solver_worker.py`` — not in a
    plugin loaded through ``importlib`` under a synthetic name, and not in a
    closure.
    """
    from joblib.externals.loky import get_reusable_executor

    store = _gpu_job_store()
    with store["lock"]:
        if key in store["done"]:
            store["done"].move_to_end(key)
            return store["done"][key]
        pending = store["running"].get(key)
        if pending is None:
            pending = get_reusable_executor(max_workers=1).submit(
                fn, *args, **kwargs
            )
            store["running"][key] = pending
        if background and not pending.done():
            return None

    try:
        result = pending.result()
    except Exception:
        # Drop the failed job so the next rerun retries instead of
        # re-raising the same stale future forever.
        with store["lock"]:
            store["running"].pop(key, None)
        _reclaim_idle_gpu_worker(store)
        raise

    with store["lock"]:
        store["running"].pop(key, None)
        store["done"][key] = result
        while len(store["done"]) > GPU_JOB_CACHE_ENTRIES:
            store["done"].popitem(last=False)
        if warm_group is not None:
            store["warm"].add(warm_group)
    _reclaim_idle_gpu_worker(store)
    return result


#: VRAM one solver child holds once it has a scene and has solved once:
#: 1794 MiB measured on LOD3 Kazan (the CUDA context is 420 MiB of that,
#: and none of it ever comes back inside a live process). Used to cap how
#: many of them may run at once.
GPU_WORKER_VRAM_BYTES = 1800 * 1024**2


@st.cache_resource(show_spinner=False)
def _gpu_pool_store():
    """Dedicated single-worker executors, by group.

    Not `get_reusable_executor(max_workers=N)`: that is loky's *singleton*,
    shared with the heavy plugins, and resizing it shuts the pool down —
    including the child that is holding a scene warm. It also hands a task
    to whichever worker is free, and this needs the opposite: a frame must
    land on the same process every time, or every worker loads its own copy
    of the city and the card fills with duplicates. One single-worker
    executor per slot gives that affinity for nothing.
    """
    return {"pools": {}, "lock": threading.Lock()}


#: VRAM a loky worker of the heavy plugins parks before it has computed
#: anything: its own CUDA context. Measured at ~420 MiB, which is why
#: ``n_jobs=16`` puts 6.5 GiB of pure overhead on the card before a single
#: tensor. Rounded up, because a context that is one page over the estimate
#: costs a worker and a context that is under it costs the run.
CUDA_CONTEXT_VRAM_BYTES = 440 * 1024**2


#: What a worker holds beyond the payload it was handed: the output it is
#: filling, the tensors it uploaded, the intermediates. Measured as roughly
#: the payload again, so a task is budgeted at twice what it carries. Being
#: wrong high costs a worker; being wrong low cost this box its whole
#: Streamlit process to the kernel's OOM killer.
WORKER_WORKING_SET_FACTOR = 2.0


#: Where joblib parks the array arguments it hands to loky workers. Left
#: alone it picks ``/dev/shm``, which on this box is tmpfs -- i.e. RAM, and
#: the one kind of RAM nothing accounts for: it belongs to no process, so
#: it shows up neither in the app's RSS nor in the cgroup's own usage, and
#: it survives the process that made it. Measured on 2026-09-22, two
#: Streamlit processes that had already died were still holding **50 GiB**
#: of it, which the next run then started out against.
JOBLIB_TEMP_FOLDER_ENV = "JOBLIB_TEMP_FOLDER"
#: The directories joblib may have used. ``/tmp`` is tmpfs here too.
JOBLIB_SCRATCH_CANDIDATES = ("/dev/shm", "/tmp")


def configure_joblib_temp_folder(scratch_root=None):
    """Point joblib's worker spill at a disk, not at RAM.

    Returns the directory it set, or ``None`` when the caller has already
    chosen one — an explicit ``JOBLIB_TEMP_FOLDER`` is a decision, not a
    default to overwrite.
    """
    existing = os.environ.get(JOBLIB_TEMP_FOLDER_ENV)
    if existing:
        return None
    root = scratch_root or resolve_scratch_root()
    folder = os.path.join(tmp_dir_for(root), "joblib")
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError:
        return None
    os.environ[JOBLIB_TEMP_FOLDER_ENV] = folder
    return folder


def sweep_orphaned_joblib_folders(directories=None):
    """Delete joblib spill folders whose owning process is gone.

    joblib removes its folder when the executor shuts down cleanly. When
    the parent is killed instead — which is exactly what happens to a run
    that runs the box out of memory — nobody removes it, and on tmpfs the
    memory is not returned until somebody does. So this runs at start-up,
    the only moment after such a death when anything can.

    The owning pid is in the name (``joblib_memmapping_folder_<pid>_...``)
    and a folder is removed only when that pid is not alive **and** the
    folder belongs to this user. A live pid is left strictly alone: another
    session of this app may be mid-run.
    """
    import shutil

    removed_bytes = 0
    removed = 0
    for parent in directories or (
        *JOBLIB_SCRATCH_CANDIDATES,
        os.environ.get(JOBLIB_TEMP_FOLDER_ENV) or "",
    ):
        if not parent or not os.path.isdir(parent):
            continue
        try:
            names = os.listdir(parent)
        except OSError:
            continue
        for name in names:
            if not name.startswith("joblib_memmapping_folder_"):
                continue
            path = os.path.join(parent, name)
            parts = name.split("_")
            if len(parts) < 4 or not parts[3].isdigit():
                continue
            pid = int(parts[3])
            try:
                os.kill(pid, 0)
                continue  # alive: another run may own it
            except ProcessLookupError:
                pass
            except PermissionError:
                continue  # someone else's process, and someone else's files
            try:
                if os.stat(path).st_uid != os.getuid():
                    continue
                size = sum(
                    os.path.getsize(os.path.join(root, f))
                    for root, _dirs, files in os.walk(path)
                    for f in files
                )
                shutil.rmtree(path)
            except OSError:
                continue
            removed_bytes += size
            removed += 1
    return removed, removed_bytes


#: Arrays smaller than this stay in RAM: a file costs an inode, an open
#: and a page-table setup, and below a few megabytes that is more than the
#: memory it saves.
SPILL_MIN_ARRAY_BYTES = 8 * 1024**2
#: Moves that floor, the way ``STREAMFLEX_EXPERIMENT_ROOT`` moves the scan
#: library. Read per call rather than at import, so a test can force every
#: array through the memmap path and see the consumers meet one.
SPILL_MIN_BYTES_ENV = "STREAMFLEX_SPILL_MIN_BYTES"


def spill_min_array_bytes():
    """The floor under spilling, env override included."""
    raw = os.environ.get(SPILL_MIN_BYTES_ENV)
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return SPILL_MIN_ARRAY_BYTES


def spill_payload_arrays(
    obj,
    directory,
    *,
    min_bytes=None,
    mmap_mode: Literal["r+", "r", "w+", "c"] | None = "c",
    _counter=None,
):
    """Move an in-memory payload's big arrays to disk, read back as memmaps.

    The bus between plugins keeps every stage's payload alive at once, and
    a ray-tracing run puts 120 CIR sets on it: measured, 326 MiB each and
    883 MiB at the worst, which is ~45 GiB of *anonymous* memory. Anonymous
    memory cannot be reclaimed -- the kernel's only move under pressure is
    to kill the process, and on 2026-09-22 it did exactly that at the
    scope's 100 GiB ceiling, taking a finished four-hour trajectory run
    with it.

    The same bytes in a file are page cache: the reader sees an ordinary
    array, the pages are faulted in on demand, and under pressure the
    kernel drops the clean ones instead of the process. Nothing downstream
    changes, because a memmap *is* an ``ndarray`` -- it slices, it does
    arithmetic, and ``np.asarray`` on it is free.

    The mode is copy-on-write rather than read-only, and that is not
    caution: ``torch.from_numpy`` on a read-only array returns a tensor
    PyTorch itself warns is undefined to write to, which is a worse
    failure than the one being avoided. Under copy-on-write an untouched
    page is still clean page cache and still reclaimable -- only a page
    somebody writes turns private, and nothing downstream writes.

    Returns ``(payload, bytes_spilled)``. Arrays already backed by a file
    are left alone, and so is anything under ``min_bytes``.
    """
    import numpy as _np

    if min_bytes is None:
        min_bytes = spill_min_array_bytes()
    if _counter is None:
        _counter = [0, 0]

    if isinstance(obj, _np.ndarray):
        if isinstance(obj, _np.memmap) or obj.nbytes < int(min_bytes):
            # Already on disk (from an earlier pass over the same payload),
            # or too small to be worth a file. Either way it is left where
            # it is -- and `payload_spill_directories` is what keeps the
            # file it already lives in from being swept away.
            return obj, _counter[0]
        path = os.path.join(directory, f"bus_{_counter[1]:06d}.npy")
        _counter[1] += 1
        try:
            os.makedirs(directory, exist_ok=True)
            # ``np.save`` writes the logical array, so the non-contiguous
            # ``re``/``im`` views that complex payloads travel as come back
            # as ordinary contiguous arrays rather than as halves of
            # something.
            _np.save(path, _np.ascontiguousarray(obj))
            spilled = _np.load(path, mmap_mode=mmap_mode)
        except (OSError, ValueError):
            # Spilling is an optimisation, never a correctness
            # requirement: a full disk, a swept directory or a racing
            # rerun must not throw away a run that has already been
            # computed. Keep the array in memory and carry on.
            logger.warning(f"Could not spill an array to {path}; keeping it in RAM.")
            return obj, _counter[0]
        _counter[0] += int(obj.nbytes)
        return spilled, _counter[0]

    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key], _ = spill_payload_arrays(
                value,
                directory,
                min_bytes=min_bytes,
                mmap_mode=mmap_mode,
                _counter=_counter,
            )
        return obj, _counter[0]

    if isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i], _ = spill_payload_arrays(
                value,
                directory,
                min_bytes=min_bytes,
                mmap_mode=mmap_mode,
                _counter=_counter,
            )
        return obj, _counter[0]

    if isinstance(obj, tuple):
        spilled = [
            spill_payload_arrays(
                value,
                directory,
                min_bytes=min_bytes,
                mmap_mode=mmap_mode,
                _counter=_counter,
            )[0]
            for value in obj
        ]
        return tuple(spilled), _counter[0]

    return obj, _counter[0]


def payload_spill_directories(obj, _seen=None):
    """Every directory the payload's memmapped arrays actually live in.

    What a payload references is *not* the directory of the most recent
    spill. A second pass over an already-spilled payload writes nothing --
    its arrays are memmaps, and memmaps are left alone -- so the newest
    directory can be empty while the bus reads entirely out of an older
    one. Keeping only the newest then deletes the live data: measured, a
    re-run wiped the very files the visualizers were drawing from, and the
    next pass died on `No such file or directory: bus_000006.npy`.
    """
    import numpy as _np

    if _seen is None:
        _seen = set()
    if isinstance(obj, _np.memmap):
        name = getattr(obj, "filename", None)
        if name:
            _seen.add(os.path.dirname(os.path.abspath(name)))
    elif isinstance(obj, dict):
        for value in obj.values():
            payload_spill_directories(value, _seen)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            payload_spill_directories(value, _seen)
    return _seen


def drop_spill_directories(parent, keep=()):
    """Remove spill directories under ``parent`` that nothing references.

    ``keep`` is the set of directories the *current* payload reads from --
    from `payload_spill_directories`, not a guess at which one is newest.
    Called after the bus has been handed its new payload and never before:
    the previous payload is read right up until the moment it is replaced.
    """
    import shutil

    if isinstance(keep, str):
        keep = (keep,)
    keep_abs = {os.path.abspath(k) for k in keep if k}
    removed = 0
    try:
        entries = os.listdir(parent)
    except OSError:
        return 0
    for name in entries:
        path = os.path.abspath(os.path.join(parent, name))
        if path in keep_abs or not os.path.isdir(path):
            continue
        try:
            shutil.rmtree(path)
            removed += 1
        except OSError:
            pass
    return removed


def process_rss_bytes():
    """Resident memory of this process, or 0 when it cannot be read.

    The one measurement that does not care how the data is *represented*.
    Counting numpy arrays is fine right up until something has already
    turned them into Python lists -- which is exactly what the saver's
    "Save as object" mode does to every 1-D array before anything
    downstream gets to look, so an array-based estimate reads a row that
    costs gigabytes as weighing nothing.
    """
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    try:
        with open("/proc/self/statm", encoding="utf-8") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0


def payload_array_bytes(obj):
    """Bytes of every numpy array under ``obj``.

    Complex arrays travel split as ``{"re": ..., "im": ...}``, and those two
    are *views* of one buffer at half its dtype width, so they add back up
    to the array rather than to twice it. A payload whose arrays have been
    spilled to memmap files carries paths rather than arrays and is counted
    at what it actually occupies, which is nearly nothing.
    """
    import numpy as _np

    if isinstance(obj, _np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return sum(payload_array_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(payload_array_bytes(v) for v in obj)
    return 0


def worker_payload_bytes(*payload_lists, working_set_factor=None):
    """RAM one configuration costs a worker, measured from the data itself.

    The *largest* entry of each list rather than the average: configurations
    are handed out without regard to their size, so any worker can draw the
    big one, and a budget built on the average is wrong exactly when it
    matters.
    """
    factor = (
        WORKER_WORKING_SET_FACTOR
        if working_set_factor is None
        else float(working_set_factor)
    )
    carried = 0
    for payloads in payload_lists:
        carried += max(
            (payload_array_bytes(item) for item in (payloads or ())),
            default=0,
        )
    return max(1, int(carried * factor))


def plan_worker_processes(
    n_tasks,
    *,
    host_bytes_per_task,
    device_bytes_per_task=0,
    device="cpu",
    requested=0,
    host_margin=0.5,
    device_margin=0.7,
):
    """How many config workers this machine can actually feed, and why.

    A worker count is not a preference, it is whatever the scarcest of four
    things allows, and on this box the scarce one is rarely the CPU: a
    SignalChannelizer run with ``n_jobs=16`` lost four configurations to
    CUDA OOM (two workers holding 8.7 and 9.3 GiB of a 24 GiB card) and
    then took the whole app down on host RAM, because sixteen copies of a
    572 MiB payload is 9 GiB before any of them computes anything.

    So the count is measured from what a task actually carries rather than
    set: ``host_bytes_per_task`` is the payload the worker is handed plus
    whatever it builds that stays in RAM, and ``device_bytes_per_task`` is
    what it puts on the card *on top of* its CUDA context. ``requested``
    is a user ceiling (0 = none), never a floor -- asking for sixteen on a
    machine that fits three is how this went wrong in the first place.

    Returns ``(n_workers, plan)``, where ``plan`` names every limit and the
    one that bound, so a run that goes sequential says why rather than just
    being slow.
    """
    n_tasks = max(1, int(n_tasks))
    limits = {"tasks": n_tasks, "cpus": max(1, os.cpu_count() or 1)}

    host_need = max(1, int(host_bytes_per_task))
    limits["host_memory"] = max(
        1, int(available_memory_bytes() * float(host_margin) // host_need)
    )

    dev = str(device).lower()
    if "cuda" in dev or "gpu" in dev or "auto" in dev:
        try:
            import torch

            has_cuda = torch.cuda.is_available()
        except Exception:
            has_cuda = False
        if has_cuda:
            per_worker = CUDA_CONTEXT_VRAM_BYTES + max(
                0, int(device_bytes_per_task)
            )
            limits["device_memory"] = max(
                1,
                int(
                    free_memory_bytes("cuda")
                    * float(device_margin)
                    // per_worker
                ),
            )

    if requested:
        limits["requested"] = max(1, int(requested))

    n_workers = max(1, min(limits.values()))
    return n_workers, {
        "n_workers": n_workers,
        "limits": limits,
        "bound_by": min(limits, key=lambda k: limits[k]),
        "host_bytes_per_task": host_need,
        "device_bytes_per_task": max(0, int(device_bytes_per_task)),
    }


def gpu_worker_slots(group, size, *, vram_margin=0.7):
    """How many solver children this group may actually run in parallel.

    Capped by free device memory rather than by core count: the work is
    host-side and single-threaded (a path solve leaves the GPU at ~0% while
    one core rebuilds the megakernel), so more processes really is more
    throughput — right up to the point where the next scene does not fit
    and the driver starts thrashing.
    """
    size = max(1, int(size))
    try:
        free = int(free_memory_bytes("cuda"))
    except Exception:
        return size
    if free <= 0:
        return size
    allowed = int(free * float(vram_margin) // GPU_WORKER_VRAM_BYTES)
    return max(1, min(size, allowed))


def submit_gpu_batch(group, jobs, *, size, on_result=None, on_wait=None):
    """Run several solver jobs across a group's own children, in parallel.

    ``jobs`` is a sequence of ``(fn, args, kwargs)``. Results come back in
    the order given, and ``on_result(index, elapsed)`` is called as each one
    lands so the caller can report progress — out of order, because that is
    the order they finish in. ``on_wait()`` is called about every 0.4 s
    while anything is still running, which is how a caller draws a bar for
    work that reports from inside the children rather than by finishing:
    one job here can be a quarter of a run.

    The children are kept between calls exactly as ``submit_gpu_job``'s warm
    worker is: each one loads the scene once and answers every later job
    from it. ``release_gpu_worker(group)`` ends them.
    """
    import time as _time

    from joblib.externals.loky import ProcessPoolExecutor

    jobs = list(jobs)
    if not jobs:
        return []
    size = max(1, min(int(size), len(jobs)))
    # `size` is how many run at once; `jobs` may be many more than that.
    store = _gpu_pool_store()
    with store["lock"]:
        pools = store["pools"].setdefault(group, [])
        while len(pools) < size:
            # An explicit idle timeout, because loky's default reaps a
            # worker between frames and the next one then reloads the
            # city. These die when the view is switched off, which is
            # `release_gpu_worker`'s job and the only thing that returns
            # their VRAM anyway.
            pools.append(ProcessPoolExecutor(max_workers=1, timeout=1800))
        pools = pools[:size]

    from concurrent.futures import FIRST_COMPLETED, wait

    # Handed out one at a time, not dealt round-robin up front. Jobs here
    # are not equal: a window of a walk costs what its receivers cost, and
    # a blocked point is cheap where a point with rich multipath is not.
    # Dealt in advance, the run ends with one worker still grinding
    # through its share while the rest sit idle -- the load falls off and
    # so does the rate. A worker that finishes takes the next window
    # instead, and keeps the scene it already has.
    queue = list(enumerate(jobs))
    results: list[Any] = [None] * len(jobs)
    started = _time.time()
    inflight = {}

    def _feed(pool_index):
        if not queue:
            return
        index, (fn, args, kwargs) = queue.pop(0)
        future = pools[pool_index].submit(fn, *args, **dict(kwargs or {}))
        inflight[future] = (index, pool_index)

    for slot in range(len(pools)):
        _feed(slot)
    while inflight:
        landed, _ = wait(
            set(inflight), timeout=0.4, return_when=FIRST_COMPLETED
        )
        for future in landed:
            index, pool_index = inflight.pop(future)
            results[index] = future.result()
            if on_result is not None:
                on_result(index, _time.time() - started)
            _feed(pool_index)
        if on_wait is not None and inflight:
            on_wait()
    return results


def release_gpu_worker(warm_group):
    """Drop a keep-warm claim and kill the worker once none are left.

    The counterpart to ``submit_gpu_job(warm_group=...)``: while a group is
    registered the child survives between calls, so the scene it loaded stays
    warm and the next render is just a figure build. Releasing the last group
    ends the process, and *that* is what returns its VRAM — deleting the
    scene inside it would not (measured: load 892 MiB, delete, flush, still
    892 MiB).
    """
    store = _gpu_job_store()
    with store["lock"]:
        store["warm"].discard(warm_group)
        # Cached figures belong to the process that made them; keeping them
        # would hand back results for a scene that no longer exists.
        for cached in [k for k in store["done"] if _in_group(k, warm_group)]:
            del store["done"][cached]
    pool_store = _gpu_pool_store()
    with pool_store["lock"]:
        pools = pool_store["pools"].pop(warm_group, [])
    for pool in pools:
        # Each of these holds a scene of its own, which is the whole reason
        # they exist and the whole reason they must not outlive the view.
        pool.shutdown(wait=False, kill_workers=True)
    return _reclaim_idle_gpu_worker(store) or bool(pools)


def _in_group(key, warm_group):
    return isinstance(key, str) and key.startswith(f"{warm_group}:")


def _reclaim_idle_gpu_worker(store):
    """Kill the worker once nothing is queued — that is what frees the VRAM.

    Only when the store is idle: the executor has one worker, so shutting it
    down while another job is pending would abort that job too — and only
    when nothing has claimed the worker with ``warm_group``, since killing it
    would throw away the scene that claim exists to keep loaded.
    """
    with store["lock"]:
        if store["running"] or store["warm"]:
            return False
    return release_worker_pool()


def parquet_uncompressed_bytes_per_row(file_path):
    """Average uncompressed Parquet payload per top-level row.

    File size is a poor proxy for nested numeric data because Parquet encoding
    and compression disappear as soon as Polars materialises it. Metadata gives
    a cheap estimate without reading the payload itself.
    """
    import glob

    import pyarrow.parquet as pq

    paths = (
        sorted(glob.glob(os.path.join(file_path, "*.parquet")))
        if os.path.isdir(file_path)
        else [file_path]
    )
    total_rows = 0
    total_bytes = 0
    for path in paths:
        metadata = pq.ParquetFile(path).metadata
        total_rows += int(metadata.num_rows)
        for row_group_idx in range(metadata.num_row_groups):
            row_group = metadata.row_group(row_group_idx)
            for column_idx in range(row_group.num_columns):
                total_bytes += int(
                    row_group.column(column_idx).total_uncompressed_size
                )
    if total_rows <= 0 or total_bytes <= 0:
        raise ValueError(f"No Parquet row-size metadata found in {file_path}")
    return max(1, (total_bytes + total_rows - 1) // total_rows)


def plan_axis_chunks(
    shape,
    splittable_axes,
    bytes_per_element,
    *,
    device="cpu",
    n_workers=1,
    margin=0.65,
    max_bytes=None,
    min_chunk=1,
) -> dict[int, int]:
    """Greedy memory-aware chunk sizes for an N-dim tensor.

    Returns ``{axis: chunk_size}`` for every axis in ``splittable_axes`` such
    that one block — ``prod(chunk over splittable) * prod(full over the rest)
    * bytes_per_element`` — fits the budget. Non-splittable axes stay whole.
    The largest current chunk is halved each step, so blocks stay roughly
    balanced instead of collapsing one axis to 1. ``max_bytes`` (when > 0)
    overrides the auto budget; otherwise the budget is
    ``free_memory_bytes(device) * margin / n_workers``.
    """
    shape = [max(1, int(s)) for s in shape]
    splittable = [ax for ax in splittable_axes if 0 <= ax < len(shape)]
    if max_bytes is not None and max_bytes > 0:
        budget = int(max_bytes)
    else:
        budget = int(
            free_memory_bytes(device) * float(margin) / max(1, int(n_workers))
        )
    bytes_per_element = max(1, int(bytes_per_element))
    budget = max(budget, bytes_per_element)
    fixed = 1
    for ax in range(len(shape)):
        if ax not in splittable:
            fixed *= shape[ax]
    chunks = {ax: shape[ax] for ax in splittable}

    def block_bytes():
        prod = fixed
        for ax in splittable:
            prod *= chunks[ax]
        return prod * bytes_per_element

    guard = 0
    while block_bytes() > budget and guard < 4096:
        guard += 1
        candidates = [ax for ax in splittable if chunks[ax] > min_chunk]
        if not candidates:
            break
        ax = max(candidates, key=lambda a: chunks[a])
        chunks[ax] = max(int(min_chunk), chunks[ax] // 2)
    return chunks


def iter_nd_blocks(shape, chunks):
    """Yield tuples of slices tiling ``shape`` into ``chunks`` blocks.

    ``chunks`` is ``{axis: size}``; axes absent from it are taken whole.
    """
    import itertools as _it

    ranges = []
    for ax, dim in enumerate(shape):
        dim = int(dim)
        step = int(chunks.get(ax, dim)) or dim
        step = max(1, step)
        ranges.append(
            [(s, min(s + step, dim)) for s in range(0, dim, step)]
            or [(0, dim)]
        )
    for combo in _it.product(*ranges):
        yield tuple(slice(a, b) for (a, b) in combo)


def suggest_row_batch_size(
    bytes_per_row,
    *,
    n_workers=1,
    margin=0.65,
    min_rows=1,
    max_rows=None,
):
    """Auto row/config batch size for the Parquet-streaming pages.

    ``bytes_per_row`` is the caller's estimate of the peak working-set cost of
    one row/config (exploded columns + intermediate frames). Returns how many
    rows fit the effective free RAM (host and cgroup limit) × margin ÷
    workers, clamped to ``[min_rows, max_rows]``.
    """
    bytes_per_row = max(1, int(bytes_per_row))
    free = available_memory_bytes()
    budget = int(free * float(margin) / max(1, int(n_workers)))
    rows = max(int(min_rows), budget // bytes_per_row)
    if max_rows is not None:
        rows = min(int(max_rows), rows)
    return int(rows)
