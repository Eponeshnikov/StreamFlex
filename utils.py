import ast
import hashlib

# custom_plotter.py
import io
import json
import numbers
import os
import pickle
import sys
import traceback
from copy import deepcopy
from typing import Literal, cast, overload

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
    # Serialize the figure to bytes for download
    fig_bytes = pickle.dumps(fig)
    st.download_button(
        label="💾 Save Fig",
        data=fig_bytes,
        file_name=f"{key}_figure.pickle",
        mime="application/octet-stream",
        key=f"{key}_save_pickle_btn",
        help="Save the current figure as a pickle file",
        width="stretch",
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


def generate_cache_filename(func, *args, **kwargs):
    """
    Generate a unique cache filename based on the function name, arguments, and keyword arguments.

    This function takes a function object, positional arguments, and keyword arguments as input.
    It combines the function name, arguments, and keyword arguments into a tuple, serializes the tuple,
    and computes the SHA-256 hash of the serialized data. The hash is then used to generate a unique
    filename with a '.pickle' extension. The filename is returned as a string.

    Parameters
    ----------
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
    return os.path.join(CACHE_FOLDER, filename)


def cache_result(reset=False):
    """
    A decorator function that caches the results of a function and stores them in a cache file.

    Parameters
    ----------
    reset (bool): If True, the cache file will be deleted and the function will be executed again.
                If False (default), the function will attempt to load the result from the cache file.

    Returns:
    function: The decorated function, which will either return the cached result or execute the function
            and store the result in the cache file.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_file = generate_cache_filename(func, *args, **kwargs)
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
            os.makedirs(CACHE_FOLDER, exist_ok=True)
            with open(cache_file, "wb") as file:
                pickle.dump(cached_data, file)

            return result

        return wrapper

    return decorator


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


def _collect_scene_meshes(scene):
    """``(name, (N, 3) vertices, flat faces)`` for every scene object."""
    meshes = []
    for obj_name, obj in scene.objects.items():
        try:
            mesh = obj.mi_mesh
            vertices = mesh.vertex_positions_buffer().numpy()
            faces = mesh.faces_buffer().numpy()
        except Exception:
            continue
        meshes.append((obj_name, vertices.reshape(-1, 3), faces))
    return meshes


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
    color_paths_by_segment=False,
    show_segment_toggle=True,
    widget_key="scene",
) -> go.Figure:
    """
    Render a Sionna scene using Plotly in Streamlit.

    Scene meshes are reduced losslessly by welding exactly coincident
    vertices.
    """
    fig = go.Figure()

    # Match Sionna's own path palette exactly
    # (sionna.rt.constants: LOS/SPECULAR/DIFFUSE/REFRACTION/DIFFRACTION_COLOR).
    path_colors = {
        "los": "rgb(128,128,128)",  # (0.5, 0.5, 0.5)
        "specular": "rgb(153,153,255)",  # (0.6, 0.6, 1.0)
        "diffuse": "rgb(153,255,153)",  # (0.6, 1.0, 0.6)
        "refraction": "rgb(255,153,153)",  # (1.0, 0.6, 0.6)
        "diffraction": "rgb(153,0,153)",  # (0.6, 0.0, 0.6)
    }
    path_widths = {"los": 4}

    if show_objects:
        meshes = _collect_scene_meshes(scene)
        welded = [
            (name, _weld_mesh(vertices, faces))
            for name, vertices, faces in meshes
        ]
        triangle_count = sum(len(parts[3]) for _, parts in welded)
        vertex_count_full = sum(len(vertices) for _, vertices, _ in meshes)
        vertex_count = sum(len(parts[0]) for _, parts in welded)
        fig.update_layout(
            meta={
                "geometry_reduction": "exact_vertex_welding",
                "triangles": triangle_count,
                "triangles_full": triangle_count,
                "vertices": vertex_count,
                "vertices_full": vertex_count_full,
            }
        )
        for obj_name, (x, y, z, i, j, k) in welded:
            if len(i) == 0:
                continue
            try:
                obj_color = get_object_color(scene.objects[obj_name])
            except Exception:
                obj_color = "lightblue"
            fig.add_trace(
                go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=i,
                    j=j,
                    k=k,
                    opacity=building_opacity,
                    color=obj_color,
                    name=obj_name,
                    showlegend=False,
                    hoverinfo="name",
                )
            )

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

    tx_names_to_render = selected_tx_names or list(scene.transmitters.keys())
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

    rx_names_to_render = selected_rx_names or list(scene.receivers.keys())
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
    color_paths_by_segment=False,
):
    """
    Add propagation paths to the Plotly figure.

    color_paths_by_segment:
        If False (default), each path is drawn as a single trace colored by
        its first interaction type. If True, each path is split into segments
        colored individually like Sionna's own renderer: the incident segment
        (TX -> first interaction) is gray (LoS color) and every subsequent
        segment takes the color of the interaction it leaves.
    """
    added_to_legend = set()

    opacity_map = {
        "specular": specular_opacity,
        "diffuse": diffuse_opacity,
        "refraction": refraction_opacity,
        "diffraction": diffraction_opacity,
    }

    def _segment_opacity(seg_type, is_true_los, path_primary_type):
        # Only a genuine LoS ray (TX -> RX, no interactions) stays fully opaque.
        # Every other segment -- including the gray *incident* segment of a
        # reflected path -- is subject to the transparency controls.
        if is_true_los:
            return 1.0
        if global_path_opacity is not None:
            return global_path_opacity
        # In per-type mode the gray incident segment has no slider of its own,
        # so it follows its path's primary interaction opacity.
        eff_type = seg_type if seg_type != "los" else path_primary_type
        return opacity_map.get(eff_type, 1.0)

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

        for rx_idx in range(num_rx):
            for tx_idx in range(num_tx):
                tx_name = list(scene.transmitters.keys())[tx_idx]
                rx_name = list(scene.receivers.keys())[rx_idx]

                if (
                    selected_tx_names and tx_name not in selected_tx_names
                ) or (selected_rx_names and rx_name not in selected_rx_names):
                    continue

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
                            else interactions_np[
                                depth, rx_idx, tx_idx, path_idx
                            ]
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
                    width = path_widths.get("los", 4) if is_true_los else 1

                    if not color_paths_by_segment:
                        # One trace per path, colored by its first interaction.
                        path_x, path_y, path_z = zip(*coords)
                        show_in_legend = show_legend and (
                            path_type not in added_to_legend
                        )
                        if show_in_legend:
                            added_to_legend.add(path_type)
                        fig.add_trace(
                            go.Scatter3d(
                                x=path_x,
                                y=path_y,
                                z=path_z,
                                mode="lines",
                                line={
                                    "color": path_colors.get(
                                        path_type, "gray"
                                    ),
                                    "width": width,
                                },
                                opacity=_segment_opacity(
                                    path_type, is_true_los, path_type
                                ),
                                name=path_type.replace("_", " ").title(),
                                showlegend=show_in_legend,
                                legendgroup=f"paths_{path_type}",
                                hoverinfo="name",
                            )
                        )
                    else:
                        # One trace per segment, colored like Sionna: the
                        # incident segment (TX -> first vertex) is LoS-gray,
                        # each later segment takes the type of its start vertex.
                        seg_types = ["los"] + vertex_types
                        for j, seg_type in enumerate(seg_types):
                            seg = coords[j : j + 2]
                            seg_x, seg_y, seg_z = zip(*seg)
                            show_in_legend = show_legend and (
                                seg_type not in added_to_legend
                            )
                            if show_in_legend:
                                added_to_legend.add(seg_type)
                            fig.add_trace(
                                go.Scatter3d(
                                    x=seg_x,
                                    y=seg_y,
                                    z=seg_z,
                                    mode="lines",
                                    line={
                                        "color": path_colors.get(
                                            seg_type, "gray"
                                        ),
                                        "width": width,
                                    },
                                    opacity=_segment_opacity(
                                        seg_type, is_true_los, path_type
                                    ),
                                    name=seg_type.replace("_", " ").title(),
                                    showlegend=show_in_legend,
                                    legendgroup=f"paths_{seg_type}",
                                    hoverinfo="name",
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


def visualization_time_control(
    plugin_instance,
    widget_manager,
    result,
    num_time_steps,
    *,
    key_prefix,
    container=None,
):
    """Shared Streamlit/smooth-Plotly time control for result visualizers.

    Returns ``(selected_time_index, animation_indices)``. The second item is
    ``None`` for the Streamlit slider and a list of absolute time indices for
    client-side Plotly animation. Trajectory metadata narrows both modes to the
    selected trajectory segment.
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
        return start, list(range(start, start + count))
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


def add_plotly_frame_slider(fig, labels, *, prefix="Time step: "):
    """Attach browser-side frame scrubbing and play/pause controls."""
    steps = [
        {
            "method": "animate",
            "label": str(label),
            "args": [
                [str(index)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": False},
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
                                "frame": {"duration": 120, "redraw": False},
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
                                "frame": {"duration": 0, "redraw": False},
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
):
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
