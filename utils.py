import ast
import hashlib

# custom_plotter.py
import json
import numbers
import os
import pickle
import sys
from copy import deepcopy
from typing import Type

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from loguru import logger
import streamlit.components.v1 as components
import re

# --- UTILITY FUNCTIONS ---


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


# --- MAIN RENDERING FUNCTION ---


def render_custom_plotly_chart(
    fig: go.Figure,
    use_container_width: bool = True,
    key: str | None = None,
):
    """
    Renders a Plotly chart with an in-app toggle to enable advanced styling controls.

    By default, a standard Streamlit chart is shown. A toggle switch allows the user
    to access custom theme selection and interactive styling options.

    Args:
        fig (go.Figure): The Plotly figure object to render.
        use_container_width (bool, optional): Expand the chart to container's width. Defaults to True.
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
    toggle_col, save_col = st.columns([1, 4])

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
        use_container_width=True,
    )

    # --- RENDER LOGIC ---
    # If custom mode is OFF, display a standard chart
    if not st.session_state[custom_mode_key]:
        st.plotly_chart(
            fig,
            use_container_width=use_container_width,
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
            st.plotly_chart(fig, use_container_width=use_container_width)
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
        with top_cols[1]:
            with st.popover("⚙️ Options"):
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
                use_container_width=use_container_width,
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
    data, save_flag, shape=None, dtype: np.dtype | Type = np.complex64
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

        # Type checking for single values (won't apply directly to list strings)
        if expected_type == "int" and not isinstance(val, int):
            # Special case: Check if it's a list where evaluation happened
            if not isinstance(val, list):
                # self.logger.warning(f"Type mismatch: Expected int, got {type(val)} for '{value_str}'.")
                raise ValueError(f"Expected an integer, got {type(val)}")
        if expected_type == "float" and not isinstance(
            val, numbers.Number
        ):  # Allow int to be treated as float
            # Special case: Check if it's a list where evaluation happened
            if not isinstance(val, list):
                # self.logger.warning(f"Type mismatch: Expected float, got {type(val)} for '{value_str}'.")
                raise ValueError(f"Expected a float, got {type(val)}")
        if expected_type == "str" and not isinstance(val, str):
            # Special case: Check if it's a list where evaluation happened
            if not isinstance(val, list):
                # self.logger.warning(f"Type mismatch: Expected str, got {type(val)} for '{value_str}'.")
                raise ValueError(f"Expected a string, got {type(val)}")

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
        return f"<span style='color: red'>Error reading logs: {str(e)}</span>"


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
                except (IOError, pickle.PickleError, EOFError):
                    pass
            result = func(*args, **kwargs)
            cached_data = result
            os.makedirs(CACHE_FOLDER, exist_ok=True)
            with open(cache_file, "wb") as file:
                pickle.dump(cached_data, file)

            return result

        return wrapper

    return decorator


def extract_scene_data(_scene, _paths, clip_at=20, resolution=(1400, 600)):
    """
    Extract scene data from Sionna preview for Three.js rendering.

    Args:
        _scene: Sionna scene object (prefixed with _ to avoid hashing)
        _paths: Computed paths from PathSolver (prefixed with _ to avoid hashing)
        clip_at: Clipping parameter for preview
        resolution: Preview resolution tuple (width, height)

    Returns:
        tuple: (geometry_data, paths_data, points_data, camera_info)
    """
    # Render scene preview to get widget data
    _scene.preview(paths=_paths, clip_at=clip_at, resolution=resolution)
    widget = _scene._preview_widget

    if not widget:
        raise ValueError("Widget not created")

    html = widget._repr_html_()
    html_file = open("filename.html", "w")
    html_file.write(html)
    html_file.close()
    match = re.search(
        r'<script type="application/vnd\.jupyter\.widget-state\+json">\s*(\{.*?\})\s*</script>',
        html,
        re.DOTALL,
    )

    if not match:
        raise ValueError("Could not extract widget state")

    widget_state = json.loads(match.group(1))
    models = widget_state["state"]

    # Extract components
    renderer_data = None
    camera_data = None
    geometries = []
    line_segments_geometries = []
    points_models = []

    for model_id, model in models.items():
        model_name = model["model_name"]
        if model_name == "RendererModel":
            renderer_data = model["state"]
        elif model_name == "PerspectiveCameraModel":
            camera_data = model["state"]
        elif model_name == "BufferGeometryModel":
            geometries.append((model_id, model))
        elif model_name == "LineSegmentsGeometryModel":
            line_segments_geometries.append((model_id, model))
        elif model_name == "PointsModel":
            points_models.append(model)

    if not (renderer_data and camera_data):
        raise ValueError("Missing renderer or camera data")

    width = renderer_data.get("_width", 800)
    height = renderer_data.get("_height", 600)
    camera_pos = camera_data.get("position", [100, 100, 100])
    camera_fov = camera_data.get("fov", 45)

    # Process scene geometry (large meshes)
    geometry_data_js = []
    for geom_id, geom_model in geometries:
        attributes = geom_model["state"].get("attributes", {})

        position_data = None
        index_data = None
        color_data = None

        if "position" in attributes:
            pos_id = attributes["position"].replace("IPY_MODEL_", "")
            if pos_id in models:
                pos_model = models[pos_id]
                pos_shape = (
                    pos_model.get("state", {})
                    .get("array", {})
                    .get("shape", [])
                )
                # Only large buffers (scene geometry)
                if len(pos_shape) == 2 and pos_shape[0] > 100:
                    pos_buffers = pos_model.get("buffers", [])
                    if pos_buffers:
                        position_data = pos_buffers[0]["data"]

        if "index" in attributes and position_data:
            idx_id = attributes["index"].replace("IPY_MODEL_", "")
            if idx_id in models:
                idx_buffers = models[idx_id].get("buffers", [])
                if idx_buffers:
                    index_data = idx_buffers[0]["data"]

        if "color" in attributes and position_data:
            col_id = attributes["color"].replace("IPY_MODEL_", "")
            if col_id in models:
                col_buffers = models[col_id].get("buffers", [])
                if col_buffers:
                    color_data = col_buffers[0]["data"]

        if position_data:
            geometry_data_js.append(
                {
                    "position": position_data,
                    "index": index_data,
                    "color": color_data,
                }
            )

    # Process paths (line segments)
    paths_data_js = []
    for geom_id, geom_model in line_segments_geometries:
        buffers = geom_model.get("buffers", [])

        position_data = None
        color_data = None

        for buffer in buffers:
            path = buffer.get("path", [])
            data = buffer.get("data", "")
            if path == ["positions", "buffer"]:
                position_data = data
            elif path == ["colors", "buffer"]:
                color_data = data

        if position_data:
            paths_data_js.append(
                {
                    "positions": position_data,
                    "colors": color_data,
                }
            )

    # Process TX/RX points
    points_data_js = []

    # Method 1: Get from PointsModel
    for points_model in points_models:
        geometry_ref = points_model.get("geometry", "")
        if geometry_ref:
            geom_id = geometry_ref.replace("IPY_MODEL_", "")
            if geom_id in models:
                geom_model = models[geom_id]
                attributes = geom_model["state"].get("attributes", {})

                position_data = None
                color_data = None

                if "position" in attributes:
                    pos_id = attributes["position"].replace("IPY_MODEL_", "")
                    if pos_id in models:
                        pos_buffers = models[pos_id].get("buffers", [])
                        if pos_buffers:
                            position_data = pos_buffers[0]["data"]

                if "color" in attributes:
                    col_id = attributes["color"].replace("IPY_MODEL_", "")
                    if col_id in models:
                        col_buffers = models[col_id].get("buffers", [])
                        if col_buffers:
                            color_data = col_buffers[0]["data"]

                if position_data:
                    points_data_js.append(
                        {
                            "position": position_data,
                            "color": color_data,
                        }
                    )

    # Method 2: Also check small geometries directly (TX/RX as points)
    for geom_id, geom_model in geometries:
        attributes = geom_model["state"].get("attributes", {})

        if "position" in attributes:
            pos_id = attributes["position"].replace("IPY_MODEL_", "")
            if pos_id in models:
                pos_model = models[pos_id]
                pos_shape = (
                    pos_model.get("state", {})
                    .get("array", {})
                    .get("shape", [])
                )
                # Small buffers (TX/RX) - shape [2, 3]
                if len(pos_shape) == 2 and pos_shape[0] == 2:
                    pos_buffers = pos_model.get("buffers", [])
                    if pos_buffers:
                        position_data = pos_buffers[0]["data"]

                        color_data = None
                        if "color" in attributes:
                            col_id = attributes["color"].replace(
                                "IPY_MODEL_", ""
                            )
                            if col_id in models:
                                col_buffers = models[col_id].get("buffers", [])
                                if col_buffers:
                                    color_data = col_buffers[0]["data"]

                        points_data_js.append(
                            {
                                "position": position_data,
                                "color": color_data,
                            }
                        )

    camera_info = {
        "width": width,
        "height": height,
        "camera_pos": camera_pos,
        "camera_fov": camera_fov,
    }

    return geometry_data_js, paths_data_js, points_data_js, camera_info


def create_threejs_html(
    geometry_data_js, paths_data_js, points_data_js, camera_info
):
    """
    Create Three.js HTML from extracted scene data.

    Args:
        geometry_data_js: Processed geometry data
        paths_data_js: Processed paths data
        points_data_js: Processed points data
        camera_info: Camera information dictionary

    Returns:
        str: Complete HTML string for Three.js rendering
    """
    geometry_json = json.dumps(geometry_data_js)
    paths_json = json.dumps(paths_data_js)
    points_json = json.dumps(points_data_js)

    width = camera_info["width"]
    height = camera_info["height"]
    camera_pos = camera_info["camera_pos"]
    camera_fov = camera_info["camera_fov"]

    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ margin: 0; padding: 0; overflow: hidden; background: #87CEEB; }}
        #container {{ width: 100%; height: 100vh; }}
        #info {{
            position: absolute; top: 10px; left: 10px;
            background: rgba(0,0,0,0.7); color: white;
            padding: 10px; border-radius: 5px;
            font-family: monospace; font-size: 12px; z-index: 100;
        }}
        #legend {{
            position: absolute; bottom: 10px; left: 10px;
            background: rgba(255,255,255,0.95); color: black;
            padding: 15px; border-radius: 5px;
            font-family: sans-serif; font-size: 13px; z-index: 100;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}
        .legend-item {{
            display: flex; align-items: center;
            margin: 8px 0;
        }}
        .legend-color {{
            width: 30px; height: 3px;
            margin-right: 10px;
            display: inline-block;
        }}
        .legend-point {{
            width: 12px; height: 12px;
            border-radius: 50%;
            margin-right: 10px;
            display: inline-block;
        }}
    </style>
</head>
<body>
    <div id="info">
        <strong>Sionna Scene with Paths</strong><br>
        Left mouse: Rotate | Right mouse: Pan | Scroll: Zoom
    </div>
    <div id="legend">
        <strong>Legend</strong>
        <div class="legend-item">
            <span class="legend-color" style="background-color: rgb(127, 127, 127);"></span>
            <span>Line-of-sight</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background-color: rgb(153, 153, 255);"></span>
            <span>Specular reflection</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background-color: rgb(255, 153, 153);"></span>
            <span>Refraction</span>
        </div>
        <div class="legend-item">
            <span class="legend-point" style="background-color: rgb(255, 0, 0);"></span>
            <span>Transmitter</span>
        </div>
        <div class="legend-item">
            <span class="legend-point" style="background-color: rgb(102, 204, 102);"></span>
            <span>Receiver</span>
        </div>
    </div>
    <div id="container"></div>
    
    <script type="importmap">
    {{
        "imports": {{
            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
        }}
    }}
    </script>
    
    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
        
        const geometryData = {geometry_json};
        const pathsData = {paths_json};
        const pointsData = {points_json};
        
        function base64ToArrayBuffer(base64) {{
            const binaryString = atob(base64);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {{
                bytes[i] = binaryString.charCodeAt(i);
            }}
            return bytes.buffer;
        }}
        
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xffffff);
        
        const camera = new THREE.PerspectiveCamera(
            {camera_fov}, {width} / {height}, 0.1, 20000
        );
        camera.position.set({camera_pos[0]}, {camera_pos[1]}, {camera_pos[2]});
        camera.up.set(0, 0, 1);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize({width}, {height});
        document.getElementById('container').appendChild(renderer.domElement);
        
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.25);
        camera.add(directionalLight);
        scene.add(camera);
        
        // Render scene geometry
        for (const geomData of geometryData) {{
            try {{
                const posBuffer = base64ToArrayBuffer(geomData.position);
                const positions = new Float32Array(posBuffer);
                
                const idxBuffer = base64ToArrayBuffer(geomData.index);
                const indices = new Uint32Array(idxBuffer);
                
                let colors = null;
                if (geomData.color) {{
                    const colBuffer = base64ToArrayBuffer(geomData.color);
                    colors = new Float32Array(colBuffer);
                }}
                
                const geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                geometry.setIndex(new THREE.BufferAttribute(indices, 1));
                
                if (colors) {{
                    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                }}
                
                const material = new THREE.MeshStandardMaterial({{
                    vertexColors: colors ? true : false,
                    flatShading: true,
                    roughness: 1.0,
                    metalness: 0.0,
                    side: THREE.DoubleSide
                }});
                
                const mesh = new THREE.Mesh(geometry, material);
                scene.add(mesh);
            }} catch (error) {{
                console.error('Error processing geometry:', error);
            }}
        }}
        
        // Render paths
        console.log('Processing', pathsData.length, 'path groups');
        for (const pathData of pathsData) {{
            try {{
                // Decode positions buffer - shape is [153, 2, 3]
                const posBuffer = base64ToArrayBuffer(pathData.positions);
                const positions = new Float32Array(posBuffer);
                
                // Decode colors buffer if available
                let colors = null;
                if (pathData.colors) {{
                    const colBuffer = base64ToArrayBuffer(pathData.colors);
                    colors = new Float32Array(colBuffer);
                }}
                
                // Create geometry with the flattened positions
                const geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                
                if (colors) {{
                    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                }}
                
                // Use LineSegments for proper rendering
                const material = new THREE.LineBasicMaterial({{
                    vertexColors: colors ? true : false,
                    linewidth: 2
                }});
                
                const lineSegments = new THREE.LineSegments(geometry, material);
                scene.add(lineSegments);
                
                console.log('Added path with', positions.length / 3, 'vertices');
            }} catch (error) {{
                console.error('Error processing paths:', error);
            }}
        }}
        
        // Render TX/RX points
        console.log('Processing', pointsData.length, 'point groups');
        for (const pointData of pointsData) {{
            try {{
                const posBuffer = base64ToArrayBuffer(pointData.position);
                const positions = new Float32Array(posBuffer);
                
                let colors = null;
                if (pointData.color) {{
                    const colBuffer = base64ToArrayBuffer(pointData.color);
                    colors = new Float32Array(colBuffer);
                }}
                
                const geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                
                if (colors) {{
                    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                }}
                
                // Larger spheres for better visibility
                const numPoints = positions.length / 3;
                for (let i = 0; i < numPoints; i++) {{
                    const sphereGeometry = new THREE.SphereGeometry(3, 16, 16);
                    const sphereMaterial = new THREE.MeshStandardMaterial({{
                        color: colors ? new THREE.Color(colors[i*3], colors[i*3+1], colors[i*3+2]) : 0xff0000,
                        emissive: colors ? new THREE.Color(colors[i*3], colors[i*3+1], colors[i*3+2]) : 0xff0000,
                        emissiveIntensity: 0.3
                    }});
                    const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
                    sphere.position.set(positions[i*3], positions[i*3+1], positions[i*3+2]);
                    scene.add(sphere);
                }}
                
                console.log('Added', numPoints, 'TX/RX points');
            }} catch (error) {{
                console.error('Error processing points:', error);
            }}
        }}
        
        console.log('Scene setup complete');
        
        const axesHelper = new THREE.AxesHelper(500);
        scene.add(axesHelper);
        
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
        
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>
    """


def render_scene_with_paths(scene, paths, clip_at=20, resolution=(1400, 600)):
    """
    Render Sionna scene with paths using Three.js in Streamlit.

    Args:
        scene: Sionna scene object
        paths: Computed paths from PathSolver
        clip_at: Clipping parameter for preview
        resolution: Preview resolution tuple (width, height)

    Returns:
        bool: True if rendering was successful
    """
    try:
        # Extract scene data (without caching since scene object can't be hashed)
        geometry_data, paths_data, points_data, camera_info = (
            extract_scene_data(scene, paths, clip_at, resolution)
        )

        # Generate HTML with caching based on the extracted data
        @st.cache_resource(ttl=300)
        def get_cached_html(
            _geometry_data, _paths_data, _points_data, _camera_info
        ):
            return create_threejs_html(
                _geometry_data, _paths_data, _points_data, _camera_info
            )

        threejs_html = get_cached_html(
            geometry_data, paths_data, points_data, camera_info
        )

        # Render the component
        components.html(
            threejs_html,
            height=camera_info["height"] + 50,
            scrolling=False,
        )

        return True
    except Exception as e:
        st.error(f"Error rendering scene: {str(e)}")
        return False
