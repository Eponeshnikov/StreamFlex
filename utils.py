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
import traceback

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
):
    """
    Render a Sionna scene using Plotly in Streamlit.
    """
    fig = go.Figure()

    path_colors = {
        "los": "gray",
        "specular": "cornflowerblue",
        "diffuse": "darkseagreen",
        "refraction": "lightcoral",
        "diffraction": "blueviolet",
    }
    path_widths = {"los": 4}

    if show_objects:
        for obj_name, obj in scene.objects.items():
            try:
                mesh = obj.mi_mesh
                vertices = mesh.vertex_positions_buffer().numpy()
                faces = mesh.faces_buffer().numpy()

                x, y, z = vertices[0::3], vertices[1::3], vertices[2::3]
                i, j, k = faces[0::3], faces[1::3], faces[2::3]

                obj_color = get_object_color(obj)

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
            except Exception:
                pass

    # Only render selected transmitters
    for tx_name, tx in scene.transmitters.items():
        # If selected_tx_names is None or empty, render all TX
        if (
            selected_tx_names is None
            or len(selected_tx_names) == 0
            or tx_name in selected_tx_names
        ):
            pos = tx.position.numpy()
            fig.add_trace(
                go.Scatter3d(
                    x=[float(pos[0])],
                    y=[float(pos[1])],
                    z=[float(pos[2])],
                    mode="markers",
                    marker=dict(size=8, color="red", symbol="circle"),
                    name=f"TX: {tx_name}",
                    showlegend=show_legend,
                    legendgroup="transmitters",
                    legendgrouptitle_text="Transmitters",
                )
            )

    # Only render selected receivers
    for rx_name, rx in scene.receivers.items():
        # If selected_rx_names is None or empty, render all RX
        if (
            selected_rx_names is None
            or len(selected_rx_names) == 0
            or rx_name in selected_rx_names
        ):
            pos = rx.position.numpy()
            fig.add_trace(
                go.Scatter3d(
                    x=[float(pos[0])],
                    y=[float(pos[1])],
                    z=[float(pos[2])],
                    mode="markers",
                    marker=dict(size=8, color="green", symbol="circle"),
                    name=f"RX: {rx_name}",
                    showlegend=show_legend,
                    legendgroup="receivers",
                    legendgrouptitle_text="Receivers",
                )
            )

    if show_paths and paths is not None:
        # --- CHANGE: Pass the scene object to the path rendering function ---
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
        )

    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        title="Sionna Scene Visualization",
        showlegend=show_legend,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
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
):
    """
    Add propagation paths to the Plotly figure.
    """
    added_to_legend = set()

    try:
        vertices_np = paths.vertices.numpy()
        interactions_np = paths.interactions.numpy()
        valid_np = paths.valid.numpy()

        source_positions_np = np.stack(
            [
                paths.sources.x.numpy(),
                paths.sources.y.numpy(),
                paths.sources.z.numpy(),
            ],
            axis=1,
        )

        target_positions_np = np.stack(
            [
                paths.targets.x.numpy(),
                paths.targets.y.numpy(),
                paths.targets.z.numpy(),
            ],
            axis=1,
        )

        # --- FIX: Get geometric antenna counts from the scene ---
        # This is the number of physical antenna elements per device.
        num_ant_per_tx = scene.tx_array.array_size
        num_ant_per_rx = scene.rx_array.array_size

        is_synthetic = paths.synthetic_array

        if is_synthetic:
            max_depth, num_rx, num_tx, num_paths, _ = vertices_np.shape
            num_rx_ant_paths, num_tx_ant_paths = 1, 1
        else:
            # These dimensions might include polarization, so we name them _paths
            (
                max_depth,
                num_rx,
                num_rx_ant_paths,
                num_tx,
                num_tx_ant_paths,
                num_paths,
                _,
            ) = vertices_np.shape

        for rx_idx in range(num_rx):
            for tx_idx in range(num_tx):
                # Get the actual TX and RX names to check if they're selected
                tx_name = (
                    list(scene.transmitters.keys())[tx_idx]
                    if tx_idx < len(scene.transmitters)
                    else f"tx{tx_idx}"
                )
                rx_name = (
                    list(scene.receivers.keys())[rx_idx]
                    if rx_idx < len(scene.receivers)
                    else f"rx{rx_idx}"
                )

                # Check if both TX and RX are selected (or if no selection is made)
                tx_selected = (
                    selected_tx_names is None
                    or len(selected_tx_names) == 0
                    or tx_name in selected_tx_names
                )
                rx_selected = (
                    selected_rx_names is None
                    or len(selected_rx_names) == 0
                    or rx_name in selected_rx_names
                )

                # Skip if either TX or RX is not selected
                if not tx_selected or not rx_selected:
                    continue

                # Loop over the antenna dimension from the paths tensor
                for rx_ant_idx in range(num_rx_ant_paths):
                    for tx_ant_idx in range(num_tx_ant_paths):
                        for path_idx in range(num_paths):
                            if is_synthetic:
                                is_valid = valid_np[rx_idx, tx_idx, path_idx]
                            else:
                                is_valid = valid_np[
                                    rx_idx,
                                    rx_ant_idx,
                                    tx_idx,
                                    tx_ant_idx,
                                    path_idx,
                                ]

                            if not is_valid:
                                continue

                            path_type = "los"
                            for depth_idx in range(max_depth):
                                if is_synthetic:
                                    interaction_type = int(
                                        interactions_np[
                                            depth_idx, rx_idx, tx_idx, path_idx
                                        ]
                                    )
                                else:
                                    interaction_type = int(
                                        interactions_np[
                                            depth_idx,
                                            rx_idx,
                                            rx_ant_idx,
                                            tx_idx,
                                            tx_ant_idx,
                                            path_idx,
                                        ]
                                    )
                                if interaction_type != 0:
                                    path_type = get_path_type_name(
                                        interaction_type
                                    )
                                    break

                            color = path_colors.get(path_type, "gray")
                            width = path_widths.get(path_type, 1)

                            # --- FIX: Calculate flat index using GEOMETRIC counts ---
                            # This correctly maps the path back to its physical antenna position,
                            # even if the paths tensor has extra dimensions for polarization.
                            flat_tx_ant_idx = tx_idx * num_ant_per_tx + (
                                tx_ant_idx % num_ant_per_tx
                            )
                            source_pos = source_positions_np[flat_tx_ant_idx]

                            path_coords = [source_pos]

                            for depth_idx in range(max_depth):
                                if is_synthetic:
                                    vertex = vertices_np[
                                        depth_idx, rx_idx, tx_idx, path_idx
                                    ]
                                else:
                                    vertex = vertices_np[
                                        depth_idx,
                                        rx_idx,
                                        rx_ant_idx,
                                        tx_idx,
                                        tx_ant_idx,
                                        path_idx,
                                    ]
                                if np.any(vertex != 0):
                                    path_coords.append(vertex)

                            # --- FIX: Calculate flat index using GEOMETRIC counts ---
                            flat_rx_ant_idx = rx_idx * num_ant_per_rx + (
                                rx_ant_idx % num_ant_per_rx
                            )
                            target_pos = target_positions_np[flat_rx_ant_idx]
                            path_coords.append(target_pos)

                            path_x, path_y, path_z = zip(
                                *[p.tolist() for p in path_coords]
                            )
                            if (
                                len(path_x) > 2
                            ):  # Has intermediate vertices beyond the TX position
                                path_x, path_y, path_z = (
                                    path_x[:-2] + (path_x[-1],),
                                    path_y[:-2] + (path_y[-1],),
                                    path_z[:-2] + (path_z[-1],),
                                )

                            show_in_legend = show_legend and (
                                path_type not in added_to_legend
                            )
                            if show_in_legend:
                                added_to_legend.add(path_type)

                            # Determine opacity based on path type and global opacity
                            path_opacity = global_path_opacity  # Default to global opacity
                            if path_type == "specular":
                                path_opacity = global_path_opacity if global_path_opacity else specular_opacity if specular_opacity else 1.0
                            elif path_type == "diffuse":
                                path_opacity = global_path_opacity  if global_path_opacity else diffuse_opacity if diffuse_opacity else 1.0
                            elif path_type == "refraction":
                                path_opacity = global_path_opacity  if global_path_opacity else refraction_opacity if refraction_opacity else 1.0
                            elif path_type == "diffraction":
                                path_opacity = global_path_opacity  if global_path_opacity else diffraction_opacity if diffraction_opacity else 1.0
                            elif path_type == "los":
                                path_opacity = 1.0
                            # Note: LOS paths use global opacity only, as requested (excluded from individual controls)

                            fig.add_trace(
                                go.Scatter3d(
                                    x=path_x,
                                    y=path_y,
                                    z=path_z,
                                    mode="lines",
                                    line=dict(color=color, width=width),
                                    opacity=path_opacity,
                                    name=path_type.replace("_", " ").title(),
                                    showlegend=show_in_legend,
                                    legendgroup=f"paths_{path_type}",
                                    hoverinfo="name",
                                )
                            )
    except Exception as e:
        st.error(f"Could not render paths: {str(e)}")
        st.code(traceback.format_exc())


def get_path_type_name(interaction_type):
    """
    Convert interaction type constant to path type name.
    """
    if interaction_type == 1:
        return "specular"
    elif interaction_type == 2:
        return "diffuse"
    elif interaction_type == 8:
        return "diffraction"
    elif interaction_type == 4:
        return "refraction"
    else:
        return "los"
