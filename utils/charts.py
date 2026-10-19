"""Custom Plotly chart rendering, animation payload limits, the time control
and the frame slider.
"""

import pickle
from copy import deepcopy

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from streamlit.elements.lib.layout_utils import Width

from utils.axis_metadata import find_axis_metadata
from utils.config_display import (
    find_plotly_configs,
    load_plot_config,
    unflatten_dict,
)

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
    if getattr(fig, "frames", None):  # animations go to the fast step player
        plotly_chart(fig, key=key)
        return
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


def plotly_chart(fig, *, key=None, height=None, **kwargs):
    """``st.plotly_chart``, except a figure with ``frames`` is shown by the
    fast step player (:mod:`fast_player`): the same frames packed once,
    redrawn one at a time with ``Plotly.react`` — long animations no longer
    ship every frame as JSON or hang the browser. Follows the Streamlit theme."""
    if not getattr(fig, "frames", None):
        return st.plotly_chart(fig, key=key, **kwargs)
    import fast_player

    plot_h = int(height or fig.layout.height or 600)
    html = fast_player.from_figure(fig, key=str(key or "fastplayer"), height=plot_h)
    return st.iframe(html, height=plot_h + 60)


def plotly_animation_html(fig, *, key="fastplayer") -> str:
    """Standalone HTML of an animated figure for download (fast player)."""
    import fast_player

    return fast_player.from_figure(fig, standalone=True, key=key)


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
