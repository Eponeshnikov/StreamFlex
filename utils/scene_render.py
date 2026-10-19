"""The one 3D scene renderer and the ray paths drawn on it."""

import traceback

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sionna.rt import MeshRadioMap, PlanarRadioMap

from utils.radiomap import (
    add_mesh_radiomap_to_figure,
    add_planar_radiomap_to_figure,
)
from utils.scene_mesh import (
    SIONNA_PATH_COLORS,
    scene_display_controls,
    scene_mesh_traces,
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
