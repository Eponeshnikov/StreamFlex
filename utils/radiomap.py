"""Object colours and radio maps (planar and mesh) on a Plotly figure."""

import drjit as dr
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sionna.rt import MeshRadioMap, PlanarRadioMap


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
