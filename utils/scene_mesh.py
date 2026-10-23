"""Scene loading, layers and the welded per-material mesh traces."""

import hashlib
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.radiomap import get_object_color


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
    return load_scene_file(scene_path, merge_shapes=merge_shapes)


_SCENE_LOAD_LOCK = threading.Lock()


def load_scene_file(scene_path: str, merge_shapes: bool = True):
    """``sionna.rt.load_scene`` that resolves a scene's relative mesh paths
    on any thread.

    OSM-built scenes reference ``meshes/<shape>.ply`` relative to the XML.
    Mitsuba resolves them through the *calling thread's* file resolver, and
    the background cache refresh runs on a pool thread: there it failed with
    "PLY file ... not found" while the same scene loaded fine on the script
    thread. The scene's directory is put in front of a private copy of the
    resolver for the duration of the load, and loads are serialized so two
    of them never share one resolver half-way through.
    """
    import mitsuba as mi
    from sionna.rt import load_scene

    scene_dir = os.path.dirname(os.path.abspath(scene_path))
    with _SCENE_LOAD_LOCK:
        thread = mi.Thread.thread()
        old = thread.file_resolver()
        resolver = mi.FileResolver()
        resolver.clear()
        resolver.append(scene_dir)
        for entry in old:
            resolver.append(str(entry))
        thread.set_file_resolver(resolver)
        try:
            return load_scene(scene_path, merge_shapes=merge_shapes)
        finally:
            thread.set_file_resolver(old)


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
    # LOD3 facades: glass and frames (`windows*`) plus railings,
    # canopies and steel doors (`facade_*`).
    if low.startswith(("window", "facade")):
        return "windows"
    if low.startswith("tree"):
        return "trees"
    if low.startswith(("road", "bridge")):
        return "roads"
    # Parked cars are inserted into a loaded scene rather than written into
    # it (pages/osm_scene/cars.py), always as one merged shape; moving cars
    # and walking people per velocity group.
    if low.startswith(("parked_cars", "car_", "moving_cars", "moving_people")):
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
