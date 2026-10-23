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


def release_scene_memory() -> int:
    """Drop every loaded scene and hand its VRAM back to the driver.

    A loaded scene keeps its geometry and OptiX acceleration structure on
    the card for as long as ``load_scene_cached`` holds it, and the
    in-process radio maps and ray casts leave Dr.Jit's cache behind; a page
    that is left calls this so the next one starts from a clean card. The
    next load of a scene is a fresh read (a second or two for an LOD3
    city). Returns the bytes handed back.
    """
    import gc

    from utils.gpu import release_gpu_memory

    load_scene_cached.clear()
    gc.collect()
    return release_gpu_memory()


def scene_file_path(scene_name) -> str | None:
    """The XML a scene name stands for: a path as given, or a Sionna scene."""
    name = str(scene_name or "")
    if os.path.isfile(name):
        return name
    import sionna.rt

    found = getattr(sionna.rt.scene, name, None)
    return str(found) if isinstance(found, str) and os.path.isfile(found) else None


def display_scene(scene_name):
    """The scene as the layer filter needs it (``merge_shapes=False``).

    A scene loaded for the ray tracer is merged, and merging throws away
    the shape names the layers go by -- land cover and LOD3 walls end up
    in one nameless shape with the rest. ``None`` when there is no file.
    """
    path = scene_file_path(scene_name)
    if path is None:
        return None
    try:
        return load_scene_cached(path, merge_shapes=False)
    except Exception:
        return None


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
    "lod3_walls",
    "terrain",
    "roads",
    "surfaces",
    "trees",
    "windows",
    "details",
    "vehicles",
    "other",
)

#: What the 3D views leave out until asked, measured in a headless browser
#: on kazan_center (LOD2, 2.2M triangles) and its LOD3 build (11.5M):
#:
#: * ``surfaces`` -- OSM land cover lying on the terrain: two thirds of an
#:   LOD2 city (1.45M triangles, 37 MB), 21 s to load instead of 3 s, and
#:   it tints ~6 % of the pixels.
#: * ``details`` -- LOD3 window frames, railings, canopies: 4.9M triangles
#:   (128 MB) of slivers a few centimetres wide.
#: * ``lod3_walls`` -- LOD3 walls with every opening cut in (4.3M
#:   triangles); hidden, each building is drawn as the plain shell under
#:   its roof (51k triangles for the city), the glass still in it.
#:
#: Full LOD3 is 313 MB of JSON and Streamlit's chart dies on it
#: (``RangeError: Invalid string length``); with these three off it loads
#: in ~4 s and orbits at ~45 fps. Trees (1 MB) and glass (8 MB) stay on.
DEFAULT_HIDDEN_LAYERS = ("surfaces", "details", "lod3_walls")

#: Layers so heavy that, switched on, they are drawn around the focus
#: (the devices) only, nearest first, up to DETAIL_BUDGET_TRIANGLES.
DETAIL_LAYERS = ("details", "lod3_walls")
#: ~1.5M triangles loads in ~10 s; 0.9M in 4-7 s, 2.2M in 21 s.
DETAIL_BUDGET_TRIANGLES = 1_500_000

#: Beyond this many numbers (3 per vertex, 3 per triangle) a figure is not
#: sent: Streamlit's chart serialises it to one string, ~20 characters a
#: number, against the browser's 2^29-character limit. 24.8M went through
#: (in 146 s), 27.9M ended in ``RangeError: Invalid string length``.
MAX_BACKDROP_VALUES = 20_000_000


def scene_layer_of(name: str) -> str:
    """Coarse category of one scene shape."""
    low = str(name).lower()
    if low.startswith(("building_", "roof_")):
        return "buildings"
    # LOD3 facades: the glass (`windows`, `windows_lod3`) apart from the
    # frames and the railings, canopies and steel doors (`facade_*`) --
    # those are the bulk of an LOD3 city and barely show.
    if low.startswith(("windows_frames", "facade")):
        return "details"
    if low.startswith("window"):
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


def _face_count(obj) -> int:
    try:
        return int(obj.mi_mesh.face_count())
    except Exception:
        return -1


def _roof_of(name: str) -> str | None:
    """The roof shape the OSM builder writes for a building's walls."""
    low = str(name)
    return "roof_" + low[len("building_"):] if low.startswith("building_") else None


def _is_lod3_wall(wall_faces: int, shell_faces: int) -> bool:
    """Walls with openings cut in, against the shell that would replace them.

    An LOD2 wall *is* that shell -- a quad per roof edge -- while an LOD3
    wall triangulated round every window is dozens of times heavier.
    """
    return shell_faces > 0 and wall_faces > 2 * shell_faces + 64


#: `scene_content_key` -> names of the LOD3 walls in that scene.
_LOD3_WALLS: OrderedDict = OrderedDict()


def _lod3_walls(scene) -> set:
    """Names of the building walls that :func:`_is_lod3_wall` picks out.

    Reads the roofs' buffers (a few hundred thousand triangles for a city)
    once per scene content: the layer filter asks on every rerun.
    """
    key = scene_content_key(scene)
    if key in _LOD3_WALLS:
        _LOD3_WALLS.move_to_end(key)
        return _LOD3_WALLS[key]
    objects = dict(scene.objects.items())
    out = set()
    for name, obj in objects.items():
        roof = _roof_of(name)
        if roof not in objects:
            continue
        try:
            mesh = objects[roof].mi_mesh
            faces = mesh.faces_buffer().numpy().reshape(-1, 3)
            vertices = mesh.vertex_positions_buffer().numpy().reshape(-1, 3)
        except Exception:
            continue
        if not len(faces):
            continue
        _x, _y, _z, i, j, k = _weld_mesh(vertices, faces)
        shell = 2 * len(_boundary_edges(np.stack([i, j, k], axis=1)))
        if _is_lod3_wall(_face_count(obj), shell):
            out.add(name)
    _LOD3_WALLS[key] = out
    while len(_LOD3_WALLS) > SCENE_MESH_CACHE_ENTRIES:
        _LOD3_WALLS.popitem(last=False)
    return out


def scene_layer_counts(scene) -> dict:
    """`{layer: triangles}` for the layers actually present in the scene.

    LOD3 walls count under ``lod3_walls``, not ``buildings``: hidden, they
    are replaced by shells of a few dozen triangles a building.
    """
    counts: dict[str, int] = {}
    detailed = _lod3_walls(scene)
    for name, obj in scene.objects.items():
        n = _face_count(obj)
        if n < 0:
            continue
        layer = "lod3_walls" if name in detailed else scene_layer_of(name)
        counts[layer] = counts.get(layer, 0) + n
    return counts


def _boundary_edges(faces):
    """Edges used by one triangle only, in the triangles' winding."""
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, first, count = np.unique(
        np.sort(directed, axis=1), axis=0, return_index=True, return_counts=True
    )
    return directed[first[count == 1]]


def roof_shell(roof_vertices, roof_faces, base_z: float):
    """LOD2 walls of a building: its roof's outline dropped to ``base_z``.

    Every outline edge (courtyards included -- they are holes in the roof)
    becomes one vertical quad from the eave down to the lowest point of the
    walls it stands in for. Display only: what the ray tracer sees is the
    scene, never this.
    """
    x, y, z, i, j, k = _weld_mesh(roof_vertices, roof_faces)
    edges = _boundary_edges(np.stack([i, j, k], axis=1))
    vertices = np.stack([x, y, z], axis=1)
    a, b = vertices[edges[:, 0]], vertices[edges[:, 1]]
    n = len(edges)
    floor = np.full((n, 1), float(base_z))
    quad = np.concatenate([a, b, np.hstack([b[:, :2], floor]), np.hstack([a[:, :2], floor])])
    idx = np.arange(n)
    faces = np.concatenate(
        [np.stack([idx, idx + n, idx + 2 * n], 1), np.stack([idx, idx + 2 * n, idx + 3 * n], 1)]
    )
    return quad, faces


def _object_material(obj):
    """`(material name, colour)` of one scene object."""
    try:
        name = str(getattr(obj.radio_material, "name", "") or "material")
    except Exception:
        name = "material"
    return name, get_object_color(obj)


def _build_scene_mesh_parts(scene, hidden_layers, focus=None, budget: int | None = DETAIL_BUDGET_TRIANGLES):
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
    objects = dict(scene.objects.items())
    detailed = _lod3_walls(scene)
    plain: list = []  # (material key, vertices, faces), always drawn
    walls: list = []  # (material key, full (v, f), shell (v, f), xy centre)
    details: list = []  # (material key, vertices, faces) of a visible detail layer
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    layer_tris: dict[str, dict[str, int]] = {}  # material -> layer -> triangles
    for name, obj in objects.items():
        layer = "lod3_walls" if name in detailed else scene_layer_of(name)
        if layer in hidden and layer != "lod3_walls":
            continue
        try:
            mesh = obj.mi_mesh
            vertices = mesh.vertex_positions_buffer().numpy().reshape(-1, 3)
            faces = mesh.faces_buffer().numpy().reshape(-1, 3)
        except Exception:
            continue
        if not len(faces):
            continue
        lo = np.minimum(lo, vertices[:, :2].min(axis=0))
        hi = np.maximum(hi, vertices[:, :2].max(axis=0))
        key = _object_material(obj)
        counts = layer_tris.setdefault(key[0], {})
        counts[layer] = counts.get(layer, 0) + len(faces)
        if layer == "lod3_walls":
            # The shell under the roof stands in for the walls, in the
            # walls' material, wherever the full walls are not drawn.
            try:
                roof = objects[_roof_of(name)].mi_mesh
                shell = roof_shell(
                    roof.vertex_positions_buffer().numpy().reshape(-1, 3),
                    roof.faces_buffer().numpy().reshape(-1, 3),
                    float(vertices[:, 2].min()),
                )
            except Exception:
                plain.append((key, vertices, faces))
                continue
            full = None if "lod3_walls" in hidden else (vertices, faces)
            walls.append((key, full, shell, vertices[:, :2].mean(axis=0)))
        elif layer in DETAIL_LAYERS:
            details.append((key, vertices, faces))
        else:
            plain.append((key, vertices, faces))

    # A heavy layer switched on is drawn nearest the focus first, as far as
    # DETAIL_BUDGET_TRIANGLES goes: all of an LOD3 city's frames are 4.9M
    # triangles, which no browser takes (see MAX_BACKDROP_VALUES).
    if focus is None and np.all(np.isfinite(lo)):
        focus = (lo + hi) / 2
    base = sum(len(f) for _k, _v, f in plain) + sum(len(shell[1]) for *_x, shell, _c in walls)
    dist, cost = [], []
    for _key, full, shell, centre in walls:
        if full is not None:
            dist.append(np.atleast_1d(np.hypot(*(centre - focus))))
            cost.append(np.atleast_1d(len(full[1]) - len(shell[1])))
    tri_dist = []
    for _key, vertices, faces in details:
        centroid = vertices[faces][:, :, :2].mean(axis=1)
        tri_dist.append(np.hypot(*(centroid - focus).T))
        dist.append(tri_dist[-1])
        cost.append(np.ones(len(faces), dtype=np.int64))
    radius = np.inf
    if dist:
        d = np.concatenate(dist)
        c = np.concatenate(cost)
        order = np.argsort(d, kind="stable")
        fits = np.cumsum(c[order]) <= (np.inf if budget is None else max(budget - base, 0))
        if not fits.all():
            radius = float(d[order][fits][-1]) if fits.any() else 0.0
    groups: dict = {}
    for key, vertices, faces in plain:
        groups.setdefault(key, []).append((vertices, faces))
    shells = 0
    for key, full, shell, centre in walls:
        near = full is not None and float(np.hypot(*(centre - focus))) <= radius
        if not near:
            shells += 1
        groups.setdefault(key, []).append(full if near else shell)
    for (key, vertices, faces), d in zip(details, tri_dist, strict=True):
        keep = faces[d <= radius]
        if len(keep):
            groups.setdefault(key, []).append((vertices, keep))

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
        "lod3_shells": shells,
        # Set when a visible heavy layer was cut to fit the budget: drawn
        # within this many metres of ``detail_focus`` only.
        "detail_radius_m": None if not np.isfinite(radius) else round(radius, 1),
        "detail_focus": None if focus is None else [round(float(v), 1) for v in focus],
        # The layer most of each material's triangles come from: lets a
        # renderer stack coplanar ground overlays (roads over water over
        # land cover over terrain).
        "material_layers": {m: max(c.items(), key=lambda kv: kv[1])[0] for m, c in layer_tris.items()},
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


def _focus_key(hidden_layers, focus, budget: int | None = DETAIL_BUDGET_TRIANGLES):
    """The focus as part of a cache key: only when it decides anything.

    Rounded to 25 m so a TX nudged by a metre does not re-weld the city.
    """
    if (
        budget is None
        or focus is None
        or all(layer in (hidden_layers or ()) for layer in DETAIL_LAYERS)
    ):
        return None
    return tuple(round(float(v) / 25.0) * 25.0 for v in np.asarray(focus, dtype=float)[:2])


def scene_mesh_parts(
    scene, hidden_layers=(), background=False, focus=None, budget: int | None = DETAIL_BUDGET_TRIANGLES
):
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
    key = (
        scene_content_key(scene),
        tuple(sorted(hidden_layers or ())),
        _focus_key(hidden_layers, focus, budget),
        budget,
    )
    store = _scene_mesh_store()
    with store["lock"]:
        if key in store["done"]:
            store["done"].move_to_end(key)
            return store["done"][key]
        pending = store["running"].get(key)
        if pending is None and background:
            pending = store["pool"].submit(
                _build_scene_mesh_parts, scene, key[1], key[2], key[3]
            )
            store["running"][key] = pending

    try:
        if pending is not None:
            if background and not pending.done():
                return None
            result = pending.result()
        else:
            result = _build_scene_mesh_parts(scene, key[1], key[2], key[3])
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
    focus=None,
    budget: int | None = DETAIL_BUDGET_TRIANGLES,
):
    """``Mesh3d`` traces for a scene backdrop, one per radio material.

    ``budget=None`` draws every visible layer in full and skips the size
    guard -- for the three.js renderer, which takes a whole LOD3 city.

    ``focus`` (x, y) is where a heavy layer that is switched on gets drawn
    first -- the devices, usually; the middle of the scene by default.

    Returns ``(traces, info)``, or ``None`` when ``background=True`` and
    the geometry is still being welded. The geometry is cached by
    :func:`scene_mesh_parts`; only the lightweight trace wrappers are
    rebuilt per call, which is why opacity and the legend flag are
    applied here and not baked into that cache.
    """
    built = scene_mesh_parts(scene, hidden_layers, background=background, focus=focus, budget=budget)
    if built is None:
        return None
    parts, info = built
    if budget is not None and 3 * (info.get("triangles", 0) + info.get("vertices", 0)) > MAX_BACKDROP_VALUES:
        # Sent anyway it never appears: the chart serialises the figure to
        # one string and the browser's string limit ends it there.
        return [], {**info, "too_heavy": True}
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
            meta={"layer": (info.get("material_layers") or {}).get(mat_name, "")},
        )
        for mat_name, color, x, y, z, i, j, k in parts
    ]
    return traces, info


def backdrop_note(info) -> str | None:
    """One line for a figure whose backdrop was cut or left out."""
    info = info or {}
    if info.get("too_heavy"):
        return (
            f"Scene left out: {info.get('triangles', 0):,} triangles is more than a "
            "browser can take. Hide layers (land cover, frames & railings, LOD3 walls)."
        )
    radius = info.get("detail_radius_m")
    if radius is None:
        return None
    shown = [layer for layer in DETAIL_LAYERS if layer not in (info.get("hidden_layers") or ())]
    what = " & ".join(LAYER_LABELS.get(layer, layer).split(" (")[0] for layer in shown)
    budget = f"{DETAIL_BUDGET_TRIANGLES / 1e6:g}M-triangle budget"
    if radius <= 0:
        return f"{what} left out: the other layers already fill the {budget} — hide land cover."
    return f"{what} drawn within {radius:.0f} m of the devices only ({budget})."


#: How the layer filter names its options.
LAYER_LABELS = {
    "lod3_walls": "LOD3 walls (else plain shells)",
    "details": "frames & railings",
    "windows": "windows (glass)",
    "surfaces": "land cover",
}


def scene_display_controls(
    scene,
    widget_key: str = "scene",
    container=None,
    default_hidden=DEFAULT_HIDDEN_LAYERS,
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
            format_func=lambda layer: f"{LAYER_LABELS.get(layer, layer)} ({counts.get(layer, 0):,})",
            help=(
                "Plotly re-sends every triangle on every rerun, so hiding "
                "the bulky layers is what makes a big scene usable. Off by "
                "default: land cover, LOD3 frames and railings, and LOD3 "
                "walls (drawn as plain shells under the roofs instead)."
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
