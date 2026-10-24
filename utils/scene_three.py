"""three.js renderer for the 3D scene figures -- the alternative to Plotly.

Plotly cannot draw a full LOD3 city: Streamlit's chart turns the figure into
one string (``RangeError: Invalid string length`` past ~25M numbers) and
plotly.js itself needs minutes for 5M triangles. This renders the *same*
figure the Plotly views build -- ``Mesh3d`` backdrop, ``Scatter3d`` devices
and rays, annotations -- with three.js on the GPU:

* every ``Mesh3d`` becomes a binary file under ``static/scene3d/`` (served by
  Streamlit's static serving, ``server.enableStaticServing``), fetched once
  by the browser and kept by URL, so a rerun re-sends only a manifest;
* markers and lines travel as small JSON in the component's ``data``;
* Plotly animation frames (``go.Frame(traces=...)``) become tracks of one
  binary file and play in the browser (play / pause, slider, speed) --
  no rerun per frame.

three.js itself is vendored in ``static/three/`` (no CDN).
"""

from __future__ import annotations

import base64
import contextlib
import contextvars
import gzip
import hashlib
import html
import json
import os
import re
import shutil
import tempfile
import time
from typing import Any

import numpy as np
import streamlit as st
from streamlit.components.v2 import component

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(_ROOT, "static")
SCENE3D_DIR = os.path.join(STATIC_DIR, "scene3d")
#: Streamlit refuses static files over 200 MB; a mesh bigger than this is
#: written in pieces.
MAX_BLOB_BYTES = 150 * 1024 * 1024
#: Mesh files kept on disk, oldest dropped first. Streamlit sizes the whole
#: static folder when it starts and above 1 GB turns static serving off --
#: three.js itself included -- so the cache stays well under that.
MAX_SCENE3D_BYTES = 640 * 1024**2
#: Files written this recently are never pruned: they are the pieces of the
#: figure being drawn, which alone can pass the cap (a long crowd walk).
PRUNE_GRACE_S = 15 * 60
#: Streamlit's own limit (streamlit/web/bootstrap.py).
STREAMLIT_STATIC_LIMIT = 1024**3

RENDERERS = ("Plotly", "three.js")
#: three.js by default: it takes any scene, full LOD3 included, at 60 fps.
DEFAULT_RENDERER = "three.js"


def renderer_control(key: str, container=None, help: str | None = None) -> str:
    """The Plotly / three.js choice, as a segmented control."""
    host = container if container is not None else st
    value = host.segmented_control(
        "3D renderer",
        RENDERERS,
        default=DEFAULT_RENDERER,
        key=key,
        help=help
        or (
            "Plotly: the usual chart, light scenes only (land cover, frames and "
            "LOD3 walls are cut to fit). three.js: draws on the GPU from binary "
            "meshes, so a full LOD3 city (11.5M triangles) is fine; animations play "
            "in the browser from one binary file."
        ),
    )
    return value or DEFAULT_RENDERER


def on_page_control(key: str, container=None, *, value: bool = True, on_change=None) -> bool:
    """Draw the 3D view on this page, or only offer it as a download.

    Off, no mesh crosses the websocket and the browser renders nothing: the
    layers, renderer and the rest stay settable and the download button
    builds the file on click -- set the view up on a weak machine, open the
    file on a strong one.
    """
    host = container if container is not None else st
    return bool(
        host.toggle(
            "Draw 3D on this page",
            value=value,
            key=key,
            on_change=on_change,
            help=(
                "Off: the 3D view is not rendered here (nothing heavy reaches this "
                "browser); its settings still apply and ⬇ Download builds the HTML "
                "file on click, to open on a stronger computer."
            ),
        )
    )


def three_ready() -> str | None:
    """Why three.js cannot run here (``None`` when it can)."""
    if not st.get_option("server.enableStaticServing"):
        if _static_bytes() > STREAMLIT_STATIC_LIMIT:
            _prune()
            return (
                "Streamlit turned static serving off at start-up: `static/` was over "
                "1 GB. The 3D mesh cache (`static/scene3d/`) has been trimmed; restart "
                "the app."
            )
        return (
            "three.js needs `server.enableStaticServing = true` in "
            "`.streamlit/config.toml` (restart the app after setting it)."
        )
    if not os.path.isfile(os.path.join(STATIC_DIR, "three", "three.module.js")):
        return "three.js is missing from `static/three/`."
    return None


# --------------------------------------------------------------------------
# Figure -> manifest
# --------------------------------------------------------------------------

_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")
_RGB = re.compile(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)")


def _rgb(color, default=(0.6, 0.6, 0.6)) -> tuple[float, float, float]:
    """A Plotly colour string as 0..1 RGB (named colours through matplotlib)."""
    text = str(color or "").strip()
    m = _HEX.match(text)
    if m:
        v = int(m.group(1), 16)
        return ((v >> 16) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255)
    m = _RGB.match(text)
    if m:
        return tuple(min(float(c), 255.0) / 255 for c in m.groups())  # type: ignore[return-value]
    try:
        from matplotlib.colors import to_rgb

        return to_rgb(text)
    except Exception:
        return default


def _fingerprint(*arrays) -> str:
    """Cheap identity of big arrays: shapes plus 4096 evenly spaced values.

    Hashing a 200 MB city on every rerun would cost half a second; a sample
    of each array tells two geometries apart just as well in practice.
    """
    h = hashlib.sha1()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.dtype, a.shape)).encode())
        flat = a.reshape(-1)
        if flat.size:
            take = np.linspace(0, flat.size - 1, min(flat.size, 4096)).astype(np.int64)
            h.update(flat[take].tobytes())
    return h.hexdigest()[:24]


#: Where the binaries go: static serving's folder, or a private one while a
#: download is being packed (`standalone_html`) -- a file of a long crowd
#: walk would otherwise push ``static/`` past Streamlit's 1 GB limit.
_BLOB_DIR: contextvars.ContextVar[str | None] = contextvars.ContextVar("sf3d_blob_dir", default=None)


def _blob_path(url: str) -> str:
    return os.path.join(_BLOB_DIR.get() or SCENE3D_DIR, url.removeprefix("scene3d/"))


@contextlib.contextmanager
def _private_blobs():
    """Binaries written inside go to a temporary folder, removed after."""
    from utils.storage import resolve_scratch_root, tmp_dir_for

    root = tmp_dir_for(resolve_scratch_root())
    os.makedirs(root, exist_ok=True)
    folder = tempfile.mkdtemp(prefix="scene3d_", dir=root)
    token = _BLOB_DIR.set(folder)
    try:
        yield folder
    finally:
        _BLOB_DIR.reset(token)
        shutil.rmtree(folder, ignore_errors=True)


def _write_blob(name: str, parts: list[np.ndarray]) -> None:
    folder = _BLOB_DIR.get() or SCENE3D_DIR
    path = os.path.join(folder, name)
    if os.path.exists(path):
        os.utime(path)  # recently used: pruned last
        return
    os.makedirs(folder, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        fh.writelines(np.ascontiguousarray(p).tobytes() for p in parts)
    os.replace(tmp, path)
    if folder == SCENE3D_DIR:
        _prune()


def _write_rows(prefix: str, rows: np.ndarray, *identity, dtype: type[np.generic] = np.float32) -> list[str]:
    """Rows (float32 unless ``dtype``) as one or more files, each under ``MAX_BLOB_BYTES``.

    Streamlit serves nothing over 200 MB: 7 200 walking people over 1 500
    frames were 302 MB in one file and came back 404. Split on row
    boundaries; the viewer fetches the parts and joins them in order.
    """
    rows = np.ascontiguousarray(rows, dtype=dtype)
    width = int(np.prod(rows.shape[1:])) if rows.ndim > 1 else 1
    per_file = max(1, MAX_BLOB_BYTES // max(rows.itemsize * width, 1))
    key = _fingerprint(rows, *identity)
    urls = []
    for part, lo in enumerate(range(0, max(len(rows), 1), per_file)):
        name = f"{prefix}_{key}_{part}.bin"
        _write_blob(name, [rows[lo : lo + per_file]])
        urls.append(f"scene3d/{name}")
    return urls


def _prune() -> None:
    try:
        files = [os.path.join(SCENE3D_DIR, f) for f in os.listdir(SCENE3D_DIR) if f.endswith(".bin")]
        files.sort(key=os.path.getmtime, reverse=True)
        recent = time.time() - PRUNE_GRACE_S
        total = 0
        for f in files:
            total += os.path.getsize(f)
            if total > MAX_SCENE3D_BYTES and os.path.getmtime(f) < recent:
                os.remove(f)
    except OSError:
        pass


def _static_bytes() -> int:
    total = 0
    for root, _dirs, names in os.walk(STATIC_DIR):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _mesh_pieces(positions, faces, colors):
    """Split a mesh whose file would exceed MAX_BLOB_BYTES (rarely needed)."""
    size = positions.nbytes + faces.nbytes + (0 if colors is None else colors.nbytes)
    if size <= MAX_BLOB_BYTES or len(faces) < 2:
        return [(positions, faces, colors)]
    half = len(faces) // 2
    out = []
    for chunk in (faces[:half], faces[half:]):
        used, inverse = np.unique(chunk.reshape(-1), return_inverse=True)
        sub = (positions[used], inverse.reshape(-1, 3).astype(np.uint32), None if colors is None else colors[used])
        out += _mesh_pieces(*sub)
    return out


def _ground_priority(trace) -> int:
    """Draw order of a coplanar ground overlay (0 = not one).

    Scenes built before the overlays were lifted apart have land cover,
    water and squares in one plane over the terrain; the viewer pulls the
    upper ones forward in depth so the picture does not flicker.
    """
    layer = str((trace.meta or {}).get("layer", "") if isinstance(trace.meta, dict) else "")
    name = str(trace.name or "").lower()
    if layer == "terrain":
        return 1
    if layer == "surfaces":
        return 3 if "water" in name else 2
    if layer == "roads":
        return 4
    return 0


def _mesh_entries(trace) -> list[dict]:
    x = np.asarray(trace.x, dtype=np.float32)
    if x.size == 0 or trace.i is None:
        return []
    positions = np.stack([x, np.asarray(trace.y, dtype=np.float32), np.asarray(trace.z, dtype=np.float32)], 1)
    faces = np.stack([np.asarray(trace[c], dtype=np.uint32) for c in "ijk"], 1)
    colors = None
    if trace.facecolor is not None and len(trace.facecolor):
        # Per-face colours as vertex colours: the meshes that carry them
        # (traffic: cars and people) share no vertices between colours.
        palette: dict[str, tuple] = {}
        face_rgb = np.array([palette.setdefault(c, _rgb(c)) for c in trace.facecolor], dtype=np.float32)
        colors = np.zeros((len(positions), 3), dtype=np.uint8)
        colors[faces.reshape(-1)] = np.repeat(np.round(face_rgb * 255), 3, axis=0).astype(np.uint8)
    out = []
    for pos, fac, col in _mesh_pieces(positions, faces, colors):
        name = f"{_fingerprint(pos, fac, *(() if col is None else (col,)))}.bin"
        _write_blob(name, [pos, fac] + ([] if col is None else [col]))
        out.append(
            {
                "url": f"scene3d/{name}",
                "nv": len(pos),
                "nt": len(fac),
                "colors": col is not None,
                "color": list(_rgb(trace.color)),
                "opacity": float(1.0 if trace.opacity is None else trace.opacity),
                "name": str(trace.name or "mesh"),
                "priority": _ground_priority(trace),
            }
        )
    return out


def _coords(trace) -> np.ndarray:
    """``(n, 3)`` float32 points of a trace, NaN rows for ``None`` breaks.

    Plotly keeps numpy input as numpy and lists as tuples, so neither
    ``trace.x or ()`` nor a plain truth test works on both.
    """
    if trace is None:
        return np.zeros((0, 3), np.float32)
    cols = [np.asarray(() if getattr(trace, c) is None else getattr(trace, c), np.float32).reshape(-1) for c in "xyz"]
    n = min(len(c) for c in cols)
    return np.stack([c[:n] for c in cols], 1)


def _xyz(trace) -> list:
    """Interleaved x, y, z with ``None`` breaks kept (Plotly's line gaps)."""
    out: list = []
    for row in _coords(trace):
        if not np.isfinite(row).all():
            out += [None, None, None]
        else:
            out += [round(float(v), 2) for v in row]
    return out


def _scatter_entries(trace) -> list[dict]:
    mode = str(trace.mode or "markers")
    xyz = _xyz(trace)
    if not xyz:
        return []
    out = []
    opacity = float(1.0 if trace.opacity is None else trace.opacity)
    name = str(trace.name or "")
    if "lines" in mode:
        color = trace.line.color if trace.line is not None else None
        out.append({"kind": "line", "xyz": xyz, "color": list(_rgb(color, (0.3, 0.3, 0.9))),
                    "opacity": opacity, "name": name})
    if "markers" in mode or "text" in mode:
        marker = trace.marker
        color = marker.color if marker is not None else None
        colors = None
        if isinstance(color, (list, tuple, np.ndarray)) and len(color) and isinstance(color[0], str):
            colors = [list(_rgb(c)) for c in color]
            color = None
        size = marker.size if marker is not None and isinstance(marker.size, (int, float)) else 6
        text = list(trace.text) if isinstance(trace.text, (list, tuple)) else None
        out.append({"kind": "points", "xyz": xyz, "color": list(_rgb(color, (0.9, 0.2, 0.2))),
                    "colors": colors, "size": float(size), "opacity": opacity, "name": name,
                    "text": text if text and len(text) <= 400 else None})
    return out


def _instance_track(base, ti: int, per_frame: list[dict]) -> dict[str, Any] | None:
    """A moving trace that asks for 3D bodies: one template, a pose per body.

    The trace carries ``meta.instances = {"vertices", "faces"}`` (the body
    in its own frame, x forward) and, in every frame, ``customdata`` rows of
    ``z_ground, heading, scale_x, scale_y, scale_z`` next to its markers.
    Written as float32 rows ``x, y, z_ground, heading, sx, sy, sz``, one
    file per track; the viewer instances the template with them.
    """
    meta = base.meta if isinstance(base.meta, dict) else {}
    template = meta.get("instances")
    if not isinstance(template, dict):
        return None
    rows, offsets, start = [], [], 0
    for by_trace in per_frame:
        tr = by_trace.get(ti)
        xyz = _coords(tr)
        pose = np.asarray(() if tr is None or tr.customdata is None else tr.customdata, np.float32).reshape(-1, 5)
        n = min(len(xyz), len(pose))
        offsets.append(start)
        rows.append(np.column_stack([xyz[:n, :2], pose[:n]]).astype(np.float32))
        start += n
    offsets.append(start)
    blob = np.concatenate(rows) if rows else np.zeros((0, 7), np.float32)
    urls = _write_rows("inst", blob, np.asarray(offsets))
    marker = base.marker
    return {
        "kind": "instances",
        "name": str(base.name or f"trace {ti}"),
        "color": list(_rgb(marker.color if marker is not None and isinstance(marker.color, str) else None)),
        "opacity": float(1.0 if base.opacity is None else base.opacity),
        "urls": urls,
        "count": len(blob),
        "offsets": offsets,
        "vertices": [float(v) for v in template.get("vertices", [])],
        "faces": [int(v) for v in template.get("faces", [])],
    }


def _body_tracks(fig, n_frames: int) -> list[dict[str, Any]]:
    """Compact moving bodies a figure carries beside its frames (``fig._sf3d_bodies``).

    Each entry: ``name``, ``color``, ``template`` (``vertices``, ``faces``)
    and ``bodies`` -- an object with ``q`` (frames, n, 4) int16 = x, y,
    z_ground, heading quantised (x == -32768: not on the map), ``origin``,
    ``unit_xy``, ``unit_z`` and ``scale`` (n, 3), like
    ``pages.position_selector.playback.Bodies``. Written as frame-to-frame
    differences (int16, wrapping): bodies move centimetres per frame, so a
    download's gzip makes a 7k crowd over 12000 frames ~40 MB instead of
    the 2.6 GB of float32 pose rows. The viewer adds them back up once.
    """
    out = []
    for entry in getattr(fig, "_sf3d_bodies", None) or ():
        b = entry["bodies"]
        q = np.ascontiguousarray(b.q, dtype=np.int16)
        if q.shape[0] != n_frames:
            raise ValueError(f"{entry['name']}: {q.shape[0]} frames of bodies for {n_frames} frames")
        delta = q.copy()
        np.subtract(q[1:], q[:-1], out=delta[1:])  # int16: wraps, and so does the sum back
        urls = _write_rows("bodies", delta.reshape(len(q), -1), np.asarray(b.origin), dtype=np.int16)
        template = entry["template"]
        out.append(
            {
                "kind": "bodies",
                "name": str(entry["name"]),
                "color": list(_rgb(entry.get("color"))),
                "opacity": float(entry.get("opacity", 1.0)),
                "urls": urls,
                "n": int(q.shape[1]),
                "origin": [float(v) for v in b.origin],
                "unit": [float(b.unit_xy), float(b.unit_z)],
                "scale": np.round(np.asarray(b.scale, np.float64), 4).reshape(-1).tolist(),
                "vertices": [float(v) for v in template.get("vertices", [])],
                "faces": [int(v) for v in template.get("faces", [])],
            }
        )
    return out


def _frame_labels(fig) -> list[str]:
    sliders = fig.layout.sliders or ()
    steps = sliders[0].steps if sliders else ()
    if steps and len(steps) == len(fig.frames):
        return [str(s.label) for s in steps]
    return [str(f.name) for f in fig.frames]


def _animation(fig) -> tuple[dict[str, Any] | None, set[int]]:
    """Plotly frames as tracks of one binary file, and the traces they move.

    Every frame names the traces it replaces (``go.Frame(traces=...)``);
    each of those becomes a *track*: per frame, its points (lines keep
    their ``None`` breaks as NaN rows). All tracks and frames are written
    back to back as float32 xyz, with per-track offsets into that run, so
    a thousand cars over a few hundred frames stay a few MB fetched once.
    """
    frames: list[Any] = list(fig.frames or ())
    if not frames:
        return None, set()
    animated = sorted({int(t) for f in frames for t in (f.traces or ())})
    if not animated:
        return None, set()
    per_frame = []
    for f in frames:
        by_trace = dict(zip((int(t) for t in f.traces or ()), f.data or (), strict=False))
        per_frame.append(by_trace)
    tracks, parts, start = [], [], 0
    for ti in animated:
        base = fig.data[ti]
        instances = _instance_track(base, ti, per_frame)
        if instances is not None:
            tracks.append(instances)
            continue
        mode = str(base.mode or "markers")
        offsets, colors = [], []
        for by_trace in per_frame:
            tr = by_trace.get(ti)
            xyz = _coords(tr)
            offsets.append(start)
            parts.append(xyz)
            start += len(xyz)
            line = getattr(tr, "line", None) if tr is not None else None
            colors.append(list(_rgb(line.color)) if line is not None and line.color else None)
        offsets.append(start)
        marker = base.marker
        color = (base.line.color if "lines" in mode and base.line is not None else None) or (
            marker.color if marker is not None and isinstance(marker.color, str) else None
        )
        tracks.append(
            {
                "kind": "line" if "lines" in mode else "points",
                "name": str(base.name or f"trace {ti}"),
                "color": list(_rgb(color, (0.9, 0.2, 0.2))),
                "frame_colors": colors if any(colors) else None,
                "size": float(marker.size) if marker is not None and isinstance(marker.size, (int, float)) else 6.0,
                "opacity": float(1.0 if base.opacity is None else base.opacity),
                "offsets": offsets,
            }
        )
    blob = np.concatenate(parts) if parts else np.zeros((0, 3), np.float32)
    urls = _write_rows("anim", blob, np.asarray([t["offsets"] for t in tracks]))
    meta = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
    tracks += _body_tracks(fig, len(frames))
    return {
        "urls": urls,
        "points": len(blob),
        "labels": _frame_labels(fig),
        "frame_ms": float(meta.get("frame_ms", 100.0)),
        "tracks": tracks,
    }, set(animated)


def figure_manifest(fig) -> dict[str, Any]:
    """What the three.js component needs from a Plotly 3D figure."""
    meshes, items = [], []
    animation, animated = _animation(fig)
    for index, trace in enumerate(fig.data):
        if trace.type == "mesh3d":
            meshes += _mesh_entries(trace)
        elif trace.type == "scatter3d" and trace.visible not in (False, "legendonly") and index not in animated:
            items += _scatter_entries(trace)
    notes = [str(a.text) for a in (fig.layout.annotations or ()) if a.text]
    base = "/" + str(st.get_option("server.baseUrlPath") or "").strip("/")
    return {
        "base": (base.rstrip("/") + "/app/static/"),
        "meshes": meshes,
        "items": items,
        "animation": animation,
        "notes": notes,
    }


# --------------------------------------------------------------------------
# The component
# --------------------------------------------------------------------------

_HTML = """
<div class="sf3d">
  <canvas class="sf3d-canvas"></canvas>
  <div class="sf3d-bar">
    <button class="sf3d-fs" type="button" title="Fullscreen">⛶</button>
    <button class="sf3d-leg" type="button" title="Legend">☰</button>
    <label><input class="sf3d-opaque" type="checkbox"> opaque</label>
    <span class="sf3d-status"></span>
  </div>
  <div class="sf3d-notes"></div>
  <div class="sf3d-legend"></div>
  <div class="sf3d-play" style="display: none">
    <button class="sf3d-toggle" type="button" title="Play / pause">▶</button>
    <input class="sf3d-slider" type="range" min="0" max="0" step="1" value="0">
    <span class="sf3d-time"></span>
    <select class="sf3d-speed" title="Playback speed">
      <option value="0.25">×0.25</option><option value="0.5">×0.5</option>
      <option value="1" selected>×1</option><option value="2">×2</option><option value="4">×4</option>
    </select>
  </div>
</div>
"""

_CSS = """
.sf3d { position: relative; width: 100%; height: 100%; min-height: 300px;
        background: var(--st-background-color); border-radius: 0.5rem; overflow: hidden; }
.sf3d-canvas { width: 100%; height: 100%; display: block; }
.sf3d-bar { position: absolute; top: 8px; left: 8px; display: flex; gap: 10px; align-items: center;
            font: 12px var(--st-font, sans-serif); color: var(--st-text-color); }
.sf3d-bar button { cursor: pointer; border: 1px solid var(--st-border-color, #888); border-radius: 4px;
                   background: var(--st-secondary-background-color); color: var(--st-text-color); }
.sf3d-notes { position: absolute; top: 8px; left: 50%; transform: translateX(-50%);
              font: 12px var(--st-font, sans-serif); color: #b8860b; text-align: center; }
.sf3d-legend { position: absolute; top: 34px; left: 8px; max-height: 70%; overflow: auto;
               font: 12px var(--st-font, sans-serif); color: var(--st-text-color);
               background: color-mix(in srgb, var(--st-secondary-background-color) 85%, transparent);
               border-radius: 4px; padding: 4px 6px; }
.sf3d-legend label { display: block; white-space: nowrap; cursor: pointer; }
.sf3d-legend i { display: inline-block; width: 10px; height: 10px; margin: 0 4px; border-radius: 2px; }
.sf3d-play { position: absolute; left: 8px; right: 8px; bottom: 8px; display: flex; gap: 8px; align-items: center;
             font: 12px var(--st-font, sans-serif); color: var(--st-text-color); padding: 4px 8px; border-radius: 4px;
             background: color-mix(in srgb, var(--st-secondary-background-color) 85%, transparent); }
.sf3d-play button, .sf3d-play select { cursor: pointer; border: 1px solid var(--st-border-color, #888);
             border-radius: 4px; background: var(--st-secondary-background-color); color: var(--st-text-color); }
.sf3d-slider { flex: 1; }
.sf3d-time { min-width: 9em; text-align: right; font-variant-numeric: tabular-nums; }
"""

_JS = """
const geometryCache = new Map();   // url -> Promise<THREE.BufferGeometry>
const animCache = new Map();       // url -> Promise<Float32Array>
const viewers = new WeakMap();     // parentElement -> viewer
let threePromise = null;

// A page saved by ``standalone_html`` brings three.js and the binaries
// along (no server to fetch them from) and sets this before mounting.
const standalone = () => globalThis.sf3dStandalone;

function getBlob(base, url) {
  const own = standalone();
  return own ? own.fetch(url) : fetch(new URL(base + url, window.location.origin).href);
}

function loadThree(base) {
  if (!threePromise && standalone()) threePromise = standalone().lib;
  if (!threePromise) {
    const abs = (p) => new URL(base + p, window.location.origin).href;
    threePromise = Promise.all([import(abs("three/three.module.js")), import(abs("three/OrbitControls.js"))])
      .then(([THREE, controls]) => ({ THREE, OrbitControls: controls.OrbitControls }));
  }
  return threePromise;
}

function geometry(THREE, base, m) {
  if (!geometryCache.has(m.url)) {
    geometryCache.set(m.url, getBlob(base, m.url).then((r) => {
      if (!r.ok) throw new Error(`${r.status} ${m.url}`);
      return r.arrayBuffer();
    }).then((buf) => {
      const g = new THREE.BufferGeometry();
      const pos = new Float32Array(buf, 0, m.nv * 3);
      const idx = new Uint32Array(buf, m.nv * 12, m.nt * 3);
      g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      g.setIndex(new THREE.BufferAttribute(idx, 1));
      if (m.colors) {
        const col = new Uint8Array(buf, m.nv * 12 + m.nt * 12, m.nv * 3);
        g.setAttribute("color", new THREE.BufferAttribute(col, 3, true));
      }
      g.computeBoundingBox();
      g.computeBoundingSphere();
      return g;
    }));
  }
  return geometryCache.get(m.url);
}

function makeViewer(root, lib) {
  const { THREE, OrbitControls } = lib;
  const canvas = root.querySelector(".sf3d-canvas");
  // Transparent: the app's own background (light or dark theme) shows.
  // A plain depth buffer (polygonOffset needs one) whose near plane
  // follows the camera's distance -- see the render loop.
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const scene = new THREE.Scene();
  scene.background = null;
  const camera = new THREE.PerspectiveCamera(45, 1, 0.5, 200000);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.screenSpacePanning = true;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x666666, 1.4));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(0.4, -0.6, 1.0);
  scene.add(sun);
  const groups = { meshes: new THREE.Group(), items: new THREE.Group(), anim: new THREE.Group() };
  scene.add(groups.meshes, groups.items, groups.anim);
  const v = { lib, root, renderer, scene, camera, controls, groups, fitted: false, alive: true,
              opaque: false, hidden: new Set(), meshKey: "", itemKey: "", animKey: "",
              anim: null, frame: 0, playing: false, lastTick: 0 };
  const ui = {
    box: root.querySelector(".sf3d-play"), toggle: root.querySelector(".sf3d-toggle"),
    slider: root.querySelector(".sf3d-slider"), time: root.querySelector(".sf3d-time"),
    speed: root.querySelector(".sf3d-speed"),
  };
  v.ui = ui;
  ui.toggle.onclick = () => {
    v.playing = !v.playing;
    ui.toggle.textContent = v.playing ? "⏸" : "▶";
    v.lastTick = performance.now();
  };
  ui.slider.oninput = () => setFrame(v, Number(ui.slider.value));
  const resize = () => {
    const w = root.clientWidth || 800, h = root.clientHeight || 600;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  v.observer = new ResizeObserver(resize);
  v.observer.observe(root);
  resize();
  const loop = () => {
    if (!v.alive) return;
    if (v.playing && v.anim) {
      const now = performance.now();
      const step = v.anim.frame_ms / Number(ui.speed.value || 1);
      const n = Math.floor((now - v.lastTick) / step);
      if (n > 0) {
        v.lastTick += n * step;
        setFrame(v, (v.frame + n) % v.anim.labels.length);
      }
    }
    controls.update();
    const dist = camera.position.distanceTo(controls.target);
    const near = Math.max(0.05, dist / 1500);
    if (Math.abs(near - camera.near) > 0.2 * camera.near) {
      camera.near = near;
      camera.updateProjectionMatrix();
    }
    renderer.render(scene, camera);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
  root.querySelector(".sf3d-fs").onclick = () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else root.requestFullscreen?.();
  };
  root.querySelector(".sf3d-leg").onclick = () => {
    const box = root.querySelector(".sf3d-legend");
    box.style.display = box.style.display === "none" ? "" : "none";
  };
  root.querySelector(".sf3d-opaque").onchange = (e) => { v.opaque = e.target.checked; applyMaterials(v); };
  return v;
}

function applyMaterials(v) {
  v.groups.meshes.children.forEach((mesh) => {
    const o = mesh.userData.opacity;
    const see = !v.opaque && o < 0.999;
    mesh.material.transparent = see;
    mesh.material.opacity = see ? o : 1;
    mesh.material.depthWrite = !see;
    mesh.material.needsUpdate = true;
    mesh.visible = !v.hidden.has(mesh.userData.name);
  });
  v.groups.items.children.forEach((obj) => { obj.visible = !v.hidden.has(obj.userData.name); });
  v.groups.anim.children.forEach((obj) => { obj.visible = !v.hidden.has(obj.userData.name); });
}

function fit(v) {
  const box = new v.lib.THREE.Box3().setFromObject(v.groups.meshes);
  if (box.isEmpty()) box.setFromObject(v.groups.items);
  if (box.isEmpty()) return;
  const c = box.getCenter(new v.lib.THREE.Vector3());
  const size = box.getSize(new v.lib.THREE.Vector3());
  const r = Math.max(size.x, size.y, size.z * 3, 10);
  v.controls.target.copy(c);
  v.camera.position.set(c.x + 0.7 * r, c.y - 0.7 * r, c.z + 0.55 * r);
  v.camera.near = Math.max(r / 5000, 0.05);
  v.camera.far = r * 20;
  v.camera.updateProjectionMatrix();
  v.fitted = true;
}

function legend(v, data) {
  const box = v.root.querySelector(".sf3d-legend");
  const seen = new Map();
  for (const m of data.meshes) if (!seen.has(m.name)) seen.set(m.name, m.color);
  for (const it of data.items) if (it.name && !seen.has(it.name)) seen.set(it.name, it.color);
  for (const t of (data.animation ? data.animation.tracks : [])) if (!seen.has(t.name)) seen.set(t.name, t.color);
  box.replaceChildren();
  for (const [name, c] of seen) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = !v.hidden.has(name);
    input.onchange = () => { input.checked ? v.hidden.delete(name) : v.hidden.add(name); applyMaterials(v); };
    const swatch = document.createElement("i");
    swatch.style.background = `rgb(${c.map((x) => Math.round(x * 255)).join(",")})`;
    label.append(input, swatch, document.createTextNode(name));
    box.append(label);
  }
}

async function updateMeshes(v, data) {
  const { THREE } = v.lib;
  const key = data.meshes.map((m) => m.url + m.opacity + m.color).join("|");
  if (key === v.meshKey) return;
  v.meshKey = key;
  const status = v.root.querySelector(".sf3d-status");
  const tris = data.meshes.reduce((s, m) => s + m.nt, 0);
  status.textContent = `loading ${(tris / 1e6).toFixed(2)}M triangles…`;
  const t0 = performance.now();
  const geoms = await Promise.all(data.meshes.map((m) => geometry(THREE, data.base, m)));
  if (key !== v.meshKey) return;  // a newer figure arrived meanwhile
  v.groups.meshes.children.slice().forEach((mesh) => { mesh.material.dispose(); v.groups.meshes.remove(mesh); });
  data.meshes.forEach((m, n) => {
    const material = new THREE.MeshPhongMaterial({
      color: m.colors ? 0xffffff : new THREE.Color(...m.color), vertexColors: m.colors,
      flatShading: true, side: THREE.DoubleSide, shininess: 5,
      // Ground overlays drawn in their stacking order where they share a plane.
      polygonOffset: m.priority > 0, polygonOffsetFactor: -2 * m.priority, polygonOffsetUnits: -4 * m.priority,
    });
    const mesh = new THREE.Mesh(geoms[n], material);
    mesh.userData = { name: m.name, opacity: m.opacity };
    v.groups.meshes.add(mesh);
  });
  applyMaterials(v);
  if (!v.fitted) fit(v);
  status.textContent = `${(tris / 1e6).toFixed(2)}M triangles · ${((performance.now() - t0) / 1000).toFixed(1)} s`;
}

function updateItems(v, data) {
  const { THREE } = v.lib;
  const key = JSON.stringify(data.items);
  if (key === v.itemKey) return;
  v.itemKey = key;
  v.groups.items.children.slice().forEach((o) => { o.geometry.dispose(); o.material.dispose(); v.groups.items.remove(o); });
  for (const it of data.items) {
    const pts = [];
    const segs = [];
    for (let n = 0; n + 2 < it.xyz.length; n += 3) {
      const p = it.xyz[n] === null ? null : [it.xyz[n], it.xyz[n + 1], it.xyz[n + 2]];
      pts.push(p);
    }
    if (it.kind === "line") {
      for (let n = 1; n < pts.length; n++) if (pts[n - 1] && pts[n]) segs.push(...pts[n - 1], ...pts[n]);
      if (!segs.length) continue;
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(segs, 3));
      const obj = new THREE.LineSegments(g, new THREE.LineBasicMaterial({
        color: new THREE.Color(...it.color), transparent: it.opacity < 1, opacity: it.opacity }));
      obj.userData = { name: it.name };
      v.groups.items.add(obj);
    } else {
      const flat = [], cols = [];
      pts.forEach((p, n) => { if (p) { flat.push(...p); if (it.colors) cols.push(...(it.colors[n] || it.color)); } });
      if (!flat.length) continue;
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(flat, 3));
      if (it.colors) g.setAttribute("color", new THREE.Float32BufferAttribute(cols, 3));
      const obj = new THREE.Points(g, new THREE.PointsMaterial({
        color: it.colors ? 0xffffff : new THREE.Color(...it.color), vertexColors: !!it.colors,
        size: Math.max(it.size, 4), sizeAttenuation: false, transparent: it.opacity < 1, opacity: it.opacity }));
      obj.userData = { name: it.name };
      v.groups.items.add(obj);
    }
  }
  applyMaterials(v);
  if (!v.fitted && !data.meshes.length) fit(v);
}

function animBlob(base, urls, floats, Type = Float32Array, then = null) {
  // One array from the parts a big track was split into (static files are
  // capped at 200 MB), in order; ``then`` runs once on it (cached after).
  const key = urls.join("|");
  if (!animCache.has(key)) {
    animCache.set(key, Promise.all(urls.map((url) => {
      return getBlob(base, url).then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${url}`);
        return r.arrayBuffer();
      });
    })).then((bufs) => {
      let out;
      if (bufs.length === 1) out = new Type(bufs[0], 0, floats);
      else {
        out = new Type(floats);
        let at = 0;
        for (const buf of bufs) {
          const part = new Type(buf);
          out.set(part.subarray(0, Math.min(part.length, floats - at)), at);
          at += part.length;
        }
      }
      return then ? then(out) : out;
    }));
  }
  return animCache.get(key);
}

function bodyRows(base, t, frames) {
  // Frame-to-frame differences, int16, added back up in place (the int16
  // array wraps exactly as the writer's subtraction did).
  const stride = 4 * t.n;
  return animBlob(base, t.urls, frames * stride, Int16Array, (q) => {
    for (let i = stride; i < q.length; i++) q[i] += q[i - stride];
    return q;
  });
}

function setBodies(v, t, obj, f) {
  const { THREE } = v.lib;
  if (!_m.mat) {
    _m.mat = new THREE.Matrix4(); _m.pos = new THREE.Vector3(); _m.quat = new THREE.Quaternion();
    _m.scale = new THREE.Vector3(); _m.axis = new THREE.Vector3(0, 0, 1);
  }
  const q = t.rows, sc = t.scale, [ox, oy, oz] = t.origin, [uxy, uz] = t.unit, k = Math.PI / 32767;
  const base = 4 * t.n * f;
  let n = 0;
  for (let b = 0; b < t.n; b++) {
    const r = base + 4 * b;
    if (q[r] === -32768) continue;
    _m.pos.set(ox + q[r] * uxy, oy + q[r + 1] * uxy, oz + q[r + 2] * uz);
    _m.quat.setFromAxisAngle(_m.axis, q[r + 3] * k);
    _m.scale.set(sc[3 * b], sc[3 * b + 1], sc[3 * b + 2]);
    _m.mat.compose(_m.pos, _m.quat, _m.scale);
    obj.setMatrixAt(n++, _m.mat);
  }
  obj.count = n;
  obj.instanceMatrix.needsUpdate = true;
}

const _m = { mat: null, pos: null, quat: null, scale: null, axis: null };

function setInstances(v, t, obj, f) {
  const { THREE } = v.lib;
  if (!_m.mat) {
    _m.mat = new THREE.Matrix4(); _m.pos = new THREE.Vector3(); _m.quat = new THREE.Quaternion();
    _m.scale = new THREE.Vector3(); _m.axis = new THREE.Vector3(0, 0, 1);
  }
  const rows = t.rows;
  const i0 = t.offsets[f], i1 = t.offsets[f + 1];
  let n = 0;
  for (let i = i0; i < i1; i++) {
    const r = 7 * i;
    if (!Number.isFinite(rows[r]) || !Number.isFinite(rows[r + 2])) continue;
    _m.pos.set(rows[r], rows[r + 1], rows[r + 2]);
    _m.quat.setFromAxisAngle(_m.axis, rows[r + 3]);
    _m.scale.set(rows[r + 4], rows[r + 5], rows[r + 6]);
    _m.mat.compose(_m.pos, _m.quat, _m.scale);
    obj.setMatrixAt(n++, _m.mat);
  }
  obj.count = n;
  obj.instanceMatrix.needsUpdate = true;
}

function setFrame(v, f) {
  const a = v.anim;
  if (!a) return;
  f = Math.max(0, Math.min(f, a.labels.length - 1));
  v.frame = f;
  v.ui.slider.value = String(f);
  v.ui.time.textContent = a.labels[f] || "";
  a.tracks.forEach((t, k) => {
    const obj = a.objs[k];
    if (t.kind === "instances") { setInstances(v, t, obj, f); return; }
    if (t.kind === "bodies") { setBodies(v, t, obj, f); return; }
    const attr = obj.geometry.getAttribute("position");
    const out = attr.array;
    const p0 = t.offsets[f], p1 = t.offsets[f + 1];
    let n = 0;
    const ok = (p) => Number.isFinite(a.arr[3 * p]) && Number.isFinite(a.arr[3 * p + 1]) && Number.isFinite(a.arr[3 * p + 2]);
    const put = (p) => { out[3 * n] = a.arr[3 * p]; out[3 * n + 1] = a.arr[3 * p + 1]; out[3 * n + 2] = a.arr[3 * p + 2]; n++; };
    if (t.kind === "line") {
      for (let p = p0 + 1; p < p1; p++) if (ok(p - 1) && ok(p)) { put(p - 1); put(p); }
    } else {
      for (let p = p0; p < p1; p++) if (ok(p)) put(p);
    }
    attr.needsUpdate = true;
    obj.geometry.setDrawRange(0, n);
    const c = t.frame_colors && t.frame_colors[f];
    if (c) obj.material.color.setRGB(c[0], c[1], c[2]);
  });
}

async function updateAnimation(v, data) {
  const { THREE } = v.lib;
  const a = data.animation;
  const key = a ? a.urls.join("|") + "#" + a.tracks.map((t) => (t.urls || []).join("|")).join(";") : "";
  if (key === v.animKey) return;
  v.animKey = key;
  v.groups.anim.children.slice().forEach((o) => { o.geometry.dispose(); o.material.dispose(); v.groups.anim.remove(o); });
  v.anim = null;
  v.ui.box.style.display = a ? "" : "none";
  if (!a) return;
  const arr = await animBlob(data.base, a.urls, a.points * 3);
  const rowsets = await Promise.all(a.tracks.map((t) => t.kind === "instances" ? animBlob(data.base, t.urls, t.count * 7)
    : t.kind === "bodies" ? bodyRows(data.base, t, a.labels.length) : null));
  if (key !== v.animKey) return;
  a.tracks.forEach((t, k) => { if (rowsets[k]) t.rows = rowsets[k]; });
  const objs = a.tracks.map((t) => {
    if (t.kind === "instances" || t.kind === "bodies") {
      let most = 1;
      if (t.kind === "bodies") most = Math.max(1, t.n);
      else for (let f = 0; f + 1 < t.offsets.length; f++) most = Math.max(most, t.offsets[f + 1] - t.offsets[f]);
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(t.vertices, 3));
      g.setIndex(t.faces);
      g.computeVertexNormals();
      const mesh = new THREE.InstancedMesh(g, new THREE.MeshPhongMaterial({
        color: new THREE.Color(...t.color), flatShading: true, side: THREE.DoubleSide, shininess: 30 }), most);
      mesh.count = 0;
      mesh.frustumCulled = false;
      mesh.userData = { name: t.name };
      v.groups.anim.add(mesh);
      return mesh;
    }
    let most = 0;
    for (let f = 0; f + 1 < t.offsets.length; f++) most = Math.max(most, t.offsets[f + 1] - t.offsets[f]);
    const size = Math.max(1, t.kind === "line" ? 2 * most : most);
    const g = new THREE.BufferGeometry();
    const attr = new THREE.BufferAttribute(new Float32Array(size * 3), 3);
    attr.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute("position", attr);
    g.setDrawRange(0, 0);
    const see = t.opacity < 1;
    const obj = t.kind === "line"
      ? new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: new THREE.Color(...t.color), transparent: see, opacity: t.opacity }))
      : new THREE.Points(g, new THREE.PointsMaterial({ color: new THREE.Color(...t.color), size: Math.max(t.size, 4),
                                                       sizeAttenuation: false, transparent: see, opacity: t.opacity }));
    obj.frustumCulled = false;  // the bounding sphere is of the empty buffer
    obj.userData = { name: t.name };
    v.groups.anim.add(obj);
    return obj;
  });
  v.anim = { ...a, arr, objs };
  v.ui.slider.max = String(a.labels.length - 1);
  setFrame(v, Math.min(v.frame, a.labels.length - 1));
  applyMaterials(v);
}

export default function (component) {
  const { data, parentElement } = component;
  const root = parentElement.querySelector(".sf3d");
  if (!root || !data) return;
  root.querySelector(".sf3d-notes").textContent = (data.notes || []).join(" · ");
  loadThree(data.base).then((lib) => {
    let v = viewers.get(parentElement);
    if (!v || !v.alive) {
      v = makeViewer(root, lib);
      viewers.set(parentElement, v);
    }
    legend(v, data);
    updateItems(v, data);
    updateAnimation(v, data).catch((e) => { root.querySelector(".sf3d-status").textContent = String(e); });
    updateMeshes(v, data).catch((e) => { root.querySelector(".sf3d-status").textContent = String(e); });
  }).catch((e) => { root.querySelector(".sf3d-status").textContent = `three.js: ${e}`; });
  // Called before every update as well as on unmount: only an element that
  // has really left the page gives up its WebGL context.
  return () => setTimeout(() => {
    const v = viewers.get(parentElement);
    if (v && !parentElement.isConnected) {
      v.alive = false;
      v.observer.disconnect();
      v.renderer.dispose();
      viewers.delete(parentElement);
    }
  }, 0);
}
"""

# The code's hash is part of the name: a component is registered once per
# name, so a server that reloads this module after an edit would otherwise
# keep running the old JS against the new manifest (a new track kind the
# old viewer silently does not draw -- the 3D bodies were simply missing).
_COMPONENT = component(
    "streamflex_scene_three_" + hashlib.sha1((_HTML + _CSS + _JS).encode()).hexdigest()[:10],
    html=_HTML,
    css=_CSS,
    js=_JS,
)


def three_chart(fig, *, key: str, height: int = 750) -> None:
    """Draw a Plotly 3D figure with three.js (see the module docstring)."""
    problem = three_ready()
    if problem:
        st.warning(problem)
        return
    _COMPONENT(data=figure_manifest(fig), key=key, height=int(height))


# --------------------------------------------------------------------------
# One HTML file to keep: the same viewer, everything inside it
# --------------------------------------------------------------------------

_STANDALONE_CSS = """
:root { --st-background-color: #ffffff; --st-secondary-background-color: #f0f2f6;
        --st-text-color: #31333f; --st-border-color: #c8c8d0; --st-font: system-ui, sans-serif; }
@media (prefers-color-scheme: dark) {
  :root { --st-background-color: #0e1117; --st-secondary-background-color: #262730;
          --st-text-color: #fafafa; --st-border-color: #555a66; }
}
html, body { margin: 0; height: 100%; background: var(--st-background-color); }
#sf3d-host { height: 100vh; }
"""

#: Read by the page's loader below; the viewer's own code is unchanged.
_STANDALONE_LOADER = """
const text = (id) => new TextDecoder().decode(Uint8Array.from(atob(document.getElementById(id).textContent),
                                                               (c) => c.charCodeAt(0)));
const moduleUrl = (src) => URL.createObjectURL(new Blob([src], { type: "text/javascript" }));
const core = moduleUrl(text("sf3d-three-core"));
const three = moduleUrl(text("sf3d-three-module").replaceAll("./three.core.js", core));
const orbit = moduleUrl(text("sf3d-orbit").replaceAll("./three.module.js", three));
const blobs = JSON.parse(document.getElementById("sf3d-blobs").textContent);
globalThis.sf3dStandalone = {
  lib: Promise.all([import(three), import(orbit)])
    .then(([THREE, controls]) => ({ THREE, OrbitControls: controls.OrbitControls })),
  fetch: async (url) => {
    const id = blobs[url];
    if (!id) return new Response(null, { status: 404 });
    const packed = await fetch("data:application/octet-stream;base64," + document.getElementById(id).textContent);
    return new Response(packed.body.pipeThrough(new DecompressionStream("gzip")));
  },
};
"""


def _manifest_urls(node) -> list[str]:
    """Every ``scene3d/...`` file a manifest refers to, in order, once."""
    out: list[str] = []
    if isinstance(node, dict):
        for value in node.values():
            out += _manifest_urls(value)
    elif isinstance(node, list | tuple):
        for value in node:
            out += _manifest_urls(value)
    elif isinstance(node, str) and node.startswith("scene3d/"):
        out.append(node)
    return list(dict.fromkeys(out))


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def standalone_html(fig, *, title: str = "Scene") -> bytes:
    """The figure as one HTML file that plays in any browser, offline.

    The same viewer as ``three_chart``; three.js and every binary mesh and
    track ride inside the page (gzip + base64, unpacked by the browser's
    ``DecompressionStream``), so the file opens straight from disk -- no
    server, no CDN. A full LOD3 city makes a file of a few hundred MB.
    """
    blobs, ids = [], {}
    with _private_blobs():
        data = figure_manifest(fig)
        data["base"] = ""
        for n, url in enumerate(_manifest_urls(data)):
            with open(_blob_path(url), "rb") as fh:
                packed = gzip.compress(fh.read(), compresslevel=6, mtime=0)
            ids[url] = f"sf3d-blob-{n}"
            blobs.append(f'<script type="application/octet-stream" id="sf3d-blob-{n}">{_b64(packed)}</script>')
    libs = []
    for element_id, name in (
        ("sf3d-three-core", "three.core.js"),
        ("sf3d-three-module", "three.module.js"),
        ("sf3d-orbit", "OrbitControls.js"),
    ):
        with open(os.path.join(STATIC_DIR, "three", name), "rb") as fh:
            libs.append(f'<script type="text/plain" id="{element_id}">{_b64(fh.read())}</script>')
    viewer = _JS.replace("export default function (component)", "const mount = function (component)", 1)
    # JSON inside <script>: only "</" could end the element early.
    manifest = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = "\n".join(
        [
            "<!doctype html>",
            '<html><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{html.escape(title)}</title>",
            f"<style>{_STANDALONE_CSS}{_CSS}</style>",
            "</head><body>",
            f'<div id="sf3d-host">{_HTML}</div>',
            *libs,
            f'<script type="application/json" id="sf3d-blobs">{json.dumps(ids)}</script>',
            *blobs,
            f'<script type="application/json" id="sf3d-data">{manifest}</script>',
            '<script type="module">',
            _STANDALONE_LOADER,
            viewer,
            (
                'mount({ data: JSON.parse(document.getElementById("sf3d-data").textContent),'
                ' parentElement: document.getElementById("sf3d-host") });'
            ),
            "</script>",
            "</body></html>",
        ]
    )
    return page.encode("utf-8")


def plotly_html(fig, *, title: str = "Scene") -> bytes:
    """A Plotly figure (animation included) as one offline HTML file."""
    page = fig.to_html(
        include_plotlyjs=True,
        full_html=True,
        auto_play=False,
        config={"scrollZoom": True, "displaylogo": False, "responsive": True},
        default_height="95vh",
    )
    return page.replace("<head>", f"<head><title>{html.escape(title)}</title>", 1).encode("utf-8")


def scene_download(fig, *, renderer: str, key: str, name: str = "scene", container=None) -> None:
    """A button saving the 3D view as an HTML file to open in a browser later.

    In the renderer it is drawn with: three.js (meshes and tracks packed in,
    full detail) or Plotly (the figure as drawn). Built only on click.
    ``fig`` may be a function returning the figure: then the figure itself is
    built on click too (a view that is not drawn on the page).
    """
    three = renderer == "three.js"
    host = container if container is not None else st
    stem = re.sub(r"[^\w.-]+", "_", str(name)).strip("_") or "scene"
    page = standalone_html if three else plotly_html

    def _data() -> bytes:
        return page(fig() if callable(fig) else fig, title=name)

    host.download_button(
        f"⬇ Download 3D view ({'three.js' if three else 'Plotly'} HTML)",
        data=_data,
        file_name=f"{stem}_{'threejs' if three else 'plotly'}.html",
        mime="text/html",
        on_click="ignore",
        key=key,
        help=(
            "One HTML file with everything inside (no server, no internet): open it "
            "from disk in any browser. "
            + (
                "three.js: the full scene and the animation, packed (a LOD3 city is a "
                "few hundred MB)."
                if three
                else "Plotly: the figure as drawn here, animation included."
            )
        ),
    )
