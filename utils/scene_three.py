"""three.js renderer for the 3D scene figures -- the alternative to Plotly.

Plotly cannot draw a full LOD3 city: Streamlit's chart turns the figure into
one string (``RangeError: Invalid string length`` past ~25M numbers) and
plotly.js itself needs minutes for 5M triangles. This renders the *same*
figure the Plotly views build -- ``Mesh3d`` backdrop, ``Scatter3d`` devices
and rays, annotations -- with three.js on the GPU:

* every ``Mesh3d`` becomes a binary file under ``static/scene3d/`` (served by
  Streamlit's static serving, ``server.enableStaticServing``), fetched once
  by the browser and kept by URL, so a rerun re-sends only a manifest;
* markers and lines travel as small JSON in the component's ``data``.

three.js itself is vendored in ``static/three/`` (no CDN).
"""

from __future__ import annotations

import hashlib
import os
import re
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
#: Mesh files kept on disk, oldest dropped first.
MAX_SCENE3D_BYTES = 4 * 1024**3

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
            "meshes, so a full LOD3 city (11.5M triangles) is fine; no animation."
        ),
    )
    return value or DEFAULT_RENDERER


def three_ready() -> str | None:
    """Why three.js cannot run here (``None`` when it can)."""
    if not st.get_option("server.enableStaticServing"):
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


def _write_blob(name: str, parts: list[np.ndarray]) -> None:
    path = os.path.join(SCENE3D_DIR, name)
    if os.path.exists(path):
        os.utime(path)  # recently used: pruned last
        return
    os.makedirs(SCENE3D_DIR, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        fh.writelines(np.ascontiguousarray(p).tobytes() for p in parts)
    os.replace(tmp, path)
    _prune()


def _prune() -> None:
    try:
        files = [os.path.join(SCENE3D_DIR, f) for f in os.listdir(SCENE3D_DIR) if f.endswith(".bin")]
        files.sort(key=os.path.getmtime, reverse=True)
        total = 0
        for f in files:
            total += os.path.getsize(f)
            if total > MAX_SCENE3D_BYTES:
                os.remove(f)
    except OSError:
        pass


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


def _xyz(trace) -> list:
    """Interleaved x, y, z with ``None`` breaks kept (Plotly's line gaps)."""
    out: list = []
    for a, b, c in zip(trace.x or (), trace.y or (), trace.z or (), strict=False):
        if a is None or b is None or c is None or not np.isfinite([a, b, c]).all():
            out += [None, None, None]
        else:
            out += [round(float(a), 2), round(float(b), 2), round(float(c), 2)]
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


def figure_manifest(fig) -> dict[str, Any]:
    """What the three.js component needs from a Plotly 3D figure."""
    meshes, items = [], []
    for trace in fig.data:
        if trace.type == "mesh3d":
            meshes += _mesh_entries(trace)
        elif trace.type == "scatter3d" and trace.visible not in (False, "legendonly"):
            items += _scatter_entries(trace)
    notes = [str(a.text) for a in (fig.layout.annotations or ()) if a.text]
    base = "/" + str(st.get_option("server.baseUrlPath") or "").strip("/")
    return {
        "base": (base.rstrip("/") + "/app/static/"),
        "meshes": meshes,
        "items": items,
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
"""

_JS = """
const geometryCache = new Map();   // url -> Promise<THREE.BufferGeometry>
const viewers = new WeakMap();     // parentElement -> viewer
let threePromise = null;

function loadThree(base) {
  if (!threePromise) {
    const abs = (p) => new URL(base + p, window.location.origin).href;
    threePromise = Promise.all([import(abs("three/three.module.js")), import(abs("three/OrbitControls.js"))])
      .then(([THREE, controls]) => ({ THREE, OrbitControls: controls.OrbitControls }));
  }
  return threePromise;
}

function geometry(THREE, base, m) {
  if (!geometryCache.has(m.url)) {
    const url = new URL(base + m.url, window.location.origin).href;
    geometryCache.set(m.url, fetch(url).then((r) => {
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
  const groups = { meshes: new THREE.Group(), items: new THREE.Group() };
  scene.add(groups.meshes, groups.items);
  const v = { lib, root, renderer, scene, camera, controls, groups, fitted: false, alive: true,
              opaque: false, hidden: new Set(), meshKey: "", itemKey: "" };
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

_COMPONENT = component("streamflex_scene_three", html=_HTML, css=_CSS, js=_JS)


def three_chart(fig, *, key: str, height: int = 750) -> None:
    """Draw a Plotly 3D figure with three.js (see the module docstring)."""
    problem = three_ready()
    if problem:
        st.warning(problem)
        return
    _COMPONENT(data=figure_manifest(fig), key=key, height=int(height))
