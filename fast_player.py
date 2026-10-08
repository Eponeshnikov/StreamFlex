"""A step player for long Plotly animations that does not hang the browser.

A Plotly figure with one ``go.Frame`` per step keeps every frame as a
JavaScript object (10 000 steps ≈ 140 MB of JSON for the ACF chart). Here the
same frames — produced by the same code as the page's animation, so the
chart shows exactly the same marks — are packed once:

* long numeric arrays go to one binary blob: ρ-like values in [−1, 1] as
  int8 (·127, −128 = NaN), small integers (cell codes) as uint8, anything
  else as float32;
* everything else (short x/y lists, labels, colours, titles) stays JSON;
* both are zlib-compressed and unpacked in the browser
  (``DecompressionStream``).

The browser keeps only the current frame live: moving the slider merges that
frame into the figure and calls ``Plotly.react``. Long line traces are drawn
with WebGL (``scattergl``).

Theme: embedded in the page, colours follow the Streamlit theme read from the
parent document (as the CIR-Generator 3D player does); standalone, they follow
``prefers-color-scheme`` with a light/dark toggle.
"""

from __future__ import annotations

import base64
import json
import zlib
from collections.abc import Iterable, Sequence
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

_MIN_BLOB = 32  # arrays at least this long go to the binary blob


class _Packer:
    def __init__(self) -> None:
        self.parts: list[bytes] = []
        self.size = 0

    def _put(self, raw: bytes) -> int:
        offset = self.size
        self.parts.append(raw)
        self.size += len(raw)
        return offset

    def array(self, a: np.ndarray) -> Any:
        a = np.asarray(a)
        if a.dtype.kind in "iub" and a.size and a.min() >= 0 and a.max() <= 255:
            return {"$b": ["u8", self._put(a.astype(np.uint8).tobytes()), int(a.size)]}
        f = a.astype(np.float32)
        finite = f[np.isfinite(f)]
        # int8 only for ρ-like data that actually spans ~[−1, 1]; tiny-scale
        # data (delays ~1e-7 s, path gains ~1e-6) would quantise to 0.
        if finite.size == 0 or 0.25 <= np.abs(finite).max() <= 1.0001:
            q = np.where(np.isfinite(f), np.clip(np.round(f * 127), -127, 127), -128)
            return {"$b": ["q8", self._put(q.astype(np.int8).tobytes()), int(a.size)]}
        pad = (-self.size) % 4  # Float32Array needs 4-byte alignment
        if pad:
            self._put(b"\0" * pad)
        return {"$b": ["f4", self._put(f.tobytes()), int(a.size)]}

    def value(self, v: Any) -> Any:
        if isinstance(v, dict):
            return {k: self.value(x) for k, x in v.items()}
        if isinstance(v, np.ndarray):
            if v.ndim == 1 and v.size >= _MIN_BLOB and v.dtype.kind in "iufb":
                return self.array(v)
            if v.ndim >= 2 and v.dtype.kind in "iufb":
                return [self.value(row) for row in v]
            return _plain(v.tolist())
        if isinstance(v, (list, tuple)):
            if (len(v) >= _MIN_BLOB and all(isinstance(x, (int, float, np.number)) or x is None
                                             for x in v)):
                return self.array(np.array([np.nan if x is None else x for x in v], float))
            return [self.value(x) for x in v]
        return _plain(v)

    def blob(self) -> bytes:
        return b"".join(self.parts)


def _plain(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        f = float(v)
        # Significant digits, not decimals: round(1e-7, 5) == 0.
        return None if not np.isfinite(f) else float(f"{f:.6g}")
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, list):
        return [_plain(x) for x in v]
    return v


def _webgl(base: dict) -> None:
    """Long line traces → scattergl (GPU); markers/text traces stay SVG."""
    for tr in base["data"]:
        if tr.get("type", "scatter") == "scatter" and tr.get("mode") == "lines":
            x = tr.get("x")
            if x is not None and len(x) >= _MIN_BLOB:
                tr["type"] = "scattergl"


def render(base_fig: go.Figure, animated: Sequence[int] | None,
           frames: Iterable[tuple], labels: Sequence[Any], *,
           standalone: bool, height: int | None = None, key: str = "fastplayer",
           prefix: str = "Time step ", start: int = 0) -> str:
    """HTML of a player over ``frames`` on top of ``base_fig``.

    Each frame is ``(data, layout_patch)`` or ``(data, layout_patch, traces)``:
    ``data`` updates the traces ``traces`` (or ``animated``, or 0..len−1),
    ``layout_patch`` is merged into the layout (a str = the title)."""
    packer = _Packer()
    packed = []
    for fr in frames:
        data, patch = fr[0], fr[1]
        traces = fr[2] if len(fr) > 2 and fr[2] is not None else animated
        if isinstance(patch, str):
            patch = {"title": {"text": patch}}
        packed.append({"d": [packer.value(d) for d in data], "l": packer.value(patch or {}),
                       "t": None if traces is None else [int(t) for t in traces]})
    base = json.loads(cast(str, base_fig.to_json()))
    layout = base["layout"]
    layout.pop("sliders", None)
    # Plotly's own play / step menus go (the player has its own); any other
    # menu — e.g. a restyle toggle — stays.
    menus = [m for m in layout.pop("updatemenus", None) or []
             if not any(b.get("method") == "animate" for b in m.get("buttons", []))]
    if menus:
        layout["updatemenus"] = menus
    layout["template"] = None
    layout["uirevision"] = "keep"  # zoom / 3D camera survive the redraws
    if height:
        layout["height"] = height
    _webgl(base)
    payload = {
        "base": base, "labels": [str(x) for x in labels],
        "prefix": prefix, "start": int(start), "standalone": standalone,
        "frames": base64.b64encode(zlib.compress(json.dumps(packed).encode(), 6)).decode(),
        "blob": base64.b64encode(zlib.compress(packer.blob(), 6)).decode(),
    }
    return (_TEMPLATE.replace("__KEY__", key)
            .replace("__PLOTLY__", get_plotlyjs())
            .replace("__PAYLOAD__", json.dumps(payload)))


def from_figure(fig: go.Figure, *, standalone: bool = False, key: str = "fastplayer",
                height: int | None = None) -> str:
    """Any Plotly figure with ``frames`` (and the slider of
    ``utils.add_plotly_frame_slider``) as a fast player — the same frames."""
    layout = fig.layout
    labels, prefix = [], "Frame "
    if layout.sliders:
        slider = layout.sliders[0]
        labels = [s.label for s in slider.steps]
        if slider.currentvalue and slider.currentvalue.prefix:
            prefix = slider.currentvalue.prefix
    if len(labels) != len(fig.frames):
        labels = [f.name if f.name is not None else str(i) for i, f in enumerate(fig.frames)]

    def frames():
        for f in fig.frames:
            data = [d.to_plotly_json() for d in (f.data or [])]
            patch = f.layout.to_plotly_json() if f.layout else {}
            patch.pop("sliders", None)
            patch.pop("updatemenus", None)
            traces = list(f.traces) if f.traces is not None else list(range(len(data)))
            yield data, patch, traces

    return render(fig, None, frames(), labels, standalone=standalone, key=key,
                  height=height or (int(layout.height) if layout.height else None),
                  prefix=prefix)


_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Step player</title>
<script>__PLOTLY__</script>
<style>
 :root{--bg:#0e1117;--fg:#fafafa;--btn:#262730;--btnh:#31333f;--bd:rgba(250,250,250,.24)}
 html,body{margin:0;background:var(--bg);color:var(--fg);font:14px "Source Sans Pro",system-ui,sans-serif}
 #bar{display:flex;gap:8px;align-items:center;padding:6px 4px;flex-wrap:wrap}
 #bar input[type=range]{flex:1;min-width:180px;accent-color:#ff4b4b}
 button,select{background:var(--btn);color:var(--fg);border:1px solid var(--bd);border-radius:6px;padding:3px 10px;cursor:pointer;font:inherit}
 button:hover{background:var(--btnh)}
 #lbl{min-width:150px;font-variant-numeric:tabular-nums}
 #load{padding:24px;opacity:.7}
</style></head><body>
<div id="load">Unpacking…</div>
<div id="__KEY__" style="display:none"></div>
<div id="bar" style="display:none">
 <button id="prev" title="previous step (←)">◀</button><button id="next" title="next step (→)">▶</button><button id="play" title="play / pause">Play</button>
 <input id="pos" type="range" min="0" value="0"><span id="lbl"></span>
 <select id="speed" title="steps per tick"><option>1</option><option selected>5</option><option>20</option><option>100</option></select>
 <button id="theme" style="display:none">◐</button>
</div>
<script>
(async function () {
const P = __PAYLOAD__;
const gd = document.getElementById("__KEY__");
async function inflate(b64) {
  const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const s = new Blob([bin]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(s).arrayBuffer());
}
const frames = JSON.parse(new TextDecoder().decode(await inflate(P.frames)));
const blob = (await inflate(P.blob)).buffer;
function arr(ref) {
  const [kind, off, n] = ref;
  if (kind === "u8") return Array.from(new Uint8Array(blob, off, n));
  if (kind === "f4") return new Float32Array(blob, off, n);
  const q = new Int8Array(blob, off, n), out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = q[i] === -128 ? NaN : q[i] / 127;
  return out;
}
function resolve(v) {
  if (Array.isArray(v)) return v.map(resolve);
  if (v && typeof v === "object") {
    if (v.$b) return arr(v.$b);
    const o = {}; for (const k in v) o[k] = resolve(v[k]); return o;
  }
  return v;
}
function merge(dst, src) {
  for (const k in src) {
    const v = src[k];
    if (v && typeof v === "object" && !Array.isArray(v) && !ArrayBuffer.isView(v) && dst[k]
        && typeof dst[k] === "object" && !Array.isArray(dst[k])) merge(dst[k], v);
    else dst[k] = v;
  }
}
const data = P.base.data, layout = P.base.layout;
layout.datarevision = 0;
document.getElementById("load").remove();
gd.style.display = ""; document.getElementById("bar").style.display = "";
await Plotly.newPlot(gd, data, layout, {responsive: true, displaylogo: false});

// ---- theme ----
const AXES = Object.keys(layout).filter(k => /^[xy]axis\d*$/.test(k));
function rgb(s) { const m = (s || "").match(/rgba?\(([^)]+)\)/); if (!m) return null;
  const p = m[1].split(",").map(Number); return p.length >= 3 && (p.length < 4 || p[3] > 0) ? p : null; }
const lum = c => (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255;
function parentColors() {
  try { const doc = window.parent && window.parent !== window && window.parent.document;
    if (!doc) return null;
    const root = doc.querySelector('[data-testid="stAppViewContainer"]') || doc.querySelector(".stApp") || doc.body;
    const st = window.parent.getComputedStyle(root), sb = window.parent.getComputedStyle(doc.body);
    const bg = rgb(st.backgroundColor) ? st.backgroundColor : sb.backgroundColor;
    const fg = rgb(st.color) ? st.color : sb.color;
    return rgb(bg) && rgb(fg) ? {bg, fg} : null; } catch (e) { return null; }
}
let forced = null, lastSig = "";
function schemeColors() {
  const dark = forced ? forced === "dark" : !window.matchMedia("(prefers-color-scheme: light)").matches;
  return dark ? {bg: "rgb(14,17,23)", fg: "rgb(250,250,250)"} : {bg: "rgb(255,255,255)", fg: "rgb(49,51,63)"};
}
function applyTheme(force) {
  const c = (!P.standalone && parentColors()) || schemeColors();
  const f = rgb(c.fg), light = lum(rgb(c.bg)) > 0.5;
  const sig = c.bg + c.fg; if (!force && sig === lastSig) return; lastSig = sig;
  const a = x => `rgba(${f[0]},${f[1]},${f[2]},${x})`;
  const R = document.documentElement.style;
  R.setProperty("--bg", c.bg); R.setProperty("--fg", c.fg);
  R.setProperty("--btn", light ? "#f0f2f6" : "#262730"); R.setProperty("--btnh", light ? "#e6e9ef" : "#31333f");
  R.setProperty("--bd", a(0.24));
  const up = {"font.color": c.fg, paper_bgcolor: c.bg, plot_bgcolor: c.bg,
              "legend.bgcolor": "rgba(0,0,0,0)", "hoverlabel.bgcolor": light ? "#fff" : "#262730"};
  for (const ax of AXES) { up[ax + ".gridcolor"] = a(light ? 0.13 : 0.14);
    up[ax + ".zerolinecolor"] = a(light ? 0.3 : 0.28); up[ax + ".linecolor"] = a(0.3);
    up[ax + ".color"] = c.fg; }
  for (const ann of (layout.annotations || [])) { ann.font = Object.assign(ann.font || {}, {color: c.fg}); }
  Plotly.relayout(gd, up);
}
if (P.standalone) {
  const b = document.getElementById("theme"); b.style.display = "";
  b.onclick = () => { forced = (lum(rgb(getComputedStyle(document.body).backgroundColor) || [0,0,0]) > 0.5) ? "dark" : "light"; applyTheme(true); };
  window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => applyTheme(true));
} else {
  try { const doc = window.parent.document;
    new MutationObserver(() => applyTheme(false)).observe(doc.documentElement,
      {attributes: true, subtree: false, attributeFilter: ["class", "style", "data-theme"]});
  } catch (e) {}
  setInterval(() => applyTheme(false), 700);
}
applyTheme(true);

// ---- player ----
const pos = document.getElementById("pos"), lbl = document.getElementById("lbl");
pos.max = frames.length - 1;
let cur = -1, pending = null;
function show(i) {
  const fr = frames[i];
  const tr = fr.t || fr.d.map((_, j) => j);
  tr.forEach((t, j) => { if (gd.data[t]) merge(gd.data[t], resolve(fr.d[j])); });
  const patch = resolve(fr.l);
  if (typeof patch.title === "string") patch.title = {text: patch.title};
  if (patch.title && typeof gd.layout.title === "string") gd.layout.title = {text: gd.layout.title};
  merge(gd.layout, patch);
  gd.layout.datarevision = (gd.layout.datarevision || 0) + 1;
  Plotly.react(gd, gd.data, gd.layout);
  lbl.textContent = P.prefix + P.labels[i] + `  (${i + 1}/${frames.length})`;
  cur = i;
}
function go(i) { i = Math.max(0, Math.min(frames.length - 1, i)); pos.value = i;
  if (pending === null) pending = requestAnimationFrame(() => { pending = null; if (+pos.value !== cur) show(+pos.value); }); }
pos.addEventListener("input", () => go(+pos.value));
document.getElementById("prev").onclick = () => go(cur - 1);
document.getElementById("next").onclick = () => go(cur + 1);
document.addEventListener("keydown", e => { if (e.key === "ArrowLeft") go(cur - 1); if (e.key === "ArrowRight") go(cur + 1); });
let timer = null; const play = document.getElementById("play");
play.onclick = () => { if (timer) { clearInterval(timer); timer = null; play.textContent = "Play"; return; }
  play.textContent = "Pause"; timer = setInterval(() => {
    if (cur >= frames.length - 1) { clearInterval(timer); timer = null; play.textContent = "Play"; return; }
    go(cur + +document.getElementById("speed").value); }, 80); };
show(Math.min(P.start, frames.length - 1)); pos.value = cur;
window.__player = {show, n: frames.length, ms(i) { const t = performance.now(); show(i); return performance.now() - t; }};
})();
</script></body></html>
"""
