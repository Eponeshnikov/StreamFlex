# StreamFlex core architecture

## What this is

StreamFlex is a Streamlit application for **wireless-channel signal simulation and
physical-layer key generation research**. It has two loosely-coupled halves:

1. **An online plugin pipeline** (`plugins/`) — interactive simulation stages that
   pass data through an in-memory `DataManager` and persist results to
   `output_data/` as Parquet/pickle.
2. **An offline analysis pipeline** (`pages/`) — Streamlit multipage scripts that
   read the Parquet files produced by the plugins and run peak detection,
   dataset preparation, model training (Optuna + XGBoost), and key generation.

The plugin framework itself (loader, snapshots, persistent widgets) is generic;
the wireless domain lives entirely inside the individual plugins and pages.

## Commands

```bash
uv sync                       # install all deps (requires `uv`); see pyproject groups below
streamlit run app.py          # run the app directly
./run_streamlit.sh            # preferred: runs inside tmux + a memory-capped systemd scope
./kill_streamlit.sh           # stop it
```

- Python **3.12** (`.python-version`). Dependencies are managed by **uv** in
  `pyproject.toml` — `requirements.txt` is a stale minimal subset, ignore it.
- `pyproject.toml` splits deps into **per-plugin dependency-groups**
  (`cir-generator`, `opti-receiver`, `signal-channelizer`, …) plus a `dev` group
  holding the heavy analysis stack (xgboost, optuna, shap, polars, torch). Torch
  comes from the `pytorch-cu130` index.
- `pyproject.toml` configures pytest and Ruff. The local `tests/` suite
  exercises the real plugin chain via Streamlit AppTest.
- `run_streamlit.sh` exports `DRJIT_LIBLLVM_PATH` (Dr.Jit/Sionna needs a
  versioned `libLLVM.so` that Fedora doesn't symlink) and runs under
  `systemd-run --scope -p MemoryMax` with swap disabled — large runs OOM the
  whole box otherwise.

## Core framework (repo root)

`app.py` is the Streamlit entry point. It builds the top navigation by scanning
`pages/` and bucketing files into categories defined in
`configs/page_categories.yaml`. **Conventions enforced here:**
- `pages/*_worker.py` and any file in the YAML `exclude:` list are **import-only
  modules, not nav pages** — they live under `pages/` only so multiprocessing can
  pickle them by qualified name. Do not expect them to render.
- The `private_` filename prefix is stripped for display titles only.

`streamflex_app.py` is the main "App" page: it instantiates the four managers,
loads plugins, renders the plugin multiselect, and runs selected plugins. It also
owns the sidebar (snapshots, cache clearing, log console, the **global trigger**).

The four managers (all root-level, all backed by `st.session_state`):
- `plugin_manager.py` — discovers `plugins/<Name>/plugin.py`, loads each via
  `importlib`, instantiates every `Plugin` subclass found. Can also `git clone`
  a plugin from a URL and `pip install` its `requirements.txt`.
- `plugin.py` — `Plugin` base class. Key method `create_widget(...)` wraps every
  Streamlit widget so its value is persisted through `WidgetManager` (keyed
  `<PluginName>_<widget_name>`) and survives reruns/snapshots. `run_notification`
  wraps `run()` with retry-on-`StreamlitAPIException`. Subclasses set
  `self.file = __file__` before `super().__init__()` so config/path resolution
  points at the plugin's own directory.
- `data_manager.py` — `DataManager.set_data/get_data(key, ...)`. This dict is the
  **bus between plugins** (see pipeline below).
- `widget_manager.py` — persistent widget state, exported/imported by snapshots.
- `state_manager.py` — `snapshots/*.pkl`: pickles `{data, widgets,
  selected_plugins}` so a whole session can be saved/restored. It also owns
  the **autosave**, which is what survives a closed browser (below).

### Surviving a closed browser

Streamlit ties both halves of a session to the websocket, and each half needs
its own fix.

**The run.** `WebsocketSessionManager.disconnect_session` calls
`AppSession.request_script_stop()` unconditionally — a closed tab, a sleeping
laptop or a client that fails to drain a big message aborts whatever the
script is doing, with no config option to stop it. That is how a trajectory
run died four minutes into OptiReceiver on 2026-09-07, leaving 8 GB of
memmaps nothing points at. `session_guard.install_session_guards()` suppresses
that one call while the script is running, and only then; the rest of the
disconnect (storage, file watchers, session caches) is untouched. `app.py`
installs it, so the offline pages are covered too — a peaks-processing or
training run dies the same way otherwise. `server.disconnectedSessionTTL`
(30 min here, streamlit's default is 120 s) is how long a *dropped* connection
can still reattach; it does nothing for a reload, which always gets a new
session, because the reconnect id lives only in the page that created it.

A page that *did* survive (a browser restored with its tabs, a laptop that
woke up) reattaches, and was the second way a run died: the reattached
client's first message is a `rerun_script`, and with `runner.fastReruns` (on
by default) a rerun stops the running script at once — through
`ScriptRunner.request_stop`, which the disconnect guard never sees. On
2026-10-23 that killed a ray-tracing run 25 minutes in, and the Generate
pill (sticky) started the trace again from zero. The guard marks a session
reattached mid-run (`connect_session`), holds that one rerun
(`AppSession.request_rerun`) and replays it when the current runner's
`SHUTDOWN` event arrives, so the browser gets its full page once the run is
done; anything clicked afterwards interrupts as usual. A reload's new
session is told by `session_guard.detached_runs()` that a run is still
going, and the main page (`_wait_for_detached_run`) holds its plugins back
instead of restoring Generate from the autosave and tracing the same scene
beside it (*Run plugins here anyway* overrides).

**The state.** So a reopened browser starts empty, and the autosave is what
refills it: `snapshots/_autosave.widgets.pkl` (widgets + plugin selection,
kilobytes) is written after every run, `snapshots/_autosave.pkl` (the same
plus the whole data bus) at most every `AUTOSAVE_DATA_INTERVAL_S`, backing
off to 20× the last write when that write was slow — the bus is gigabytes on
a trajectory run. `streamflex_app.autoresume_state` loads them into a fresh
session *before the first widget is created*, since the restored plugin
selection is written straight into the multiselect's session-state key. Both
names start with `_`, which is what keeps them out of `list_snapshots` and the
manual picker. Writes go through `_write_pickle` (temp file + `os.replace`):
the autosave overwrites the same two names forever, and one of those writes
will eventually be interrupted.

`utils/` is a large shared toolbox used by both plugins and pages — a package
(`config_display`, `files`, `charts`, `logs`, `cache`, `storage`, `radiomap`,
`scene_mesh`, `scene_render`, `axis_metadata`, `memory`, `workers`, `gpu`,
`chunking`) whose `__init__` re-exports every name, so `from utils import x`
is unchanged; tests that patch a global patch the submodule reading it. Notable
exports: `cache_result` (file-based **pickle** cache in `.cache/`, works across
processes — not `st.cache_data`), `file_input` (file picker that also browses a
`default_dir` on disk, returning an `UploadedFile`-compatible object), Plotly/
Sionna scene rendering helpers, and `logger_init`/`get_colored_logs` (Loguru →
`logs/`).

### Animated result plots

`utils.visualization_time_control` is the shared time control for the Signal
Channelizer and OptiReceiver visualizers **and for the CIR-Generator's taps
and CFR tabs**. Its *Smooth animation (Plotly)*
mode — the default — used to return one frame per time step, and on a
trajectory run the time axis *is* the RX walk: 3000 points became 3000
frames and a single **361 MB** delta, which the browser could not drain. The
websocket dropped and, before `session_guard` existed, that killed the
OptiReceiver stage running behind it.

Frames are the only lever. The samples *inside* a frame are the waveform, and
thinning those moves a pulse or a peak off where it actually is — the same
reason the scene renderer refuses to decimate geometry. So the frame set is
strided down to `MAX_ANIMATION_FRAMES` / `MAX_ANIMATION_PAYLOAD_BYTES`,
covering the whole span at a coarser step, and callers pass `frame_bytes` (the
estimated wire cost of one frame) so the cap follows the real payload rather
than a round number. The slider mode is never decimated: it sends one frame.
`add_planar_radiomap_to_figure` does the same thing for radiomap cells
(`MAX_RADIOMAP_CELLS_PER_AXIS`), and for the same reason. The striding is
`decimate_animation_indices`, public because the frame cap is not specific to
the time axis — the CIR-Generator animates a *trajectory RX* axis the same
way, and it is the same 3000 frames.

The taps and CFR tabs had their own time picker, and it offered the smooth
mode only when the time axis carried **no** trajectory metadata — that is,
never on the runs where scrubbing a walk is the point, and without any frame
cap on the modes it did offer. Both now go through the shared control
(`_render_time_selector` is gone; a second, non-decimating picker is exactly
the trap this removes). They also stack the duplex directions the way the CIR
tab does: taps get one row per direction, CFR two (magnitude over phase).
`_duplex_side_rows` is the *whole* duplex control — while both rows are
drawn it renders no direction selector at all, because there is nothing left
for one to choose; unticking the box brings it back, and a layout with
several batches per side keeps it either way (it then picks which batch, and
both of its sides are stacked). Stacked rows share a y-range through
`_shared_ranges`, computed once over every frame and side: two directions on
different scales do not compare, and an axis that rescales per frame cannot
be read at all.

### The one 3D scene renderer

Every page and plugin that draws a Sionna scene goes through
`utils.render_sionna_scene_plotly` / `scene_mesh_traces` — Scene Preview,
the OSM builder's post-build preview, the CIR-Generator's scene figures
and the Position Selector. Do not add another mesh-to-Plotly path.

Plotly ships every triangle over the websocket on **every rerun**, so an
LOD3 city is a ~56 MB message per redraw. The levers, in the order they
matter:
- **Layer filter.** `scene_layer_of` buckets shapes by the OSM writer's
  naming into `SCENE_LAYERS`; `scene_display_controls` renders the
  multiselect (plus the material-legend toggle) and every caller passes
  the result down. On a Kazan LOD3 city the land-cover surfaces alone are
  two thirds of the triangles: hiding them takes the preview from 56 MB
  to 18 MB, buildings-only to 8 MB. **Layers only exist on a scene loaded
  with `merge_shapes=False`** — merging renames every shape `no-name-N`;
  the controls detect that and say so instead of offering one bucket.
- **Grouping by radio material**, not by shape. Plotly pays a fixed cost
  per trace and a city has ~2000 shapes. This is also why nothing needs
  to load merged just to be drawn: the renderer already collapses to one
  trace per material, and merging only costs the names.
- **`weld_mesh` (`_weld_mesh`)** merges exactly coincident vertices —
  lossless, no rounding. ~25% off the vertex count; triangles unchanged.
- **No decimation.** Snapping vertices to a lattice was tried and
  removed: it shrinks the payload but collapses walls into each other,
  and the scene falls apart visually long before it gets small.

**The rays are the other half of the payload**, and they are *not*
geometry: a munich solve at depth 3 returns 12389 rays, i.e. 32213
segments once they are coloured Sionna-style. `add_paths_to_figure` used
to emit one Scatter3d per segment — 32213 traces, ~10 MB of JSON, and a
plotly.js scene object per trace, which is what made the view lag rather
than the city behind it. Segments are now **batched into one trace per
(interaction type, opacity, width)**, polylines separated by `None`:
3 traces and 1.6 MB for the same 12389 rays, every one of them still
drawn. Nothing is decimated here — the frame cap and the radiomap cap
drop *samples*, and a ray dropped from a picture of multipath is the
picture's subject. Coordinates are rounded to millimetres (Plotly ships
them as JSON text) and ray traces carry `hoverinfo="skip"`: a hover pick
pass over 96k vertices costs more than the type name it could report,
which the legend already carries. Scatter3d is WebGL (`gl3d`) already —
there is no renderer to switch on, and the cost that was hurting is
per-trace bookkeeping, not rasterization.

`color_paths_by_segment` (Sionna's own colouring: grey incident segment,
then the colour of each interaction) defaults to **on**. Its checkbox
lives inside `render_sionna_scene_plotly`, which the CIR-Generator's 3D
view cannot use — that figure is built in a child process with no
Streamlit, where the widget threw and was swallowed, so the plugin owns
its own control and passes the value in `render_kwargs`
(`show_segment_toggle=False` from the worker).

`opacity_from_power` fades each ray by its own `|a|²` (summed over
antennas): the per-type sliders stay the ceiling, the strongest ray on
screen gets exactly their opacity, and `power_range_db` below it a ray
reaches `MIN_POWER_OPACITY_FRACTION` of it — a floor, not zero, because
the mapping is there to rank rays, not to hide them. Opacity is a trace
property, so the levels are snapped to `POWER_OPACITY_LEVELS`; that is
what keeps the trace count bounded (~34 on the munich solve) instead of
one trace per distinct power. The 3D animation buckets frames by
*(name, opacity)* for the same reason — keying on the name alone would
flatten every level onto the first one's.

`scene_mesh_parts` caches the welded geometry on `scene_content_key`
(shape names + face counts, free to compute — a rebuilt scene keys
differently by itself). Opacity and the legend flag are applied when the
traces are built, *not* baked into that key: they change constantly and
each entry runs to tens of megabytes.

`load_scene_cached(path, merge_shapes)` is the shared loader, with
`ttl=300, refresh_mode="background"` (Streamlit ≥1.61): a scene rebuilt
by the OSM builder while the app is open used to need a manual cache
clear, and now swaps itself in off the script thread.

**Building off the script thread.** `refresh_mode="background"` only ever
refreshes an entry that already exists — it cannot make a cold
computation asynchronous, and there is no Streamlit API that can. What
works is `scene_mesh_parts(..., background=True)`: the first call submits
the weld to a one-worker `ThreadPoolExecutor` (kept in a
`cache_resource` singleton alongside its own result dict — the worker has
no ScriptRunContext, so it must not touch Streamlit's cache) and returns
`None`. The caller draws a placeholder and polls with
`st.fragment(run_every=0.5)`, applying the decorator per run so
`run_every=None` **stops** the polling once the mesh lands: a standing
timer would re-send the whole figure every half second, which is the very
cost this is avoiding. Welding is numpy and Mitsuba buffer reads, so it
releases the GIL often enough to keep the page usable (worst measured
stall ~0.3 s against a 3 s hard block). This takes the *build* off the
critical path, not the websocket transfer — the payload still goes out
synchronously once it exists.

## The plugin pipeline (this is the important part)

Plugins communicate **only** through `DataManager` string keys. Each stage reads
its predecessor's key and writes its own. Selecting and running the plugins in
order (or firing the sidebar **global trigger**, which every plugin's autosave/
generate pill can opt into) walks the chain:

```
CodeGenerator ──> "gold_codes" / "m_sequences" / "kasami_codes"
                        │
IQGenerator  ──reads codes──> "iq_generator"
                        │
Pulse-Shaping ──reads "iq_generator"──> "pulse_shaping"
                        │
CIR-Generator ──> "cir_generator"     (independent source: channel impulse responses)
                        │
SignalChannelizer ──reads "pulse_shaping" + "cir_generator"──> "signal_channelizer"
                        │
OptiReceiver ──reads "signal_channelizer" + "pulse_shaping" + "cir_generator"──> "optimal_receiver"
                        │
ResultsSaverPlugin ──reads ANY of the above keys──> Parquet in output_data/
```

Every payload is a `list[dict]` where each item has
`{"config_info": {"id", "parameters", "source_info", ...}, "results": {...}}`.
`source_info` carries the upstream `config_info` so lineage is preserved down the
chain. Complex arrays are stored split as `{"re": ..., "im": ...}`.

**Downloading a 3D view** (`utils/scene_three.scene_download`, drawn after
every 3D chart: Position Selector preview and playback, Scene Preview, OSM
scene preview, CIR-Generator 3D). One HTML file in the renderer the view
is drawn with, built only on click (callable `data=`). Plotly:
`fig.to_html(include_plotlyjs=True)` -- the figure as drawn, animation
included. three.js: `standalone_html` runs `figure_manifest` and packs
every `scene3d/*.bin` it names (gzip, base64, one `<script
type="application/octet-stream">` each) plus the vendored three.js
(base64, turned into blob-URL modules with their relative imports
rewritten) into the page; the viewer JS is the component's own, its
`export default` made a local `mount` called on a host `<div>`. The viewer
asks `globalThis.sf3dStandalone` for three.js and for each binary
(`DecompressionStream("gzip")` over a `data:` fetch) before it goes to
static serving, so the file opens from `file://` with no server or
network (`window.location.origin` is `"null"` there). Kazan LOD3 playback
(0.85M triangles shown, 1000 cars, 7205 people, 1500 frames): 228 MB, 17 s
to build; checked in headless Chromium (swiftshader).

**3D off the page** (`on_page_control`, toggle *Draw 3D on this page*; in
the Position Selector one sidebar toggle for its one 3D view, elsewhere next to the renderer, `onpage3d_<cfg>` in the
CIR-Generator). Off, no chart is sent. In the Position Selector and the
OSM preview the figure is not built either: `scene_download` takes a
function and builds it synchronously on click (`background=False`, so no weld poller runs), with the layers,
renderer and *Show* choice of that moment -- a weak machine prepares the
file, a strong one plays it. The Position Selector has exactly one 3D
view and one control for it (sidebar toggle + renderer/opacity/layers;
"Off" was dropped -- download-only already sends and builds nothing). The
view is a container created at its old place below the tabs and filled
at the end of the run, because what it can show is only known then: the
Scenario tab and the Auto-scenario tab `register_playback` their walk
(`SOURCES_KEY`, reset each run), and a segmented control above the view
(`pos_sel_3d_content`) picks the scene with TX/RX or one walk. *Open a
saved scenario* sets it to the scenario walk. The per-tab *3D playback*
toggles are gone.

**Which frames.** The playback is put on frames ``frame_dt`` apart over a
``window`` of the walk's clock (`build_playback(frame_dt=, window=)`).
The step choices are the walk's own dt times 1, 2, 4, 5, 10, 20, 25, 50,
... up to half the window (`frame_steps`), each labelled with its frame
count, so the walk itself is never interpolated at a multiple of its step.
Everything else is resampled onto the frame instants the way the CIR
Generator puts a traffic file on the walk: `traffic.combine.sample_at`
(positions linear between the file's samples, heading along the shortest
turn, held at the ends), 256 instants at a time; the walk and Eve by
`Walk.at` (linear between walked points, half a step of slack at each end
so consecutive handover legs meet). This matters for the crowd: people
are stored every 0.1 s, cars and the walk every 0.005 s, and a nearest-
sample pick made the people jump every 0.1 s. Frames past the cap are
spread evenly over the window with a warning naming the window length
that keeps the chosen step.

**Cars and people as compact tracks.** They used to be Plotly frames of
pose rows (7 float32 = 28 B per body per frame): Kazan's 8205 bodies over
3000 frames were a 410 MB file, 8.7 GB peak on the server (the go.Frame
objects), and 6000 frames would not open. Now `playback.Bodies` holds
them as int16 per body and frame -- x, y in `unit_xy` (1 cm up to a 650 m
span) from the traffic's centre, z_ground in cm, heading in pi/32767,
x = -32768 when the body is off the map -- with the per-body scale
stored once. `playback_figure(models=True)` hands them to three.js on
`fig._sf3d_bodies` (no Plotly trace, no frames); `scene_three` writes each
as frame-to-frame int16 differences (wrapping subtraction; the viewer's
Int16Array sum wraps back exactly) and the viewer poses an InstancedMesh
from them (`setBodies`). Bodies move centimetres per frame, so the
differences gzip ~18x: 12000 frames = 81 MB file (54 s, 6.4 GB peak),
20001 (the walk's own 0.005 s over 100 s) = 102 MB (87 s, 7.4 GB peak),
both decoded correctly in headless Chromium at late frames. Plotly still
gets markers, from `Bodies.rows(f)`. The resampling is `_lerp_plan`
(``sample_at``'s rules without speeds and road classes: 75 s -> 7 s for a
7k crowd). The frame cap follows the bytes (`frame_cap`): three.js 8 B per
body per frame against 192 MB on the page (fetched raw from static
serving) or 1.5 GB in a file (what the browser holds unpacked), Plotly
~30 B against 48 MB; a LOD2 walk with a few hundred bodies gets many more
frames than the 7k crowd. `standalone_html` packs its binaries in a
private temp folder under the scratch root (`_private_blobs`): writing
them to `static/scene3d` let one long download push `static/` past the
1 GB at which Streamlit turns static serving off at start-up. *Time window* is a range slider snapped to the walk's
dt; *Jump to* fills it with a leg (`walk_legs`, multi-tower walks only) or
with ±min(5 s, 10 % of the walk) around a handover (`handovers`: the next
leg's start, where the tower changes). The file name carries the window.

**VRAM kept after the 3D view was closed** (found by attaching to the
live server, 11.3 GB): the 3D view never touched the card -- three.js and
Plotly draw in the browser. Two things did. (1) The LOD3 city was loaded
twice (merged + unmerged, ~5 GB): the scene cache had been cleared (page
switch, `release_scene_memory`) and the scene loaded again, while the old
objects stayed referenced by the module dicts of two earlier runs of the
page, which `_release_vram` -- defined in the page script and passed as
`on_change` -- carried in its `__globals__` from the WidgetMetadata of
widgets no longer drawn (the old *3D preview* checkbox). Callbacks now
live in `position_selector/state.py` (`release_vram`); a test refuses
`on_change=_page_function` in the page. (2) 6 GB of free pages in Dr.Jit's
allocator after the run's ray casts (roof candidates, occupancy, radio
maps; `drjit.whos`: 5.4 of 11.46 GiB used): `release_vram()` now runs at
the end of every run of the page, not only on a tab switch.

Opening a saved scenario on another scene: ~25 s on Kazan LOD3, of which
~11 s is `raycast_roof_candidates` (1014 buildings sampled), ~3 s reading
the JSON files, ~2 s the scene loads, ~2 s the frames. The files get a
progress bar per file (`load_scenario_files(on_file=)`), the roofs one per
25 buildings in a slot above the tabs (the call sits in the Manual tab,
invisible from the Scenario tab). The roof result is memoised in the
process (`roof._ROOFS`, 8 entries) instead of `st.cache_data`: a cached
function writing to a placeholder made outside it is a
`CacheReplayClosureError`.

See [per-plugin data contracts](PLUGIN_PIPELINE.md) and the [Pages architecture](../pages/docs/ARCHITECTURE.md).

## Directories

- `output_data/plugins/traffic/` — traffic files from the Position
  Selector's Traffic tab, listed by the CIR-Generator's moving-car
  multiselect. A file is a population of vehicles on its own clock, so it
  is input to a run rather than a product of one.
- `output_data/` — pipeline outputs. `.gitignore` keeps the folder skeleton via
  `.gitkeep`; only `processed/` and `dataset/` carry committed gitkeeps. Plugin
  subfolders are created at runtime. **Not the only root** — see below.
  `experiment_ir/` is the merged VNA measurement library (name-encoded
  metadata, hard-linked to the `experiment_ir_2026` /
  `experiment_ir_duplex_2026` archives); it is input, not output.
- `configs/` — `page_categories.yaml` (nav), `plotly/` & `plots/` (chart themes
  for the visualization pages), `sweep/` (CIR-Generator parameter-sweep
  presets), `material_scattering.json` (the CIR-Generator's diffuse-scattering
  tables), `parked_cars.json` (its kerbside-parking layouts,
  `STREAMFLEX_PARKED_CARS_PRESETS` moves the file).
- `sionna-scenes/` — Sionna ray-tracing scene assets (`.xml`/`.blend`/`_tx_locs.npy`).
- `snapshots/` — saved session pickles.
- `.cache/` — `cache_result` pickle cache (`Category^hash.ext`). `.tmp/` — scratch.
- `logs/` — Loguru output rendered in the in-app log console.

## Storage roots

The box has a second 2 TB SSD at `/mnt/data`, so a plugin that touches the disk
picks *where* through `st.segmented_control`. There are **two independent
choices**, both in `utils/storage.py`, both rendered via `create_widget` so they persist
and snapshot like any other widget:

| Control | Constant | `Project` | `Data disk` |
|---|---|---|---|
| `storage_root_control` — results | `STORAGE_ROOTS` | `output_data/` | `/mnt/data/streamflex/output_data/` |
| `scratch_root_control` — `.tmp`/`.cache` | `SCRATCH_ROOTS` | `.` | `/mnt/data/streamflex/` |

They are separate on purpose: scratch is write-heavy and disposable, so
parking it on the big disk while results stay in the project (or the reverse)
is a reasonable thing to want.

Only the *root* moves. Everything under it — `plugins/<plugin>/<timestamp>/`,
`first_component/`, `resultsaver/`, `.tmp/`, `.cache/` — is laid out identically
on both disks, so the offline pages read a run the same way once pointed at it.

Who offers what:
- **SignalChannelizer**, **OptiReceiver** — both controls. OptiReceiver's
  results choice covers the convolution/wavelet/peak run folder *and* the
  manual first-component bundle.
- **ResultsSaverPlugin** — results only (it has no scratch of its own).
- **CodeGenerator**, **CIR-Generator** — scratch only (they write no results,
  just the `cache_result` pickle cache).

Things that follow from that:
- Both choices travel to loky workers **inside the per-config payload**
  (`output_folder`, `temp_dir`, `cache_dir`), not through module-level
  constants — a re-imported plugin module in a worker has no Streamlit
  session. Payloads that predate the choice fall back to
  `FALLBACK_TMP_DIR`/`FALLBACK_CACHE_DIR`. For code that cannot reach a
  payload there is `STREAMFLEX_STORAGE_ROOT` / `STREAMFLEX_SCRATCH_ROOT`, the
  same escape hatch `STREAMFLEX_TIMESTAMP` uses for the run timestamp.
- `storage_root_status`/`scratch_root_status` refuse the data disk unless
  something is actually **mounted** at `/mnt/data` and fall back to the
  project root. An unmounted mount point is an ordinary empty directory on the
  system drive, so files written into it silently fill `/` instead of the SSD.
- `cache_result(reset, cache_dir=...)` resolves its folder **at call time**;
  `generate_cache_filename(cache_dir, func, ...)` hashes only the call, never
  the directory, so moving the cache does not change what a call keys to (it
  just starts cold on the new disk).
- ResultsSaver's streaming cleanup fence is `PLUGIN_OUTPUT_ROOTS` — one
  `plugins/` directory **per storage root**. An upstream stage may have
  written its run to either disk and its files must stay reclaimable, while
  everything outside those directories is still refused. `streamflex_app.py`
  has the same shape: its sizes, its cache clearing and its plugin-output
  panel all iterate `all_tmp_dirs()` / `all_cache_dirs()` / `STORAGE_ROOTS`
  rather than a hard-coded folder — and clear with `sum()`, not `any()`,
  which would short-circuit and leave the second root untouched.
- The ResultsSaver path text inputs hold a full path, so switching the control
  rebases them (`rebase_storage_path`) — paths under a known root move,
  hand-typed paths are left alone. Rewriting a persisted widget value takes
  all three steps (WidgetManager, the `current_value` cache, the Streamlit
  key), the same dance the hardware presets do.
- The offline `pages/` still read `output_data/...` unconditionally; only the
  plugins can be redirected so far.

## Tests

`tests/` is an AppTest-driven suite over the real plugin chain
(`tests/apps/chain_app.py` + `tests/testlib.py`); `pages/tests/` holds the
offline geometry tests. The vehicle ones are all in the second group and
deliberately so — a traffic plan is a walk-network dict and numpy, so
`test_traffic.py`, `test_parked_cars.py` and `test_moving_cars_channel.py`
need no Mitsuba, no GPU and no city. The last of those builds the channel
with `object.__new__` and sets only the traffic attributes, which is what
keeps "when are the cars looked at" testable apart from "what does the
solver return". Two things bite when a plugin changes:

- **A channel built with `__new__` is a contract too.**
  `tests/test_duplex_streamlit.py` constructs a `RayTracingChannel` with
  `RayTracingChannel.__new__` and sets exactly the attributes `__call__`
  reads — no scene load, no solver, no `__init__`. So anything new that
  `__call__` touches has to be read with a `getattr` default, the way
  `trace_diagnostics` and `valid_path_mask` already are. Adding the
  traffic attributes without that took 16 of those 22 tests down on
  `'RayTracingChannel' object has no attribute 'traffic_plan'`, none of
  which have anything to do with cars.
- **Widget names are the contract.** `testlib`'s `*_ws()` helpers write
  the serialized widget state directly, so renaming a widget silently
  turns a test into "run with defaults" rather than failing loudly. The
  TX/RX chains are per station (`tx_base_*` / `tx_mobile_*`,
  `rx_imp_base_*` / `rx_imp_mobile_*`) and the forward path is
  base→mobile, so TX seeds go to `base` and RX seeds to `mobile`.
- **Measured scans are addressed by name, and the name must be in the
  library.** The `measurement_files` multiselect drops any saved value
  that is not in its current options, so a test (or a run on another
  machine) points `STREAMFLEX_EXPERIMENT_ROOT` at its own folder and
  writes synthetic scans there — `cir_experiment_ws` in `testlib` seeds
  the rest. This is also why moving the real files changed which entries
  a stale snapshot has selected.
- **Hardware presets overwrite seeded values.** Each plugin copies its
  selected preset into the widget state once and records that in an
  `_applied_*` marker. `chain_app.py` selects the *ideal* devices (the app
  itself defaults to a real SDR) and pre-seeds those markers, so a test's
  explicit impairments survive. `"hardware_presets": "app_defaults"` in
  the spec opts out; a dict of state-key → preset name picks specific
  devices.

## Conventions worth keeping

- When adding a CIR channel model, extend the `channel_models` registry in
  `plugins/CIR-Generator/registry.py` — do not add a generic discovery layer.
- New plugin widgets must go through `self.create_widget(...)`, otherwise state
  won't persist or snapshot.
- A typed input that only feeds an explicit action (a run/build/train
  button, a plugin's generate pill) gets `on_change="ignore"` (Streamlit
  ≥1.64: slider, number_input, select_slider, selectbox, text_input), so
  editing it does not rerun the page or plugin and redraw every figure.
  Never on a widget something on screen reads before that action — a
  caption, a preview, another widget's options or `disabled`. In the
  plugins the pending edits land on the next rerun, i.e. **Refresh**; the
  CIR-Generator does this for every typed model parameter
  (`LIVE_PARAMETER_WIDGETS` / `"live": True` opt out), and
  `create_widget` skips its app-scope rerun for such widgets so the click
  that committed them is not lost. Downloads use `on_click="ignore"` and a
  callable `data=`, so nothing is serialised until clicked.
- `st.cache_*` leaves `_`-prefixed arguments out of the key: an `_mtime`
  meant to invalidate a cache does nothing. Whole-file page loads go
  through `utils.read_parquet_cached`.
- When you change the dataset row schema in the peaks/first-component workers,
  **old Parquets become incompatible and must be rebuilt** — downstream readers
  assume the current columns.
- Keep `data_manager` payloads in the `{config_info, results}` shape with `re/im`
  split complex arrays so the next stage and ResultsSaver can consume them.
