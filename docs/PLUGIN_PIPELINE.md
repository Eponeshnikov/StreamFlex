# StreamFlex plugin pipeline

The [core architecture](ARCHITECTURE.md) describes plugin loading, state and the shared data bus. This reference records individual plugin contracts.

- **CodeGenerator** (`gold_codes` / `m_sequences` / `kasami_codes`) — generates
  spreading codes. Three generators (Gold / M-sequence / Kasami); polynomials
  chosen by Auto / preferred-pair / manual-binary / batch-CSV; index ranges
  selectable; output format binary / field / bipolar / decimal.

- **IQGenerator** (codes → `iq_generator`) — modulates code sequences into IQ
  symbols. PAM or QAM, configurable `bits_per_symbol` and constellation rotation;
  length set by target bit-length or by repetition count (codes tiled/truncated).
  Shows constellation + I/Q signal preview.

- **Pulse-Shaping** (`iq_generator` → `pulse_shaping`) — upsamples and applies a
  pulse-shaping filter (Sionna). Filters: RRC, RC, Rectangle, Sinc; configurable
  span, samples/symbol, roll-off `beta`, optional window (Hann/Hamming/Blackman).
  Stores filter energy/output shape; visualizes filter response + ACF.

- **CIR-Generator** (→ `cir_generator`) — generates channel impulse responses
  (delays `tau` + complex gains `a`) for several channel models: Sionna **Ray
  Tracing** (scene-based, TX/RX placement, building selection), **TDL** (38.901),
  and an **Experimental measured-JSON** model. Model registry lives in
  `plugins/CIR-Generator/config.py` (`channel_models`, `BASE_CIR_GEN_PARAMS`,
  visualizers), loaded dynamically. `config.py` is a facade: the code lives in
  sibling modules (`registry`, `models`, `params_rt`, `rt_channel`,
  `visualizers`, `viz_*`, `scene_*`, `measured`, `cars_ui`, `encoding`,
  `scattering`) loaded with `plugin.load_sibling` — by path, outside
  `sys.modules`, so cloudpickle ships them to loky workers by value; cached per
  load of the calling module, so `plugin.py`'s per-rerun module values stay
  fresh. The same pattern splits every plugin's `plugin.py`. UI is config-driven: each param dict declares
  its widget, single/list mode, dtype, and optional dynamic `options_callback`;
  list params expand via `itertools.product`. Emits `scene_metadata` (TX/RX
  coords, last axis size 7) — the carrier that gets geometry into the keygen
  Parquet. Independent source; does not consume the IQ chain.

  **Device batching wants the largest batch that fits, not a round
  number.** `RayTracingChannel._batched_cir` splits the TX×RX product into
  chunks and solves each separately, and a path solve carries a large
  *fixed* cost that the batch size does not divide: Dr.Jit re-renders the
  whole megakernel to PTX on every `dr.eval` (it must — the kernel cache is
  keyed on a hash of the generated IR), which on a city scene runs into
  minutes on one CPU core with the GPU at 0%. Halving the batch therefore
  doubles the run. `plan_device_batches` sizes it from what each axis
  actually costs:
  - the candidate buffer is `max_num_paths_per_src × num_sources`
    (`sb_candidate_generator`), so it constrains the **TX** batch and the
    transmit antennas, and is blind to how many receivers ride along;
  - receivers instead *dilute* that fixed buffer — every target shares it,
    which is what `target_candidate_capacity_per_rx` pins, and a target is
    an RX **antenna** unless the array is synthetic;
  - the returned CIR is the full `[rx, rx_ant, tx, tx_ant, paths, time]`
    tensor, so the antenna counts and time steps cost host memory without
    ever appearing in the device count.

  Paths-per-link is the one term nothing can know up front, so the first
  batch uses `INITIAL_PATHS_PER_LINK` and every later one re-plans from the
  paths already returned — the same two-stage sizing peaks processing uses
  on its window rows. Budgets are measured **once per solve**
  (`_memory_budgets`): Dr.Jit's caching allocator keeps holding VRAM
  between batches, so re-measuring would shrink every plan after the first.
  `max_tx_per_batch` / `max_rx_per_batch` are optional ceilings on top of
  that, `0` meaning none — they used to default to a flat 128, which on a
  3000-point trajectory is 24 solves where memory allowed 4.

  **What kind of ray each path was is stored with it, because nothing can
  reconstruct it later.** `Paths.interactions` lives on the solver object,
  and that object is killed with the child process the moment the CIR comes
  back — so a parquet either carries the interaction chain or the question
  is unanswerable for that run. `path_interaction_signature`
  (`private_scene_preview_2.py`) packs the ordered `InteractionType` codes
  of a ray one per nibble into a single `int64`: depth d at bits 4d..4d+3,
  `0` = NONE, so a chain of `0` is LoS and the count of non-zero nibbles is
  the ray's own bounce depth. That is lossless to 15 bounces (past which the
  chain is dropped and only the depth kept — a truncated chain would be a
  wrong answer, not a partial one) and it fits a parquet column, which a
  `[..., paths, depth]` array does not.

  The two arrays — `path_interaction_chain` / `path_interaction_depth` —
  are shaped **exactly like `tau`**, so a path index addresses the same ray
  in all of them, and they ride through every transform `tau` takes: the
  per-link compaction in `RayTracingChannel.__call__`, the rx→time swap
  (`_swap_path_annotations`), the double-sided batch duplication. Padded
  slots read as LoS, so `valid_path_mask` is what separates them, exactly as
  it is for the delays. `interaction_encoding` travels next to them: a
  reader decodes from the file, never from its own copy of the code table,
  because a parquet written before a code existed is indistinguishable from
  one written after. The reverse duplex side is *not* re-derived —
  `interactions` is a property of the `Paths` object and always in the
  forward device order, so the signature is read once and the caller does
  not transpose it back the way it does for a/tau. Consumed by
  `pages/private_cir_ray_analysis.py`.

  **The CFR and the taps are a matrix product, not a broadcast.**
  `get_cfr`/`get_taps` form `Σ_p a[..., p, t] · kernel(τ)[..., p, k]`, and
  writing that as `a[..., p, t, 1] * kernel[..., p, 1, k]` followed by a
  reduce materialises `[links, paths, time, k]`: on a 100-RX city solve at
  15 659 paths, 14 time steps and 1000 subcarriers that is **327 GiB** of
  complex128 for a 22 MiB answer, and the combo dies with an allocation
  error before anything is stored. `_sum_over_paths` contracts the path axis
  with `einsum`, so time and k never meet it, and builds the kernel one
  block of receivers and paths at a time (`plan_axis_chunks`, capped at
  `PATH_SUM_BLOCK_BYTES` — the work is the same at any block size, so there
  is nothing to buy by filling memory). `pages/tests/test_path_sum_blocking.py`
  pins it against the formula it replaced at three block sizes: a blocked
  reduction that drops a block is invisible in the output's shape. The TDL
  duplex path in `plugin.py` still uses the plain broadcast — its path count
  is a handful of taps, so the product stays small there.

  **Diffuse scattering is a material property, not a solver flag** — every
  material Sionna ships carries `scattering_coefficient=0`, so *Comp
  Diffuse* on its own returns no diffuse paths. The tables live in
  `configs/material_scattering.json` (`STREAMFLEX_MATERIAL_SCATTERING_PRESETS`
  moves the file) and the widget is a **multiselect of preset names**: one
  name per combo in list mode, several merged left to right in single mode,
  so a `"*"` preset can be laid down first and individual materials
  corrected on top. It used to be a text box holding a dict literal declared
  `dtype: "str"` — and a dict is not a `str`, so parsing its own default
  `"{}"` raised `Expected a string, got <class 'dict'>` and killed *every*
  ray-tracing run before the solver (the parse error now names the
  parameter, and `tests/test_plugin_cir_generator.py` checks every text-box
  default against its declared dtype). `_parse_material_scattering` still
  accepts the literal, so older snapshots and parquets resolve unchanged.
  Cached CIRs key on the preset *name*, so editing a table in place needs a
  cache reset.

  The table lives on the **materials**, not in the solver call, so it has to
  be written before *every* solve: `compute_paths` applies it, not only
  `_ensure_scene_validity`. And anything that rebuilds a channel from a
  combo's stored `parameters` goes through `channel_init_params` — filtering
  on `RayTracingChannel.__init__` alone drops the params the plugin *wrapper*
  consumes (`material_scattering`, the JSON position overrides), which is how
  the 3D view and Power Analysis came to solve at S = 0 and show no diffuse
  rays while the generation run had them.

  **Ray tracing runs in a child process.** Dr.Jit's allocator never returns
  device memory to the driver by itself — freeing every buffer and calling
  `dr.flush_malloc_cache()` leaves the process holding all of it (the pages
  sit in CUDA's stream-ordered pool until trimmed; `release_gpu_memory`
  does that now, see *VRAM* below) — so the simplest thing that frees a
  finished solve is the process exiting.
  `utils.submit_gpu_job` sends the work to a loky child and kills the worker
  once the result is in; `pages/rt_solver_worker.py` holds the callables,
  under `pages/` for the same reason the other `*_worker.py` modules are
  (a cold child has to import them by qualified name, and the plugin's
  `config.py` is loaded through `importlib` under a synthetic name — so the
  worker reaches `channel_models` by *path*). The registry entry carries
  `"gpu_subprocess": True`; TDL and the measured models stay in-process,
  since they never touch Dr.Jit and would only pay the start-up.
  Cold start is 0.8 s total on an LOD3 city (0.5 s for `sionna.rt` plus the
  CUDA context, 0.27 s for the scene), against minutes for the solve.
  `generate_cir_wrapper` keeps its name and signature on purpose:
  `generate_cache_filename` hashes `func.__name__`, so entries already in
  `.cache/` stay valid.

  **Three** call sites solve, not one, and missing any of them keeps the card
  occupied exactly as before. Besides `generate_cir_wrapper` there is
  `additional_method_wrapper`, which builds a *second* channel per combo for
  the registry's `additional_methods` — and `get_cfr`/`get_taps` call
  `compute_paths()` internally, so those are full solves too, not lookups.
  The Power Analysis tab reached into an
  `st.cache_resource` channel and called `_check_paths()`, i.e. ran a **full
  path solve inside Streamlit** the first time it was opened — one click
  undoing everything moving the generation out had bought.
  `_get_path_power_data` is now a `submit_gpu_job` call returning finished
  numpy arrays.

  **Streamlit runs the body of every tab on every rerun**, so a visualizer
  is not opt-in just because its tab is closed — `visualize_cir` calls every
  `additional_methods` visualizer unconditionally. That is how a run served
  entirely from `cache_result`, with no solve at all, still ended up holding
  4.2 GiB: `viz_3d_scene` rebuilt the channel each rerun (a live scene is
  **892 MiB** of VRAM on LOD3 Kazan, measured, and Dr.Jit never gives it
  back) and re-solved in process when *show paths* was on, while
  `viz_power_analysis` started a solve of its own. Both now sit behind an
  explicit toggle, default off, the way the Position Selector gates its 3D
  preview.

  Switching the 3D toggle off has to *end a process* to give the memory
  back — dropping the `st.cache_resource` entry would not have: a scene
  loaded, deleted and followed by `flush_malloc_cache()` still holds all
  892 MiB. So the 3D view runs against a **keep-warm child**:
  `submit_gpu_job(warm_group=...)` leaves the worker alive between calls so
  an interactive view is not reloading an LOD3 scene per slider move
  (measured: first render 4.8 s, repeat 0.02 s), a warm claim stops one-shot
  jobs from reclaiming the worker out from under it, and
  `release_gpu_worker` ends it and drops the figures it cached. The parent
  never loads the scene at all — it stays at the bare CUDA context
  throughout. `rt_solver_worker.scene_figure` assembles the whole figure
  there, because `render_sionna_scene_plotly` takes the live Mitsuba `scene`
  and `Paths`; only a finished Plotly figure crosses back. One consequence:
  the scene-layer multiselect is gone from this view (it is a Streamlit
  widget and there is no Streamlit in the child) — no loss, since the ray
  tracer loads with `merge_shapes=True` and layers only exist unmerged.
  The in-parent `_reconstruct_channel`/`_get_cached_channel` were deleted
  rather than left unused: they are exactly how the leak would come back.

  What is left is the **420 MiB** CUDA context, which never comes back once
  anything in the process touches the GPU.

  Anything imported first by a cold child must not pull Mitsuba's
  variant-specific types before `sionna.rt` selects a variant. Both
  `config.py` and `private_scene_preview_2.py` did (`from mitsuba import
  Point3f`) and only worked because something else had already imported
  `sionna.rt`; both now `import sionna.rt` above it.

  **Parked cars are placed at solve time, not written into the scene.**
  A city out of the OSM builder has empty streets, and a kerbside row of
  cars is a specular reflector at exactly receiver height — measured on
  the reference geometry, one car between a TX and an RX 16 m apart adds
  its own arrival 10 ns after the LoS. `pages/osm_scene/cars/` plans
  the layout and `RayTracingChannel._place_parked_cars` inserts it, so
  every solve gets it: the params live on `RayTracingChannel.__init__`,
  which is what `channel_init_params` filters on, so the 3D view and
  Power Analysis rebuild the same streets rather than empty ones.
  - **The road network comes from the `_osm.json` sidecar**, the same one
    the Position Selector's Path Generator walks on — centrelines with
    class, width and the `bridge`/`tunnel` flags, in scene metres.
    Nothing in a written Mitsuba scene can say where a street is (meshes
    are merged per material), so a scene without the sidecar gets no
    cars and the preview says so instead of guessing. `munich`, `etoile`
    and the other shipped scenes are therefore out.
  - **A kerb carries two numbers, not one.** Whether a car may stand
    there at all (`class_permission`) is separate from how full the kerb
    then is (`class_density`), because they answer different questions
    and folding them into one density is what lined every arterial in a
    render with cars. The permission half is **measured**: OSM states it
    per side (`parking:left`/`:right`/`:both`, plus the superseded
    `parking:lane:*`), and a 2×4 km box over the centre of Kazan — 1957
    vehicle ways, 490 tagged, 878 tagged sides — gives P(allowed | the
    side is tagged) of `service` 0.58, `residential` 0.38,
    `unclassified` 0.18, `tertiary` 0.14, `primary` 0.08, `secondary`
    **0.03**, links 0.00. The table this replaced carried secondary at
    0.35 and tertiary at 0.55. Measured on the `kazan_center` sidecar,
    arterials now take 40 cars of 2564 (1.6%) and `service` takes 1658
    (65%) — which is where the cars in a photograph of the place are.
    **A guessed class is not believed like a tagged one.**
    `derive_roads` has no OSM classes — it infers one from the corridor
    width — so those priors, which are about the *tags*, do not transfer:
    a wide corridor in the old town of Munich is far more likely a street
    people park on than the arterial its width suggests. Taking them at
    face value cut munich from ~2700 parked cars to 765, almost all of
    the loss on corridors guessed `secondary`. A derived class is
    therefore blended 0.6 of the way towards `TYPICAL_PARKING_PRIOR`
    (0.46 — the same box's class priors weighted by how many ways carry
    each), which puts munich at ~1560. A class set to *zero* stays zero
    however it was arrived at, so a preset saying "no parking on
    motorways" is not blended back up into a third of a kerb.
    A way that carries the tag itself **overrides the prior in both
    directions**, and its statement also overrides the both-sides coin;
    `separate` counts as no, because that parking is mapped as its own
    area and a kerbside row would be it counted twice. `service` is 1315
    of those 1957 ways and its subtypes park nothing like each other, so
    a courtyard aisle, a `driveway` and a `parking_aisle` are no longer
    one class. Tables are **preset names** from
    `configs/parked_cars.json`, merged left to right with a `"*"`
    wildcard, exactly like the material-scattering presets and for the
    same reason: a text box holding a dict literal is the bug that killed
    every ray-tracing run once already. The first preset has to be the
    no-op one — an empty multiselect falls back to its first option.
  - **The row is parked against the buildings, not at the traffic lane.**
    A sidecar carries one `width` per way and that width is the
    carriageway, so a car planned from it alone stands at the edge of the
    lane with the pavement and often ten metres of open ground between it
    and the building it is supposed to be parked against. The gap is not
    in OSM at all, so it is measured off the scene's own geometry —
    `derive_roads.annotate_side_clearance`, the probe a derived network
    already uses — and written back as the per-vertex `side_clearance`
    the planner reads. A derived network measured it on the way in and is
    unchanged at the default `hug_fraction` of 1.0 — measured on munich,
    parallel cars with no margin reproduce the documented 2.83 m median
    to the nearest facade with 10% beyond 4 m. The preview offers the
    same measurement behind a button, because a preview that shows the
    row at the kerb while the run parks it at the wall is worse than no
    preview, and it is memoised per scene per process: the measurement is
    a downward ray per square metre of city and a sweep rebuilds the
    channel once per combo. `facade_margin_m` (0.3 m, half the raster
    cell) is a nod to the resolution of that measurement rather than a
    fix — on munich it costs 4% of the cars and takes the number with a
    corner in a cell the obstacle mask calls blocked from 4 to 3, and
    those last few are at the limit of what the check can tell, since the
    mask marks a whole square metre blocked from one ray at its centre.
  - **Orientation is drawn per kerb, not per car**: parallel 0.78,
    perpendicular 0.14, diagonal 0.08, from the 158 sides in that box
    that state one. A bay is pitched by the car's *width* rather than its
    length and turns the body across the kerb, so it is four times the
    metal per metre and broadside to the street — a different reflector,
    not a denser one. A street too narrow to hold one falls back to a
    parallel row rather than losing the parking.
  - **Sides are drawn per way, not per car.** `both_sides_prob` decides
    whether a street is parked on both kerbs or one (which one is its own
    draw); a real street is consistent along its length. Each way draws
    from `default_rng([seed, way_index])`, so raising the cap or editing
    a zone across town does not reshuffle the street being looked at.
  - **Zones are multiplicative and clipped**, drawn as a box or lasso on
    the preview map: 0 clears an area, 0.5 halves it, *Only inside zones*
    turns them into the allow-list. Multiplicative means overlapping
    zones compose the same way whatever order they were drawn in — the
    one thing someone dragging boxes cannot be asked to track.
  - **A car is dropped, never moved**, when it lands within
    `clearance_m` of a device: on a trajectory run the RX list *is* the
    time axis, so nudging a receiver out of a car body would move a
    sample in time. That also fixes the order — cars go in after
    `_run_radiomap_and_rx_sampling`, so the sampling radio map does not
    see them, which costs nothing (it picks RX spots, it is not a
    result).
  - **The ground ray is cast from below**, because the first hit from
    underneath is the carriageway while a ray from above stops on a tree
    crown, a balcony or the bridge deck the street runs under. Cars *on*
    a bridge are the exception and are cast from above — on Kazan that is
    27 cars of 4000, each of them 5 m out otherwise. The cast goes
    against the whole scene rather than a ground-only sub-scene:
    `mi.load_dict` **renames any shape handed to it**, so building one
    rewrites the id of the scene's ground mesh and the next `scene.edit`
    dies on an assertion.
  - **One merged shape, `parked_cars`**, of `sionna.rt.scene.low_poly_car`
    instanced per car (20 triangles each; 4000 cars is 80k against a
    2.2M-triangle city). Merged because the solver and the Plotly
    renderer both pay per shape, and nothing downstream needs to address
    one car — what reaches the CIR is which material a ray hit. The
    material is its own `ITURadioMaterial` metal, so the scattering
    presets can address it by name.
  - **The preview is 2D and opt-in.** The popover's body runs on every
    rerun whether it is open or not — Streamlit has no "is this showing",
    the same as for a tab — and a whole city of car footprints is 0.46 MB
    of Plotly JSON, so the map sits behind a default-off toggle the way
    the 3D view and Power Analysis tabs do. The counts and the per-class
    table cost nothing and are always shown. Roads are drawn one trace
    per density bucket and every car in one batched trace, coordinates
    rounded to centimetres: a trace per way is the same payload mistake
    as a Scatter3d per ray segment.
  - **Three sources of street geometry, in order.** The sidecar; then,
    for a scene the OSM builder did not write, the geometry itself; then
    nothing, and `parked_cars_info` says which one ran and why it placed
    none. `pages/osm_scene/derive_roads.py` is the middle one and it is
    the Position Selector's own walkable-space rule taken one step
    further: cast a ray down per cell, call what stands up an obstacle,
    hull the buildings to cut the empty apron, keep the largest connected
    component of the free space — that component *is* the street network,
    which is why the courtyards fall out of it as closed zones — and then
    thin it to its medial axis and trace that into centrelines. Measured
    on Sionna's `munich` (2367 named buildings and one flat `mesh-ground`,
    no roads at all, no georeference to fetch real ones against): 332
    streets, 3267 cars, 1 s. What it cannot recover is stated where it is
    used — there are no OSM classes in geometry, so a class is **guessed
    from the corridor width**, which is what keeps the shipped presets
    working; nothing knows about access, one-way or surface; and open
    space wider than `MAX_HALF_WIDTH_M` is a square, not a street to run
    a row of cars down.
    - **The crossing number, not the neighbour count**, decides what a
      junction is. A thinned skeleton runs diagonally in staircases and
      the pixel at a step touches three others while still being the
      middle of one line: counted naively, two thirds of munich comes out
      as junctions, every street is chopped into 20 m pieces, and the
      parking clearance then clears both ends of each piece. 10098
      "junctions" become 305 real ones.
    - **Streets are stitched through their crossings** by angle and width
      (`merge_through_junctions`), so a street is one way rather than a
      piece per block — and the crossings ride in the network's own
      `junctions` list instead. `plan_cars` prefers that list when it is
      there, and then stops trimming the *way ends*, which are dead ends
      or the scene edge rather than junctions.
    - **Each car is placed from its own kerb, measured per side**
      (`side_clearance`): a ray cast out of every centreline vertex, both
      ways, at a small fan of angles, projected back onto the normal. One
      width per street cannot do it — it has to be the width the
      *tightest* point can hold, so every car on the wide stretches ends
      up out towards the middle of the road; and the isotropic distance
      transform cannot either, because it answers with the *nearest*
      obstacle in any direction, so a street built up on one side only
      puts its cars in the open ground opposite. Measured on munich, the
      median car sat 5.0 m from the nearest facade with 69% of them more
      than 4 m out; per side it is 2.8 m and 90% within 4. **A side with
      no facade is not a kerb** and gets no cars.
      Three things keep a car's corners out of the walls, and all three
      are needed: the clearance is a **rolling minimum over the car's own
      length** (measured at a point, a car is a 4.4 m box and its corners
      reach the doorway recess first), the fan catches walls that run at
      an angle, and each angled ray is **capped at `PROBE_REACH_FACTOR` ×
      the car's length** — past its own nose the ray is asking about a
      facade the car never reaches, and letting that constrain the offset
      pushes the row back to the middle again (uncapped: median 3.0 m and
      12% beyond 4 m, capped: 2.8 m and 9%). Measured end state: 2720
      cars, **zero** of 10880 corners inside a building.
      The pavement taken off each side is 1 m, deliberately tight: the
      point of a kerbside row is that it is at the kerb, and a generous
      pavement is indistinguishable from parking down the middle.
      Pinned by `test_derived_cars_never_end_up_inside_a_building` and
      `test_derived_cars_park_against_the_facades_not_down_the_middle`.
      None of this touches a sidecar network: an OSM way carries one
      width and that width is already the carriageway, so it keeps the
      plain `width / 2` offset.
    - The derivation needs the scene *loaded*, so the preview runs it in
      a `submit_gpu_job` child behind a button and caches the polylines —
      the parent never loads a scene for this, and a live LOD3 city is
      892 MiB of VRAM Dr.Jit never gives back.
  - **The OSM builder can bake them in instead** (*Parked cars*, off by
    default), and that is the better home for a static layout: it places
    them from the walk network it has just collected, so the classes and
    widths are **OSM's own** rather than inferred, and the z comes from
    the ground mesh's own interpolation (`surface_fn`) rather than a ray
    cast. They go out as one `parked_cars` shape with a `vehicle`
    material role, counted in the build report and declared under
    `synthetic_geometry` next to the windows and the scattered trees —
    OSM maps no parked cars anywhere, so they are invention and the
    sidecar says so.
    The plugin recognises that shape by name (`cars.PARKED_CARS_SHAPE`):
    with no preset selected it leaves it alone (the scene is what it is),
    with one it replaces it rather than parking a second row on top, and
    a no-op preset strips it — which is the empty-street baseline an
    ablation against a baked scene needs. *Replace cars already in the
    scene* turns the last two off.
  - **`pages/osm_scene/__init__.py` exports the builder lazily** (PEP
    562). Importing it pulls earcut, the roofs and the whole mesh stack —
    0.65 s — and the two importers that matter here never build anything:
    the ray-tracing child wants `cars` (67 ms now) and the Position
    Selector wants `walkmap`.
  - **Cars off must hash as it did before the feature existed.** The
    default selection is the `no_cars` preset, not an empty list, so the
    plugin asks `parked_cars_requested` and strips the whole
    `PARKED_CAR_PARAMS` group out of the init params when nothing would
    be placed — otherwise `cache_result` keys change and every
    ray-tracing entry already in `.cache/` is orphaned by the upgrade.

  **Moving cars are the half of the traffic that makes the channel
  change.** Parked cars make a scene complicated; they do not make it
  *dynamic*. A van crossing the line of sight is the dominant scatterer
  appearing and disappearing, and at 2.8 GHz a car at 14 m/s carries
  130 Hz of Doppler on every ray that touches it.
  `pages/osm_scene/traffic/` plans the vehicles, the Position
  Selector's **Traffic** tab generates and previews them, and the
  CIR-Generator loads the file next to its positions.
  - **Speed is set as the network's mean, not a table's.**
    `target_mean_speed_ms` weights the per-class free-flow speeds by how
    much road each class carries, so it is the speed of the *traffic*: a
    network that is mostly courtyard aisles is slow however fast its one
    arterial is. It is free-flow — what a driver wants — and what it
    achieves is lower by however much the traffic is in its own way (a
    5 m/s target comes out at 4.0 on the Kazan network, a 12 m/s one at
    7.3), which is the mean the run reports afterwards.
  - **The dynamics are car-following, not a speed per car.** Each driver
    runs the Intelligent Driver Model against whatever is in front of it,
    so queues, jams and stop-and-go waves *emerge* from density rather
    than being scripted — measured on the Kazan network over five
    minutes, mean speed falls 5.72 → 2.33 m/s from 60 to 700 cars, with
    nothing stopped for a whole minute at 60 and 4.9% at 700. Drawing an independent random speed per car — the obvious
    thing — puts vehicles through each other within two seconds. A
    blockage is a car with a *late release*, which is what actually
    produces a city queue; raw density mostly produces slow traffic.
  - **A route is a closed circuit, not a trip.** A trip is planned to a
    destination, every trip to the same destination on the same
    shortest-path tree, and a city full of them funnels onto the same
    arterials. Measured on munich with **a hundred cars on ninety
    kilometres of road** — 1.3 cars per kilometre, a city with nothing
    in it — 5% of the lanes carried 38% of every traversal and 92 of the
    100 cars went down one of them, which is why they queued at a
    density at which they should never have met. `find_circuits` grows
    cycles in the lane graph instead: the first sweep prefers lanes no
    circuit has taken, so the loops spread over the city, and later
    sweeps may reuse one, because strictly edge-disjoint cycles cover
    less than half the network (munich 43%, against 76% with reuse). A
    circuit has three properties a trip does not. It never ends, so
    nothing is stranded however long the run — a trip is planned to
    `speed_max × duration` and a long enough run outlives it. Its cars
    can be spaced evenly round it, so the run starts at the density it
    would otherwise have to settle into. And it spreads: the busiest 5%
    of lanes carry 20% of the traversals against 37% for trips.
    `route_mode="trips"` is the old behaviour, kept because a trip is
    the right model for a car that is going somewhere.
  - **A junction is a set of car-sized cells, not one mutex.** The
    conflict points are found exactly — munich has 125 027 distinct
    crossing lane pairs — and a union-find then merged every point
    within 8 m of another *transitively*, which collapsed a whole
    junction, and on a dense city several of them, into a single
    reservation: **367 of them for 1742 lanes**. One car crossing
    therefore excluded every other movement there, including the ones
    its own path never meets. The resolution thrown away ranks exactly
    like the gridlock in the runs — san_francisco 1646×, etoile 987×,
    munich 341×, kazan 33×, and those are the scenes in the order they
    seized up. A resource is now the *place* where paths cross, binned
    at `CONFLICT_CELL_M`, and every lane through it shares it. Both
    halves matter: keying on the lane *pair* instead is the opposite
    mistake, since a third lane through the same point shares no key
    with either and drives through both. At 3 m a cell is coarser than
    the 2.8 m between the two lanes of one street, so the two directions
    do share cells at a crossing; halving it to 1.5 m separates them and
    was measured to be worth 5% of the mean speed for 40% more runtime,
    which is why it is not halved.
  - **A cell is handed back as the car passes it**, not when the whole
    crossing is behind it. Holding the lot until the rear bumper leaves
    makes the hold time the *traversal* time, so slower traffic holds
    junctions longer, which makes traffic slower still — the feedback
    that turns a busy network into a stopped one rather than a slow one.
    Only the cells still in front are re-asked for; asking again for the
    ones already released would freeze the car in the middle of the
    junction it is halfway through.
  - **An odd lane is kept, not halved away.** ``total // 2`` lanes per
    direction discards it, and on a real network that is most of the
    arterials: measured on the kazan sidecar, `secondary` and `primary`
    have a median carriageway of **10.5 m**, which is three lanes, and
    the halving built two. Keeping it takes kazan from 62.1 to 69.0 km
    of lane and the mean speed up 3.5% on every seed tried. The extra
    lane goes with the way's own direction, the convention a single
    shared corridor already follows, so the pieces of one street stay
    consistent with each other.
  - **A car claims a crossing well before its braking distance**
    (`CLAIM_REACH_M`/`_S`, 30 m or 4 s). Claiming at the old six metres
    meant braking almost to a stop at the line to find out whether it
    could go, and with a stop line every 71 m that is the whole speed:
    measured on kazan at 300 cars, 3.84 → 4.14 m/s on every seed, and on
    an empty network 19.7 → 24.4 km/h. Past about 40 m it reverses, because
    claiming early is also holding early.
  - **Pulling away is the most expensive number in the model.** With a
    stop line every 71 m almost all of the time goes on accelerating
    again, so `IDM_ACCEL` sets the speed more than anything else does:
    1.2 → 2.0 m/s² is worth 18% of the mean speed on kazan and 13% on
    munich, on every seed, with no collisions either way. The textbook
    IDM value for a car is 1.0-1.5 and is calibrated for motorway
    comfort; 2.0 is 0-50 km/h in about seven seconds. It keeps paying
    above that (2.5 is worth another 5%) and stops being ordinary
    driving, which is why it stops at 2.0.
  - **The speed level is still two to four times below the traffic
    engineering tables, and five ways of fixing it measured worse.**
    On kazan the fundamental diagram comes out at 1.6 veh/km/lane →
    19.7 km/h and 12.9 → 5.8 km/h, where an urban table says 35-50 and
    20-30; flow tops out near **74 veh/h/lane** against a real 400-900.
    What was tried, all measured, all reverted: letting cars going the
    same way share a cell (1.0 km/h, because the keep-clear protection
    goes with it); vehicle-actuated **signals** (5.4 km/h — 93.9% of cars
    sit inside a 40 m detector at any time, so a phase gating a whole
    axis stops far more traffic than a reservation gating only the
    movements that actually cross: 66.9% of cars see green against 90.9%
    of claims granted); **major-road priority** without gap acceptance
    (3.65 km/h — the major car enters an occupied crossing and the clamp
    turns that into a standoff); and dropping `service` roads from the
    through network (11.9 against 13.9). `junction_control="signals"`
    is kept as an option and is not the default.
    **Gap acceptance was then built and it does not pay either.** The
    minor movement waiting for a `CRITICAL_GAP_S` hole in the major one
    is safe -- no collisions, no clamps -- and still slower: 5.2 km/h
    against 6.2 on kazan at 800 cars, on every seed, whether or not the
    major movement is also offered the crossing first. The reason is the
    network rather than the rule: 642 of kazan's 944 vehicle ways are
    `service`, so the roads that would get priority are a hundred of
    them and prioritising that minority starves the majority, while the
    reservation already gives the crossing to whoever is ready. It is
    the right model for a city whose arterials carry the traffic, and
    these derived networks are not that.
  - **Neither the mutex nor the clamp can resolve a standoff, so the
    reservation has to prevent one.** With the junction control removed
    the emergency clamp — hold whoever would end up inside someone else
    — fires 841 705 times on a 700-car munich run and the network stops
    within sixty seconds: **167 pairs of cars stood nose to nose for the
    whole rest of the run and 261 of the 700 were locked in one**.
    Neither car can reverse and nothing breaks the tie, so a clamp is
    permanent. That is why the reservation is *sticky* and *atomic* — a
    car takes every cell its path needs before it enters, and holds them
    — and why the clamp is only the check that the reservation worked.
    SUMO's answer to the same problem is to teleport a vehicle that has
    waited too long, which is not available here: a car that vanishes
    mid-run takes every path that touched it with it.
  - **The three pairwise tests are neighbour-limited, not pairwise.**
    Who is in front, who is about to cross and who is inside whom are all
    O(N²) written the obvious way, in time *and* memory: at 2000 vehicles
    the crossing check alone wanted an `(N, N, samples, 2)` array of
    700 MB per integration step. Candidates come from a KD-tree instead,
    and `_RouteBundle` samples every car in one `searchsorted` rather
    than a Python call per car per step. Measured on the Kazan centre
    over 300 steps: 2000 cars in 4.4 s, 4000 in 12.4 s, with no single
    call over 1.3 s afterwards.
  - **Bodies, never circles.** A circle covering a 4.4 m car is 2.4 m in
    radius and two of them reach 4.8 m — wider than the 2.8 m between one
    lane and the next, so every car reads as in permanent conflict with
    whatever drives alongside it and two cars cannot even be *placed* in
    adjacent lanes. The conflict test, the placement test and the
    parked-car blocker all use the real footprints, and the standstill
    gap is added along the car, where a bumper-to-bumper gap is, not
    across it. On a 2000-car request that is 1660 placed at 2.14 m/s
    instead of 1419 at 0.93.
  - **A narrow corridor is one-way.** Two directions sharing one lane is
    the honest geometry and it is what jams the network: the crossing
    rule stops one of the pair, the other cannot pass, and they block
    everything behind them — 332 pairs ended a 2000-car run nose to nose
    with 92% of the traffic stationary. It only became possible once
    `split_at_junctions` existed, and the direction follows the way's own
    vertex order rather than a coin: the pieces of one street keep that
    order, and flipping each piece independently chops the street into
    segments that meet head to head (2.34 m/s against 2.09).
  - **No accidents, and it is checked rather than assumed.** Three things
    were each necessary. The closing rate is the relative velocity
    projected on the car's own heading — as a difference of two speeds an
    oncoming car at 16 m/s reads as 0 m/s and the two drive through each
    other. Crossing conflicts compare the two predicted tracks sample by
    sample *in time* (two routes that cross are not a conflict if one car
    is through before the other arrives), with priority to whoever is
    nearer the meeting point and **never** to a car that is already
    stopped — a stopped car cannot yield any harder, and handing it the
    yield leaves the moving one to drive into the back of the queue.
    Under both, the step is *proposed*: anything that would end up inside
    something else does not move, which is clear by induction because the
    previous state was. `overlapping_pairs` re-derives every overlap from
    the finished arrays; over 15 configurations from 60 to 700 cars it is
    zero.
  - **The lanes come out of what the parked cars left**, per side
    (`parked_free_sides`), so a street parked down one kerb has its
    traffic shifted across rather than squeezed from both, and one too
    narrow for a full lane is still driven if a car fits at all —
    requiring 2.8 m took 494 of 944 vehicle ways out of the network. A
    parked car is also a *leader that never moves*, so traffic queues
    around one rather than through it.
  - **Ways are cut at the nodes they share** (`split_at_junctions`). A
    sidecar stores OSM ways, and an OSM way is not split at every
    junction: a side street ends on the *middle* of a main road. Lanes
    joined end-to-start therefore could not see a single T-junction —
    447 of 1050 lanes had nowhere to go and the median route was 165 m,
    twenty seconds before a car ran out of road. Cutting took that to
    314 m; preferring turns that lead somewhere (`forward_reach`) halved
    the truncations again.
  - **The geometry is rebuilt for every instant, and that is a solve per
    walked point.** A receiver *is* an instant in trajectory mode, so the
    traffic that belongs to it is the traffic at that instant and nothing
    else. This used to group points into *geometry frames* — refresh the
    cars only once one had moved further than a settable `Geometry step`,
    capped by `Max geometry frames` — which bought a few dozen solves
    instead of a few thousand. It was removed: the delay argument for it
    is sound (a car that moves less than the run's delay resolution, 37 cm
    at 800 MHz, cannot move a delay) but blockage is *binary*, and the
    quantisation moved the point at which a van cut the line of sight,
    which is the event the whole feature exists to capture.
    **The price is measured and it is the point count**: on munich at
    depth 3 a solve costs 0.25 s whether it covers 1 receiver or 400 — the
    cost is the fixed Dr.Jit megakernel rebuild, which is exactly why
    batching was worth so much — so a 3000-point walk is 3000 × one
    solve, minutes on munich and days on an LOD3 city. The lever is the
    number of walked points.
  - **One run is split across processes, because a solve leaves the
    machine idle.** The walk *is* the RX axis, so a window of that axis is
    a stretch of time and an independent solve:
    `rt_solver_worker.solve_cir_shard` gives each child a window (it
    builds the *whole* channel, so the receiver list, the traffic clock
    and the device order stay the run's own — only the loop is narrowed,
    and the cars are placed at the **global** receiver index), and
    `config.merge_cir_shards` concatenates the windows back along the RX
    axis. The one thing that cannot be assumed there is the padded
    **path** slot count: it is per solve, so a window that found more
    paths comes back wider and everything shaped like `tau` — the mask,
    the interaction chain, the depth — is padded to the widest before it
    is joined, with the conventions the rest of the pipeline reads (−1,
    False, 0). `duplex_metadata` and `trace_diagnostics` are taken from
    the first window rather than merged: they describe the run, every
    child built the same channel, and the motion table is complete in
    each because the scene holds every receiver in every child.
    **Checked by equality, not by inspection**: on munich with traffic,
    2, 4 and 8 processes reproduce the serial `a`, `tau`,
    `scene_metadata`, the valid-path mask and the interaction chains,
    duplex included (96 points, two sides, 3.06× on four processes) —
    bit for bit on the smaller runs, and on a 200-point one within
    **1.5e-7 of peak amplitude and 0.3 ps of delay**, which is what the
    *same serial solve repeated twice in one process* also differs by
    (measured: 102 of 2400 entries, max 1.54e-7). The path solver
    accumulates on the GPU and is not bit-deterministic, so exact
    equality is a property of small runs, not of the split.
    `Tracing processes` is a plugin-level widget and not a model
    parameter, because it changes how long a run takes and never what it
    returns — it must stay out of `cache_result`'s hash.
    **The windows are handed out, not dealt.** They are not equal in cost
    — a receiver in the open sees far more paths than one in a courtyard
    — so one window per worker ends the run with a single process
    grinding out an unlucky share while the rest sit idle, which is what
    a falling core count and a falling rate look like near the end.
    `submit_gpu_batch` keeps one job in flight per worker and gives the
    next window to whoever finishes first, and the plugin cuts
    `SHARD_WINDOWS_PER_PROCESS` windows per worker. Measured on a cost
    profile that rises along the walk, four workers reach 49% of a
    perfect split at one window each, 63% at two, 73% at four — and on a
    *level* walk (munich, 160 points) sixteen windows came out 8% slower
    than four, because each window pays its own batch re-plan and VRAM
    probe. Two is that trade-off, not a guess.
  - **A run keeps what it finished.** The whole-config `cache_result`
    entry is written only at the end, so a run that died (a reattached
    browser, a stop, a crashed child) used to start the next one from
    zero. The worker (`_solve_window`) now writes every block it solves to
    `generate_cir_wrapper^<hash>.parts/rx<a>-<b>.pkl` — the directory is
    named after the entry's own cache file (`checkpoint_dir_for`), so it
    is keyed by everything the entry is, cleared with its category and by
    *Reset Cache*, and removed once the entry is written — and reads it
    back instead of solving. A walk whose traffic follows the receivers is
    a solve per point anyway, so it is cut into blocks of
    `max(64, n_rx/64)` receivers on **walk-wide** edges (a run split into
    a different number of windows still finds them); every other run
    batches receivers into one solve and is checkpointed per window only.
    The window count follows the `Tracing processes` widget, not the
    VRAM-capped slot count, for the same reason. Progress is reported per
    window in link-passes, and blocks read back are sent as `skipped` so
    the bar leaves them out of the rate. Checked on munich with traffic
    (50 points, blocks of 20): a run killed after the first block resumed
    with only the other two solved, and matched an uninterrupted run as
    closely as two uninterrupted runs match each other (0.7 % per-link
    amplitude, 0.7 ps).
  - **`get_cfr` / `get_taps` reduce the CIR the run already produced.**
    They take a `cir=(a, tau)` and only regenerate one when it is
    missing — and regenerating means a *second full solve of the whole
    run*, per combination, which on a trajectory with moving cars is a
    second solve per walked point. The warm channel's `_cir_memo` hid
    that whenever the call landed in the same process with identical
    arguments, and hid it not at all once a run was **split across
    processes**: each child holds one window of the walk, so nothing
    there is the run these ask for. Measured on a 200-point duplex walk:
    the CIR takes 19.5 s, `get_cfr` after a split run takes **17.0 s
    regenerating and 0.2 s from the stored arrays** — and the two CFRs
    agree to 1.7e-7 against a peak of 3.3e-4, which is the float32 level
    of the reduction. This is also the phase that used to leave the page
    on "Starting…" with an empty bar after the tracing bar had finished.
  - **A run says how far it has got, because it is now a solve per
    point.** The solve happens in a loky child with no Streamlit, so the
    batch loop writes `{done, total, elapsed_s, rate_s}` to a file
    (`_report_solve_progress`, temp file + `os.replace`, the shape
    OptiReceiver's per-config bars already use) and the plugin submits
    the job **in the background** and polls it, drawing the bar on the
    script thread. Blocking on `background=False` is what left a
    trajectory run sitting on "Starting…" for its whole length with one
    core busy and nothing to read. `progress_path` is an argument of
    `rt_solver_worker.solve_cir` and *not* of the channel: it belongs to
    a run, not to a model, so it must not reach `cache_result`'s hash.
    Until the first report lands the bar says what that wait is — a cold
    child loading the scene and building the first kernel, which is tens
    of seconds on a city.
    **What it counts is the run, not a pass.** One sweep of the receiver
    list is not the run: a duplex solve traces each side in its own
    sweep and a traffic snapshot traces every instant, so counted per
    sweep the bar fills up and starts again — which reads as the run
    going *backwards*, once per side and once per instant.
    `_begin_solve_progress` declares the passes before the first one
    starts (`__call__` for the duplex pair, `_traffic_snapshot_cir` for
    the instants, and a snapshot's inner calls leave the open scope
    alone), `_batched_cir` adds its pass to `_progress_base` when it
    ends, and the reported `done` is cumulative. The count is also
    *counted* rather than derived from the loop variables: `ti` steps by
    the TX batch size, so the old `ti · len(rx)` skipped a batch's worth
    of links whenever the TX batch was wider than one. Across shards the
    plugin sums those cumulative counts, and it says "reverse side" only
    once **every** window is on it — the windows are independent solves
    and hand over at their own pace, so the maximum announced the second
    half of the run while most of the first was still being traced.
  - **Which mode applies is a fact about the run, not a setting.** When
    the RX list *is* the time axis (`RX axis → time axis`), receiver r is
    the instant r·dt and the traffic is moved to it: this rides on the
    existing RX batch loop, with the batch cut to **one receiver** (a
    batch is a stretch of time and one scene holds the traffic of one
    instant), which is why traffic following the walk forces the batched
    path whatever the batching setting says (the unbatched solve does the
    whole RX axis at once, and the whole axis is one geometry).
    Otherwise the cars stand at one instant with the
    Doppler carrying the motion, which is *exact* for the windows this
    normally runs: 128 samples at 800 MHz is 160 ns, over which a car
    moves two microns. `_check_frozen_traffic_window` says so when the
    window is long enough for that to stop being true — measured against
    the run rather than a setting: a car that moves further than one
    sample of path length (`c/fs`) **or** than its own width (blockage is
    binary) over the window is a car the frozen geometry misplaces. A
    slow sweep has a coarse delay resolution and a window seconds long,
    and it is the width that catches it.
  - **Velocity is a property of the shape, so cars are grouped by it.**
    Sionna stores a velocity as a texture attribute on the mesh, and the
    solver pays per shape, so velocities are snapped to a lattice
    (`MOVING_CAR_VELOCITY_STEP_MS`, 0.5 m/s = 4.7 Hz at 2.8 GHz) and the
    cars that land on the same node share a shape — the trick
    `POWER_OPACITY_LEVELS` plays on ray opacity. Over the cap the lattice
    *coarsens*; dropping the rare groups would silently take those cars'
    Doppler to zero while leaving them in the scene.
  - **The duplex reverse side sees the traffic Δt later**, because it
    does: a van at 14 m/s is 14 cm further on at 10 ms, a sixth of a
    wavelength at 2.8 GHz. `moving_cars_duplex_shift` off is the
    perfectly-reciprocal baseline.
  - **Several files merge into one population** on one clock (finest
    step, the span they all cover), and a car that would ever touch one
    already accepted is **dropped whole** — two files generated over the
    same streets never saw each other and do drive through each other
    (measured: nine overlapping pairs from a 40-car and a 30-car plan).
    Re-simulating is not an option: it would move cars away from the
    positions their own file already committed to. To run files as
    separate configurations instead, the parameter goes to **list mode**,
    which sweeps one combo per file the way every other list parameter
    does.
  - **The 3D view draws the traffic itself, because the view never
    calls the code that places it.** `compute_paths` places the cars only
    for a *frozen* run; a run that follows the RX walk moves them per
    batch inside `_batched_cir`, which the visualizer does not go
    through — so a traffic run's 3D scene had no cars in it at all.
    `rt_solver_worker.moving_car_geometry` returns the car triangles at
    the instants being shown (the plan, plus one downward ray per car),
    and the view adds them as its own trace. **Not** by putting them in
    the scene: the Plotly backdrop is cached on the scene's shape list
    (`scene_content_key`), so a frame of cars in the scene re-welds the
    whole city per frame and evicts the entry it just filled. The
    animation ships the triangle table once and swaps only the vertices
    (a plan's car count is constant, which
    `test_every_traffic_frame_has_the_same_triangle_table` pins), so a
    frame of 150 cars is 47 KB.
    **The rays are re-solved per frame** (*Re-solve the rays for each
    frame*, default on): a car is geometry, so rays drawn against a frame
    that moved it are rays of a scene that no longer exists — measured on
    munich, a car standing between a 1.6 m TX and a 1.5 m RX 24 m apart
    takes the link from 4 paths to 2. The cars go into the scene for the
    solve and come **out before the backdrop is built**, because
    `scene_mesh_parts` keys its weld on the scene's shape list and a
    frame of cars left in re-welds the city per frame. Which instant the
    *cached* solve belongs to is tracked on the channel
    (`_view_traffic_time`), so a rerun that changes nothing about the
    rays reuses it, and frames are additionally kept in session state
    under a key covering everything a frame is made of —
    `submit_gpu_job` keeps four results, which for a 30-frame animation
    of solves is no cache at all. The loop has its own progress bar with
    seconds per frame and time left, estimated over the frames actually
    built rather than the cached ones, and naming the frame *in flight*
    rather than the last one finished. Turning the switch off is the old
    behaviour: one solve, cars drawn on top of rays that pass through
    them, and the caption says so.
    **The frames are solved in parallel, because one solve leaves this
    machine idle.** A path solve is host-bound and single-threaded — the
    GPU sits at 0–5% while one core rebuilds the Dr.Jit megakernel, the
    same finding as the `py-spy` PTX note — so throughput comes from more
    processes, not from a faster solve. `utils.submit_gpu_batch` runs a
    group's frames across its own single-worker executors, one scene per
    child, frames pinned to a child by index so a worker answers from the
    scene it already has. **Not** `get_reusable_executor(max_workers=N)`:
    that is loky's singleton, shared with the heavy plugins, and resizing
    it shuts down the child holding the warm scene. Measured on LOD3
    Kazan at depth 5 with diffraction: 18 frames take 31.5 s on one
    process and 7.1 s on six (**4.4×**), 12 frames take 5.7 s on four.
    Each child holds **1794 MiB** of VRAM once it has solved, so
    `gpu_worker_slots` caps the count against what is free and
    `release_gpu_worker` kills them with the rest of the view — measured,
    the card comes back to the bare context.
    What does *not* help, and was measured before being built: solving
    only the receiver being drawn. A solve costs the same for 1 receiver
    as for 3000 (1.7 s vs 1.6 s on Kazan with diffraction), because the
    cost is the fixed per-solve work. Diffraction is the multiplier that
    matters — 0.10 s → 1.9 s on the same scene, 15–20×.
  - **A 3D axis range clips the meshes inside it**, so whatever the view
    is fitted to is also what gets *drawn*. *Fit view to* offers the
    scene's own bounding box (default, read from
    `rt_solver_worker.scene_bounds` — the scene cannot be wrong about its
    own extent) or a box around the devices and their rays, which zooms
    in on a walk in a big city and cuts the rest of the city away. The
    tight fit is the one that was there before, unnamed.
  - **What ended up in the scene is a filterable column.**
    `vehicle_summary()` reports counts, sources, whether the Doppler was
    on and the reason when nothing was placed; it rides in
    `trace_diagnostics` (the one channel-side dict that survives the trip
    back from a solve in a child process) and the plugin copies it onto
    the combo's **parameters** as `vehicles`. That is the half of a row
    the offline pages build filters from, and a sweep with and without
    traffic is only readable if "with traffic" is something you can
    filter on. The struct's scalar leaves are picked up by
    `filter_specs.discover_param_filters` with no registry entry; the
    list fields (file names, preset names) are correctly skipped.

  **The measured scans** (`Experimental JSON`) live in one folder,
  `output_data/experiment_ir/`, and the **file name is the index** —
  `<band>_<scenario>_<pos>_<dirs>_<power>dBm_<cols>x<rows>_<date>.json`,
  parsed by `plugins/CIR-Generator/experiment_data.py` (that folder's README
  has the scheme). There is no manifest to rebuild: drop a file in and it
  shows up, and an unrecognised name still loads through the legacy parser
  or straight from its contents. `experiment_ir_2026/` and
  `experiment_ir_duplex_2026/` are the archive of the original layout —
  the merged files are *hard links*, so nothing is stored twice and old
  relative paths out of snapshots still resolve (`resolve_path`).
  `STREAMFLEX_EXPERIMENT_ROOT` moves the library elsewhere (a second disk,
  a test fixture) and is read per call, not at import.

  **These scans are an indoor bench check on real hardware**, not a
  measured counterpart of the ray-traced city — no hardware exists at
  that scale, so the RT track stays simulation-only and the measured one
  validates it on a table. The scanned patch is 1.23 × 1.17 m (42×40 at
  30 mm), and a scene is irrelevant to it.

  **The scan's `x`/`y` are positioner millimetres** and everything else —
  the TX/RX knobs, `scene_metadata`, every distance downstream — is
  metres, so `load_measurement` converts once (`POSITIONER_UNIT_M`) and
  nothing after it sees millimetres. Left unconverted the patch read as a
  1.2 km field with the TX standing inside it. Parquets from measured
  runs written before 2026-08-28 carry the millimetres in `rx_x`/`rx_y`
  and have to be regenerated.

  Two measurement generations, one loader. Old scans hold one `S21AVG`
  record per point; the 2026-07 duplex scans hold **two records per point**
  (`S21AVG` then `S12AVG` at the same `x`/`y`) plus a `noise_results`
  block. `load_measurement` keys records by `(x, y)` and splits them by
  direction — concatenating `results`, which is what the plugin did before
  those scans existed, turned 1680 points into 3360 pseudo-positions whose
  second half was the reverse link mislabelled as new geometry. Three
  knobs follow from that: *Reverse link* (measured S12 / simulated AWGN /
  auto — 5.8 GHz exists only one-way, so it still simulates — / none),
  *Receiver noise* (SNR knob or the scan's own measured floor), and side
  1's `scene_metadata` swapping TX and RX when the reverse link is real,
  the same convention the ray tracer's duplex sides use.

  **A duplex scan is duplex data, so it emits both sides on its own.**
  `double_sided_cir` ("Emit both sides") used to gate this, and with it
  off — its default — a `…_s21s12_…` file selected with *Reverse link =
  Auto* loaded S12 and threw it away: one batch, no reverse direction, no
  way to tell from the UI. Two sides now follow from
  `measured_files_have_reverse` (the file name already says so; no file is
  opened for a parsed one), the checkbox is left as the way to get a
  *simulated* second side out of an S21-only scan, and *Reverse link =
  None* is the explicit opt-out. `double_sided_cir` is rewritten to what
  was actually emitted, because readers that predate `duplex_metadata`
  split sides on that flag alone.

  The batch axis is then a duplex axis and says so the way the ray tracer
  does: `get_duplex_metadata()` returns `side_order = ["forward_ab",
  "reverse_ba"]` (`physical_reverse_channel` only when side 1 is a second
  *measurement*), which is what labels the visualizers' batch selector
  *TX → RX* / *RX → TX* and what `private_peaks_processing` reads to split
  a run into sides at all. It carries **two measured times, and only one
  of them is the channel's**: `duplex_interval_s` is the median S21→S12
  gap *within* a point (0.63 s on the 2026-07 scans) — the experiment's
  real reciprocity interval, the counterpart of the ray tracer's Δt knob —
  while `scan_interval_s` is the step from one scanned point to the next
  (4.4 s), which is a VNA sweep plus a positioner move.

  With *Position Source = JSON File* the RX list is a **path over the
  scan**: the Position Selector's Grid path takes its lattice from a scan
  and exports each point's `(col, row)`, and matching column/row counts
  map the walk onto measured points by index, so a path drawn at scene
  scale still addresses the right measured point. Anything else falls
  back to nearest-coordinate matching (and says so). A path that revisits a point
  simply repeats that measurement. The segment's `grid.measurement` names
  the scan the lattice came from, so a path drawn on a different one is
  called out rather than silently downgraded to coordinates.

  **`RX axis → time axis` works here too**, and means the same thing: a
  walk over scanned points is a time order, not a list of receivers, so
  point r becomes `time_idx` r exactly as a ray-traced trajectory point
  does, and the JSON upload auto-enables it the same way. The transfer
  runs on different tensors — the sparse CIR is empty, so the walk lives
  on the taps / CFR, whose trailing axis is the IFFT sample, not time.
  `transfer_measured_rx_axis_to_time` swaps RX with axis 5 for those;
  running the sparse `transfer_rx_axis_to_time` over them would swap the
  walk onto the sample axis and shred the impulse response.

  **It needs no path.** Unlike the ray-traced models, whose RX list is
  only an order if a JSON says so, a scan *is* an ordered walk and every
  record carries a timestamp — so the toggle stays live at *Position
  Source = All measured points*, and the whole scan becomes the time axis
  in scan order. `get_time_axis()` is that clock, rebased to the first
  used point: `measured_time_s` (the instants themselves — the scan's
  clock is uneven, ~3.7–5.5 s per point on the 42×40 files) plus their
  median step, and a synthesized `trajectory_segments` entry so the
  visualizers have a walk to scrub. A path that revisits points makes the
  stamps go backwards; then `dt` is `None` (the median of the
  forward-only gaps is an artefact — five minutes on a scan measured
  every four seconds) and the walk is an order, not a clock.

  *Time axis Δt* picks the step: the trajectory JSON's own `dt` (default
  — what the walk was drawn to take, and absent when no path is loaded)
  or the measured cadence (real, ~4 s per point, i.e. a VNA sweep plus a
  positioner move: the whole 1680-point scan spans two hours). Neither is
  the duplex gap; that one is between the two *sides* of one point and
  rides in `duplex_metadata` either way.

- **SignalChannelizer** (`pulse_shaping` + `cir_generator` → `signal_channelizer`)
  — applies the CIR to the pulse-shaped signal (delay-and-sum), optionally adds
  AWGN at a chosen SNR(dB). Batched, device-selectable (auto/CPU/GPU). Large
  outputs spill to disk: `results["signal"]["data"]` is either in-memory `re/im`
  arrays or paths to `.pickle`/`.bin` memmap files (flagged by `save_file`).

  **The SNR contract** (all three fields per measurement, i.e. one entry per
  CIR, shaped `[B,R,RA,T,TA,TS]`, in *both* noise models):
  - `results["snr_db_per_measurement"]` — the SNR that measurement achieved.
    Under **Fixed SNR** the block sizes each measurement's AWGN from its own
    energy, so this is the configured constant. Under **Thermal Noise +
    Noise Figure** the kTBF variance is the knob and the SNR follows the path
    loss: an RT trajectory sweeps it from −110 to +35 dB, and blocked points
    are `-inf`. `parameters["snr"]` is its dB-domain mean — a summary, not a
    label; filter on the per-measurement column instead.
  - `results["noise_energy"]` — noise energy **summed over the stored
    window**. The bare per-sample kTBF variance the thermal path reported
    before 2026-08-26 overstated every per-ray SNR by `10·log10(N)`.

  The thermal model's `B` is **the sample rate, not a widget**. The AWGN is
  one i.i.d. complex sample per output sample, so it is white across the whole
  Nyquist band and `kTB` is that band's noise power only when `B = fs`; any
  other `B` is a hidden gain on the noise floor (`10log10(B/fs)`) that also
  breaks the tie between this SNR and Eb/N0. Move the floor with the noise
  figure instead. Two knobs genuinely do nothing to SNR: **Bandwidth
  estimation** is metadata plus hardware-preset warnings, and **RX gain** sits
  in both `thermal_noise_variance` (as `NF + G`) and the waveform scale, so it
  cancels exactly.
  - `results["input_signal_energy"]` — the energy a *raw* CIR amplitude
    multiplies: `|a|² · input_signal_energy` is a path's received energy.
    **Not `pulse_shaping.energy`.** The plugin puts the waveform through
    `apply_tx_impairments`, and a duplex run with per-side transmit powers
    renormalises it to unit RMS and folds `sqrt(10^((P_dBm−30)/10))` into the
    coefficients instead — so substituting the pulse energy, as the peaks
    pipeline did until 2026-08-26, costs `10log10(N_tx/E_ps) + (P_dBm−30)`,
    exactly 20.09 dB on the 44 dBm trajectory runs.

  These three are what makes the offline per-ray SNR add back up to the
  per-measurement one; `tests/test_snr_chain.py` pins that identity end to
  end.

  **The CFO and the thermal noise are applied outside the channel block**, to
  the array it returns — only Fixed-SNR noise is added inside it. So the
  memmap save path must write `final_output_tensor`, never copy `out_file`:
  the block's own file predates both. Copying it (every run before
  2026-08-27, i.e. every trajectory-sized run, since `use_memmap` is how they
  fit in RAM) shipped a noiseless, CFO-free signal to OptiReceiver while the
  metadata advertised the noise — peaks resolved perfectly at a reported
  −169 dB. Fixed-SNR runs were never affected, which is exactly why they
  still went blind around −30 dB. Parquets written before 2026-08-26 have none of them and fall back to
  the old approximations — peaks processing warns when it has to.

- **OptiReceiver** (`signal_channelizer` + `pulse_shaping` + `cir_generator` →
  `optimal_receiver`) — matched-filter receiver: convolves the signal with the
  reference pulse, optional wavelet denoising, then peak detection to estimate
  path delays. Detects LoS/first component (K-factor / optionally an ML
  classifier). Chunked + memmap for big data. **Also has a manual "Export
  first-component bundle" button** (`export_first_component_bundle`) that pickles
  the per-measurement signals + detected peaks + ground-truth CIR to
  `output_data/first_component/export_<ts>.pkl` (+ a `.pkl.store/` mmap dir) —
  this is the **sole feed** into the offline first-component pages below.
  "Save Results in File" persists an artifact into the run folder **only when
  the payload still references it** (`persist_conv`/`persist_wave_den`): the
  pre-wavelet convolution output is superseded by the denoised one, and
  `response` is dropped entirely unless "Signal" is in *Results for saving*.
  Anything written but unreferenced is unreclaimable — the saver's streaming
  cleanup only deletes paths the parquet points at.

  **It also decides what the "true" components are**
  (`results["visible_components"]`, `compute_visible_components`, core in
  `pages/component_visibility.py`). The ray tracer's path list is not it: with
  diffuse scattering at depth 5 it is ~11 600 paths per link (up to 70 000),
  and on the noiseless `results_rt_snr_2000` run that put the detection rate
  at **0.5 %** and wrote 374 M label-2 rows. Most of those paths are not
  missed, they are *unresolvable* — isochronous rays in one resolution cell
  are one peak however good the receiver is. So a component is defined from
  what reaches the matched filter, `r(t) = Σ a_i·g(t − τ_i)` with `g` the
  **actual reference's** autocorrelation (tabulated at 1/32-sample lags by a
  phase ramp on its power spectrum): a local maximum of `|r|` on the sample
  grid, whose strongest single contribution comes from a ray inside the main
  lobe (otherwise it is a stronger ray's sidelobe), standing above
  `κ·σ`, `σ² = ρ²·Σ|a_i|² + σ_n²/E_s`, `κ = √ln(1/P_fa)` — by **height and
  by prominence** (the dip to a stronger neighbour, Sparrow's criterion with
  the pedestal in place of zero); `margin_db` is the smaller of the two, with
  `height_margin_db`/`dip_margin_db` stored beside it. Without the dip test
  15 % of the munich "visible" set were shoulders on a stronger component's
  slope, found 55 % of the time — by chance; with it one munich config went
  from 8.5 components per measurement found 80 % of the time to 7.4 found
  89 %. `ρ²` is the
  reference's mean sidelobe power past the near kernel (−36.6 dB for Gold-11 /
  RRC 0.22 / sps 4) and `σ_n²/E_s` is `noise_energy / N_out /
  input_signal_energy`, so **nothing is fitted**. Measured against what the
  receiver actually detected on that run (4 scenes × 500 links), P(detect)
  rises monotonically with the margin over `σ` and crosses 50 % at 5–10 dB in
  every scene; at the default `P_fa = 1e-3` (+8.4 dB) the truth is 8–12
  components per link and the receiver finds 85–93 % of them.
  Each component keeps its **dominant ray** (`ray_index`, and that ray's
  interaction chain), `n_rays` in its ±½-main-lobe cell and the dominant
  ray's share of the cell energy — 1.0 for a clean specular path, small for a
  speckle peak built out of diffuse rays. Stored padded `[B,Rx,RxA,Tx,TxA,T,C]`
  down to 6 dB *under* the floor so the pages can loosen the cut as well as
  tighten it. Rays are pruned before `r` is built only when their summed
  `|a|` cannot move `|r|` by 5 % of that storage floor anywhere (triangle
  inequality; `test_pruning_never_changes_the_visible_set`). ~8 ms per link.
  The CIR-Generator's own `recognisability` (a logistic fit of detection on
  the energy share, noiseless, per ray) predates this and was **miscalibrated
  for diffuse runs**: 59 % of the rays the receiver *did* detect on
  `results_rt_snr_2000` had U ≥ 0.95, because a detected speckle peak is
  matched to one ray that carries a tiny share of the link's energy. It is
  still computed and still the fallback cut on per-ray datasets.

  **A receiver timing residual moves every detection off the CIR's
  delays**, and the truth has to move with it. The `field_gpsdo` sync
  preset draws a 20 ns residual per station per 1 s resync epoch plus a
  0.01 ppm clock drift (10 ns over an epoch), and `apply_rx_impairments`
  resamples each capture so a path at τ is detected at `τ - timing_error`.
  Components computed from the unshifted CIR then sit off every detection:
  on `results_rt_path_realistic*` the strongest peak coincided with a
  component 7.5% of the time and true peaks scored a median XGBoost
  confidence of 0.03. `apply_rx_impairments(return_timing_error=True)`
  now returns the per-capture error, the plugin stores it as
  `results["rx_timing_error_s"]` and computes the visible components on the
  shifted delays (`measurement_components(tau_shift=...)`). Δτ is immune
  (the error is common to every path of a capture) — only labels, counts
  and delay errors against truth were wrong. Datasets built before this
  are re-matched on load by the offline key generator
  (`ingest.realign_component_labels`, per resync epoch, offset
  `a + b·(t − t₀)`); per-ray truth in peaks processing does not read the
  stored error yet. The same absence of any timing impairment is what made
  `results_rt_path_more` look good: both ends detect on one sample grid
  (integer samples, no interpolation), 90% of its intervals have
  Δτ_A = Δτ_B exactly, and the two ends' errors correlate at 0.94.

- **ResultsSaverPlugin** (reads any key → Parquet) — flattens a plugin payload
  into a Polars Parquet (default `output_data/plugins/resultsaver/`), with a
  streaming mode for memmap-backed arrays and trigger-based autosave. **The
  bridge from the online pipeline to the offline pages.** Both outputs are
  written as *directories* of `batch_NNNN.parquet` (Polars reads a directory
  path transparently): the raw table is flattened `batch_size` configs at a
  time so peak RAM follows the batch, not the run — `rows_to_frame` turns every
  numpy array into Python lists, so one frame for a whole run costs tens of GB.
  The **Preview table** toggle only builds the first `preview_rows` configs and
  is never required for saving.

**Heavy-stage note:** SignalChannelizer and OptiReceiver can run their configs in
parallel via joblib `loky` (an `n_jobs` widget, default 1). Per-config work is in
module-level `_*_one_config` functions; processing blocks (`DelayAndSumBlock`,
`ConvolutionBlock`, `PeakFinderBlock`, `WaveletDenoisingBlock`) take a
`show_progress` flag because loky workers have no Streamlit context. They write
timestamped output under `output_data/plugins/<plugin>/<ts>/` and guard the
module-level timestamp via `STREAMFLEX_TIMESTAMP`.

**A parallel run still has to report progress, and `show_progress=False` is
exactly what takes that away.** With the blocks' own bars off in the workers,
OptiReceiver's only signal was "one more config finished" — and joblib's
`return_as="generator"` hands those over in *submission* order, so an early
finisher waits for everything dispatched before it. On a run with a handful of
long configs the bar therefore sat at 0% for tens of minutes and then jumped.
Three things fix it, all in `plugins/OptiReceiver/plugin.py`:
- `HostArrayBlock._progress_bar` decides where a chunk loop's bar goes:
  `st.progress` on the script thread, a `_SinkProgressBar` inside a worker,
  which writes the same fractions to `<temp_dir>/optireceiver_progress/<run>/
  cfg_<id>.json` (one file per config, temp file + `os.replace`). Coarse
  stages are marked by `_worker_phase` so the reported fraction spans the
  whole config, not each block's own 0→1.
- `_drain_outcomes` iterates the joblib generator on a helper thread and hands
  outcomes over a queue, so the script thread stays free to redraw the bar
  every 0.4 s from those files — a bar that only moves on completions is
  useless when there are fewer configs than workers.
- `_parallel_outcomes` prefers `return_as="generator_unordered"`. Completion
  order is then not submission order, so both branches collect results in a
  dict keyed by `config_id` and append them in id order once the run ends —
  the stored ids stay what the sequential path produces
  (`tests/test_plugin_opti_receiver.py::test_parallel_path_matches_sequential`).

The progress state is a **dict keyed by thread id**, not a `threading.local`:
this module is registered with `cloudpickle.register_pickle_by_value`, so every
module-level object is serialized with each task and a `_thread._local` cannot
be pickled at all — it breaks every parallel run at dispatch.

**VRAM is the shared resource these stages fight over**, and three separate
things hold it:

- **Sionna's `Block.__call__` uploads every array argument to the GPU**
  (`torch.as_tensor(v, device=config.device)`, and `config.device` is
  `cuda:0` whenever a card exists) — *before* `call()` runs, and independent
  of the plugin's own auto/CPU/GPU widget, which only sets `_device_pref`
  for the block's internal chunking. Every block here then copies it
  straight back with `.detach().cpu().numpy()` on its first statement, so
  the round trip is pure waste: measured, a 977 MiB input cost 978 MiB of
  VRAM and was discarded. It also lands *outside* `plan_axis_chunks`, so
  the careful chunk budget never sees it. `HostArrayBlock` (defined in both
  plugins, base of all five blocks) overrides `_convert` to let
  `np.ndarray` through untouched. This is what OOM'd nine OptiReceiver
  configs on 2026-09-07 — the traceback lands in `_convert`, not in any
  block.
- **Each loky worker holds a CUDA context of its own**, ~420 MiB, so
  `n_jobs=16` parks ~6.5 GiB before a single tensor. joblib keeps the pool
  warm and reaps it on an idle timeout of *minutes*, which is exactly when
  the next stage starts. `utils.release_worker_pool()` shuts it down at
  once; it reads loky's module-level singleton rather than calling
  `get_reusable_executor()`, which would *start* a pool to shut it down.
- **Sionna RT's Dr.Jit memory comes back only with a pool trim.** Dr.Jit
  (1.5) allocates with `cuMemAllocAsync`; `dr.flush_malloc_cache()` frees
  its cache with `cuMemFreeAsync` *into the device's stream-ordered pool*,
  and the pool keeps every page — which is why the flush alone measured as
  returning nothing (1.3.1 and 1.5 alike). `release_gpu_memory()` now
  follows the flush with `dr.sync_thread()` and `cuMemPoolTrimTo(pool, 0)`
  on the device's current and default pools (driver API through ctypes,
  only on CUDA contexts that already exist): measured 1 GiB freed → 1 GiB
  back, and an LOD3 scene (2.27 GiB) deleted and released → 1.9 GiB back.
  What is still *referenced* stays: a scene held by `load_scene_cached`,
  the solver of a keep-warm child. The Streamlit process traces in-process
  in one place, the Position Selector (radio maps, multipath grids, ray
  casts — 20 GB parked on 2026-10-23); those maps release after each
  compute (`radio_maps._releasing_vram`), the page releases on a tab switch
  and on the 3D preview switched off, and `app.py` calls
  `utils.release_scene_memory()` (drop every cached scene + release) when
  the user leaves a page. The child process is still the way to bound a
  solve's peak.

`streamflex_app.py` calls both helpers in a `finally` after **every** plugin,
so a failed or interrupted stage hands the card back too, and logs how much
came back.
