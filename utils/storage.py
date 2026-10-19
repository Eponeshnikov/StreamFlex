"""Storage and scratch roots: resolution, status and their sidebar controls."""

import os
from collections import OrderedDict

import streamlit as st

CACHE_FOLDER = ".cache"


# ---------------------------------------------------------------------------
# OUTPUT STORAGE ROOTS
# ---------------------------------------------------------------------------
# Every plugin that spills results to disk writes under a *root*, which used to
# be the repo-local ``output_data/`` unconditionally. The box also carries a
# second SSD, so the root is a per-plugin choice now. Only the root moves: the
# ``plugins/<plugin>/<timestamp>/`` layout underneath is untouched, so a run
# reads back the same way wherever it landed.

PROJECT_STORAGE_ROOT = "output_data"
DATA_DISK_MOUNT = "/mnt/data"
DATA_DISK_STORAGE_ROOT = os.path.join(
    DATA_DISK_MOUNT, "streamflex", "output_data"
)

# Label -> root. Insertion order is the order of the segmented control.
STORAGE_ROOTS: "OrderedDict[str, str]" = OrderedDict(
    (
        ("Project", PROJECT_STORAGE_ROOT),
        ("Data disk", DATA_DISK_STORAGE_ROOT),
    )
)
DEFAULT_STORAGE_LABEL = "Project"
# joblib (loky) workers have no Streamlit session. Plugins that cannot thread
# the root through their payload export it here before dispatching, the same
# trick STREAMFLEX_TIMESTAMP already uses for the run folder.
STORAGE_ROOT_ENV = "STREAMFLEX_STORAGE_ROOT"

# Scratch is a *separate* choice from results: the memmap spill (``.tmp/``)
# and the pickle cache (``.cache/``) are write-heavy and disposable, so it is
# reasonable to park them on the big disk while results stay in the project —
# or the other way round.
PROJECT_SCRATCH_ROOT = "."
DATA_DISK_SCRATCH_ROOT = os.path.join(DATA_DISK_MOUNT, "streamflex")
SCRATCH_ROOTS: "OrderedDict[str, str]" = OrderedDict(
    (
        ("Project", PROJECT_SCRATCH_ROOT),
        ("Data disk", DATA_DISK_SCRATCH_ROOT),
    )
)
DEFAULT_SCRATCH_LABEL = "Project"
SCRATCH_ROOT_ENV = "STREAMFLEX_SCRATCH_ROOT"
TMP_FOLDER = ".tmp"


def _root_status(root: str, project_root: str) -> tuple[bool, str]:
    """Shared usability check behind the storage and scratch roots.

    Anything off the project root lives on the data disk, which counts as
    usable only when something is actually mounted at :data:`DATA_DISK_MOUNT`.
    An unmounted mount point is an ordinary empty directory on the system
    drive, so files written into it would quietly fill ``/`` instead of the
    SSD.
    """
    if root == project_root:
        return True, ""
    if not os.path.ismount(DATA_DISK_MOUNT):
        return False, f"`{DATA_DISK_MOUNT}` is not mounted"
    # Probe the deepest part of the root that exists rather than creating it:
    # this runs on every rerun, and the writers already build their own
    # folders when they actually write.
    probe = os.path.abspath(root)
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    if not os.access(probe, os.W_OK):
        return False, f"`{probe}` is not writable by this user"
    return True, ""


def storage_root_status(label: str) -> tuple[bool, str]:
    """Report whether ``label``'s results root can be written to right now.

    Returns
    -------
    tuple[bool, str]
        ``(usable, reason)``; ``reason`` is empty when usable.
    """
    if label not in STORAGE_ROOTS:
        return False, f"unknown storage `{label}`"
    return _root_status(STORAGE_ROOTS[label], PROJECT_STORAGE_ROOT)


def scratch_root_status(label: str) -> tuple[bool, str]:
    """Report whether ``label``'s scratch root can be written to right now."""
    if label not in SCRATCH_ROOTS:
        return False, f"unknown scratch location `{label}`"
    return _root_status(SCRATCH_ROOTS[label], PROJECT_SCRATCH_ROOT)


def resolve_storage_root(label: str | None = None) -> str:
    """Resolve a storage label to a root path.

    ``None`` means "whatever the parent process chose" — read from
    :data:`STORAGE_ROOT_ENV`, which is how a loky worker recovers it. An
    unusable root falls back to the project root rather than writing into an
    unmounted stub.
    """
    if label is None:
        return os.environ.get(STORAGE_ROOT_ENV) or PROJECT_STORAGE_ROOT
    usable, _ = storage_root_status(label)
    return STORAGE_ROOTS[label] if usable else PROJECT_STORAGE_ROOT


def resolve_scratch_root(label: str | None = None) -> str:
    """Resolve a scratch label to a root path (see :func:`resolve_storage_root`)."""
    if label is None:
        return os.environ.get(SCRATCH_ROOT_ENV) or PROJECT_SCRATCH_ROOT
    usable, _ = scratch_root_status(label)
    return SCRATCH_ROOTS[label] if usable else PROJECT_SCRATCH_ROOT


def tmp_dir_for(scratch_root: str) -> str:
    """Memmap spill directory under ``scratch_root``."""
    return os.path.normpath(os.path.join(scratch_root, TMP_FOLDER))


def cache_dir_for(scratch_root: str) -> str:
    """``cache_result`` pickle cache under ``scratch_root``."""
    return os.path.normpath(os.path.join(scratch_root, CACHE_FOLDER))


def resolve_cache_dir(label: str | None = None) -> str:
    """Cache directory for a scratch label (default: what this process chose)."""
    return cache_dir_for(resolve_scratch_root(label))


def _usable_scratch_roots() -> list[str]:
    return [
        root
        for label, root in SCRATCH_ROOTS.items()
        if scratch_root_status(label)[0]
    ]


def all_tmp_dirs() -> list[str]:
    """Every ``.tmp`` across the usable scratch roots.

    The app's cache panel reports and clears through this: with the choice
    made per plugin, one session's scratch can be spread over both disks, and
    a single hard-coded folder would both under-report and under-delete.
    """
    return [tmp_dir_for(root) for root in _usable_scratch_roots()]


def all_cache_dirs() -> list[str]:
    """Every ``.cache`` across the usable scratch roots."""
    return [cache_dir_for(root) for root in _usable_scratch_roots()]


def rebase_storage_path(path: str, root: str) -> str:
    """Move ``path`` under ``root`` when it sits under some known root.

    Used by the path text inputs: switching the storage control should carry
    the pending output path over instead of stranding it on the old disk. A
    path outside every known root is user-chosen and left alone.
    """
    if not path:
        return path
    normalized = os.path.normpath(path)
    for known in STORAGE_ROOTS.values():
        known = os.path.normpath(known)
        if normalized == known:
            return root
        if normalized.startswith(known + os.sep):
            return os.path.join(root, os.path.relpath(normalized, known))
    return path


def _root_control(
    plugin,
    widget_manager,
    *,
    roots,
    default_label: str,
    status_fn,
    resolve_fn,
    fallback_root: str,
    widget_name: str,
    container,
    label: str,
    help_text: str | None,
    help_prefix: str,
) -> str:
    """Shared body of the storage and scratch segmented controls."""
    target = st if container is None else container
    if help_text is None:
        help_text = help_prefix + ", ".join(
            f"{name} = `{path}`" for name, path in roots.items()
        )
    choice = plugin.create_widget(
        widget_manager=widget_manager,
        widget_type=target.segmented_control,
        widget_name=widget_name,
        value_param="default",
        default_value=default_label,
        args=(label, list(roots)),
        kwargs={"help": help_text},
    )
    # A single-select segmented control returns None once the user deselects.
    if choice not in roots:
        choice = default_label
    usable, reason = status_fn(choice)
    if not usable:
        target.warning(
            f"`{choice}` is unavailable ({reason}) — using "
            f"`{fallback_root}` instead.",
            icon="⚠️",
        )
    return resolve_fn(choice)


def storage_root_control(
    plugin,
    widget_manager,
    *,
    widget_name: str = "storage_root",
    container=None,
    label: str = "Results storage",
    help_text: str | None = None,
) -> str:
    """Segmented control choosing where this plugin writes its results.

    Goes through ``plugin.create_widget`` so the choice persists across reruns
    and travels in snapshots. Returns the resolved root path, already fallen
    back to the project root if the chosen disk is not available.
    """
    return _root_control(
        plugin,
        widget_manager,
        roots=STORAGE_ROOTS,
        default_label=DEFAULT_STORAGE_LABEL,
        status_fn=storage_root_status,
        resolve_fn=resolve_storage_root,
        fallback_root=PROJECT_STORAGE_ROOT,
        widget_name=widget_name,
        container=container,
        label=label,
        help_text=help_text,
        help_prefix="Where this plugin writes results: ",
    )


def scratch_root_control(
    plugin,
    widget_manager,
    *,
    widget_name: str = "scratch_root",
    container=None,
    label: str = "Cache & scratch",
    help_text: str | None = None,
) -> str:
    """Segmented control choosing where ``.tmp``/``.cache`` live.

    Deliberately separate from :func:`storage_root_control`: scratch is
    write-heavy and disposable, so parking it on the big disk while results
    stay in the project (or the reverse) is a reasonable thing to want.

    Returns the resolved scratch *root* — pass it through
    :func:`tmp_dir_for` / :func:`cache_dir_for` for the actual folders.
    """
    return _root_control(
        plugin,
        widget_manager,
        roots=SCRATCH_ROOTS,
        default_label=DEFAULT_SCRATCH_LABEL,
        status_fn=scratch_root_status,
        resolve_fn=resolve_scratch_root,
        fallback_root=PROJECT_SCRATCH_ROOT,
        widget_name=widget_name,
        container=container,
        label=label,
        help_text=help_text,
        help_prefix=(
            "Where this plugin's memmap spill and pickle cache live: "
        ),
    )
