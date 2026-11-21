"""Grouping metadata for the scene selector.

``sionna-scenes/catalog.json`` says which scenes belong to a measurement
(an indoor room or an outdoor campaign) and which are standalone models.
It is presentation only: scene *paths* never change here, because a traced
CIR is cached under the scene path string, and renaming a scene would
orphan every entry already keyed by it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SCENES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sionna-scenes")
)
CATALOG_FILE = os.path.join(SCENES_DIR, "catalog.json")

#: Display order of the groups; anything unknown goes last.
GROUP_ORDER = ("measured/indoor", "measured/outdoor", "standalone")

GROUP_LABELS = {
    "measured/indoor": "Измерения · комната 1405",
    "measured/outdoor": "Измерения · улица",
    "standalone": "Самостоятельные сцены",
}

_CACHE: tuple[int, dict] | None = None


def _catalog() -> dict:
    global _CACHE
    try:
        stamp = os.stat(CATALOG_FILE).st_mtime_ns
    except OSError:
        return {}
    if _CACHE is not None and _CACHE[0] == stamp:
        return _CACHE[1]
    try:
        with open(CATALOG_FILE, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return {}
    by_path = {}
    for entry in payload.get("scenes") or []:
        if isinstance(entry, dict) and entry.get("path"):
            by_path[str(entry["path"]).replace(os.sep, "/").lstrip("./")] = entry
    _CACHE = (stamp, by_path)
    return by_path


def _key(name: str) -> str:
    text = str(name).replace(os.sep, "/")
    if os.path.isabs(text):
        text = os.path.relpath(text, SCENES_DIR).replace(os.sep, "/")
    text = text.lstrip("./")
    prefix = "sionna-scenes/"
    if text.startswith(prefix):
        text = text[len(prefix):]
    return text


def entry_for(scene_name: str) -> dict | None:
    """Catalog entry for an XML path, a display name or an absolute path."""
    key = _key(scene_name)
    catalog = _catalog()
    if key in catalog:
        return catalog[key]
    return next(
        (entry for entry in catalog.values() if entry.get("asset_path") == key),
        None,
    )


def discover_scene_files(directory: str | Path) -> dict[str, str]:
    """List selectable XMLs, keeping the stable path of migrated assets."""
    root = Path(directory)
    if not root.is_dir():
        return {}
    aliases = set()
    if os.path.abspath(root) == SCENES_DIR:
        aliases = {
            entry["asset_path"]
            for key, entry in _catalog().items()
            if entry.get("asset_path") and (root / key).is_file()
        }
    scenes = {}

    def add(path: Path) -> None:
        key = path.relative_to(root).as_posix()
        if key not in aliases and path.is_file():
            scenes[key] = str(path)

    for current, directories, files in os.walk(root):
        folder = Path(current)
        descendants = []
        for name in sorted(directories):
            if name.startswith(".") or name == "__pycache__":
                continue
            child = folder / name
            if child.is_symlink():
                if child.is_dir():
                    for item in sorted(child.glob("*.xml")):
                        add(item)
            else:
                descendants.append(name)
        directories[:] = descendants
        for name in sorted(files):
            if name.endswith(".xml"):
                add(folder / name)
    return scenes


def group_of(scene_name: str) -> str:
    entry = entry_for(scene_name)
    return str(entry.get("group") or "standalone") if entry else "standalone"


def group_label(group: str) -> str:
    return GROUP_LABELS.get(group, group)


def label_for(scene_name: str) -> str:
    """``group · display`` for a selector; the value itself is left alone."""
    entry = entry_for(scene_name)
    if not entry:
        return str(scene_name)
    return f"{group_label(entry.get('group') or 'standalone')} · {scene_name}"


def sorted_items(items: dict) -> dict:
    """Reorder ``{display: path}`` by group, leaving keys and paths intact."""
    order = {name: i for i, name in enumerate(GROUP_ORDER)}

    def sort_key(item):
        display, path = item
        group = group_of(path)
        return (order.get(group, len(order)), group, display)

    return dict(sorted(items.items(), key=sort_key))
