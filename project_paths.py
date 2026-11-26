"""Resolve former StreamFlex storage paths without filesystem aliases."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

PROJECT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _aliases() -> dict[Path, Path]:
    with (PROJECT / "configs/path_aliases.json").open(encoding="utf-8") as stream:
        entries = json.load(stream)
    research = PROJECT / "research/tools/artifact_layout.json"
    if research.is_file():
        with research.open(encoding="utf-8") as stream:
            layout = json.load(stream)
        entries = layout["migrated"] | layout["aliases"] | entries
    return {Path(old): PROJECT / new for old, new in entries.items()}


def resolve_project_path(path: str | Path) -> Path:
    """Return the current path for an absolute or project-relative name.

    Keep saved selection names and cache keys at the caller. Resolve here
    when accessing the filesystem. Paths outside this checkout are untouched.
    """
    candidate = Path(os.path.abspath(PROJECT / path))
    aliases = _aliases()
    visited: set[Path] = set()
    while candidate.is_relative_to(PROJECT):
        for ancestor in (candidate, *candidate.parents):
            key = ancestor.relative_to(PROJECT)
            target = aliases.get(key)
            if target is not None:
                if key in visited:
                    raise ValueError(f"Circular StreamFlex path: {path}")
                visited.add(key)
                candidate = target / candidate.relative_to(ancestor)
                break
            if ancestor == PROJECT:
                return candidate
    return candidate
