#!/usr/bin/env python3
"""Install the selected StreamFlex repositories and their locked dependencies."""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GITLAB = "https://gitlab.epohub.netcraze.pro/StreamFlex"
PAGES_BRANCH = "research/rtw8-fourway-20261117"
PAGES_CONFIG = ROOT / "configs/local_pages.toml"

PLUGINS = {
    "CIR-Generator": "cir-generator",
    "CodeGenerator": "codegenerator",
    "IQGenerator": "iqgenerator",
    "OptiReceiver": "opti-receiver",
    "Pulse-Shaping": "pulse-shaping",
    "ResultsSaverPlugin": "resultssaverplugin",
    "SignalChannelizer": "signal-channelizer",
}
PLUGINS_IMPORTING_PAGES = {"CIR-Generator", "OptiReceiver"}


def clone(url: str, branch: str, destination: Path) -> bool:
    """Install into a complete directory and keep any existing checkout."""
    if destination.is_symlink():
        raise ValueError(f"Expected a directory, found a link: {destination}")
    if (destination / ".git").exists():
        print(f"Using existing checkout {destination.relative_to(ROOT)}", flush=True)
        return False

    carried = []
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"Expected a directory: {destination}")
        if destination.name == "pages":
            tracked = subprocess.check_output(
                ["git", "ls-files", "-z", "--", "pages/"], cwd=ROOT
            )
            bundled = {
                ROOT / path.decode() for path in tracked.split(b"\0") if path
            }
            carried = list(destination.iterdir())
            if any(
                path not in bundled or path.is_symlink() or not path.is_file()
                for path in carried
            ):
                raise ValueError(f"Unexpected files in {destination}")
        elif any(destination.iterdir()):
            raise ValueError(f"Destination is not empty: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        work = Path(directory)
        stage = work / "checkout"
        subprocess.run(
            ["git", "clone", "--single-branch", "--branch", branch, "--", url, stage],
            check=True,
        )
        for path in carried:
            target = stage / path.name
            if target.exists() or target.is_symlink():
                if path.name == ".gitkeep":
                    continue
                raise ValueError(f"Both repositories own pages/{path.name}")
            shutil.copy2(path, target)
        previous = work / "previous"
        if destination.exists():
            destination.rename(previous)
        try:
            stage.rename(destination)
        except OSError:
            if previous.exists():
                previous.rename(destination)
            raise
    return True


def configure_pages(names: list[str]) -> None:
    files = []
    for name in names:
        file = name if name.endswith(".py") else f"{name}.py"
        if (
            Path(file).name != file
            or file.endswith("_worker.py")
            or not (ROOT / "pages" / file).is_file()
        ):
            raise ValueError(f"Unknown page: {name}")
        if file not in files:
            files.append(file)
    with tempfile.NamedTemporaryFile(
        prefix=".pages-", dir=PAGES_CONFIG.parent, mode="w", delete=False
    ) as handle:
        temp = Path(handle.name)
        handle.write(f"private = {json.dumps(files)}\n")
    temp.replace(PAGES_CONFIG)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", action="append", choices=(*PLUGINS, "all"), default=[])
    pages = parser.add_mutually_exclusive_group()
    pages.add_argument("--pages", action="store_true", help="enable all Pages")
    pages.add_argument("--page", action="append", default=[], help="enable a named page")
    parser.add_argument("--research", action="store_true", help="clone Research")
    parser.add_argument("--list", action="store_true", help="list available components")
    parser.add_argument("--gitlab-root", default=GITLAB, help="GitLab group URL or mirror")
    parser.add_argument("--skip-sync", action="store_true", help="clone sources only")
    args = parser.parse_args()

    if args.list:
        print("Plugins:", ", ".join(PLUGINS))
        print("Optional repositories: pages, research")
        return

    selected = set(PLUGINS) if "all" in args.plugin else set(args.plugin)
    groups = {PLUGINS[name] for name in selected}
    gitlab = args.gitlab_root.rstrip("/")
    if not args.skip_sync and shutil.which("uv") is None:
        parser.error("Install uv or pass --skip-sync to clone sources only.")
    try:
        if args.pages or args.page or selected & PLUGINS_IMPORTING_PAGES:
            installed = clone(f"{gitlab}/pages.git", PAGES_BRANCH, ROOT / "pages")
            groups.add("pages")
            if args.pages:
                PAGES_CONFIG.unlink(missing_ok=True)
            elif args.page or installed:
                configure_pages(args.page)

        for name in sorted(selected):
            branch = PAGES_BRANCH if name == "OptiReceiver" else "master"
            clone(f"{gitlab}/Plugins/{name}.git", branch, ROOT / "plugins" / name)

        if args.research:
            clone(f"{gitlab}/research.git", "master", ROOT / "research")

        if not args.skip_sync:
            command = ["uv", "sync", "--locked", "--inexact", "--no-default-groups"]
            for group in sorted(groups):
                command.extend(("--group", group))
            subprocess.run(command, check=True, cwd=ROOT)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")

    print("Start: uv run --no-sync streamlit run app.py")


if __name__ == "__main__":
    main()
