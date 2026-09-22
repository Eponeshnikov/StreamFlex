import os

import streamlit as st
import yaml  # Required to parse the config file

from session_guard import install_session_guards
from utils import (
    configure_joblib_temp_folder,
    logger,
    sweep_orphaned_joblib_folders,
)

# Every page runs through this entry script, so this is the one place that
# covers the offline pages too: without it a closed tab aborts a peaks-
# processing or training run exactly the way it aborts a plugin chain.
install_session_guards()

# ...and the same reason puts these two here. joblib hands its workers'
# array arguments over through a spill folder, and left to itself it picks
# /dev/shm — tmpfs, i.e. RAM that belongs to no process: it is in neither
# the app's RSS nor the cgroup's usage, and it outlives whatever made it.
# A run killed for using too much memory therefore leaves its spill behind
# for the *next* run to start against (measured: 50 GiB from two processes
# that had already died). So the folder is moved onto a disk, and start-up
# is the only moment after such a death when anything can sweep up.
_joblib_folder = configure_joblib_temp_folder()
_swept, _swept_bytes = sweep_orphaned_joblib_folders()
if _swept:
    logger.info(
        f"Reclaimed {_swept_bytes / 1024**3:.1f} GiB from {_swept} joblib "
        f"spill folder(s) left behind by processes that are gone."
    )


# --- Helper Function for Formatting Titles ---
def format_title(filename):
    """
    Formats a filename into a human-readable title.
    - Removes the .py extension.
    - Replaces underscores with spaces.
    - Capitalizes the title.
    - Removes the 'Private ' prefix if it exists.
    """
    name_without_extension = os.path.splitext(filename)[0]
    title = name_without_extension.replace("_", " ").title()
    if title.startswith("Private "):
        title = title.replace("Private ", "", 1)
    return title


# --- Functions for Loading and Applying Config Rules ---
def load_config(path="configs/page_categories.yaml"):
    """Loads the category configuration from a YAML file."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error(
            f"Configuration file not found at '{path}'. Please create it."
        )
        return None
    except yaml.YAMLError as e:
        st.error(f"Error parsing YAML file: {e}")
        return None


def check_file_match(filename, rules, match_type):
    """
    Checks if a filename matches a set of rules.
    - filename: The name of the file to check.
    - rules: A list of rule dictionaries (e.g., {'type': 'startswith', 'value': 'prefix'}).
    - match_type: 'all' (all rules must pass) or 'any' (at least one rule must pass).
    """
    results = []
    for rule in rules:
        rule_type = rule.get("type")
        value = rule.get("value")
        if rule_type == "startswith":
            results.append(filename.startswith(value))
        elif rule_type == "contains":
            results.append(value in filename)
        elif rule_type == "endswith":
            results.append(filename.endswith(value))

    if match_type == "all":
        return all(results)
    elif match_type == "any":
        return any(results)
    return False


# --- Main Application Logic ---

# Load the configuration from the YAML file
config = load_config()

# Initialize the pages dictionary with a static "App" page
pages = {
    "App": [
        st.Page("streamflex_app.py", title="Streamflex"),
    ],
}

if config and "categories" in config:
    # Dynamically add keys for each configured category and a default "Resources" category
    for category in config["categories"]:
        pages[category] = []
    pages["Resources"] = []

    # Get the list of files from the 'pages' directory
    try:
        files_in_pages = os.listdir("pages")
    except FileNotFoundError:
        st.error(
            "The 'pages' directory was not found. Please ensure it exists."
        )
        files_in_pages = []

    # Files listed under `exclude` in the config are import-only modules
    # (e.g. shared configuration) that live under pages/ but are not pages —
    # skip them so they don't show up as nav entries.
    excluded_files = set(config.get("exclude", []))

    # Scan pages directory and assign files to categories. Files ending in
    # `_worker.py` are import-only helpers that must live under pages/ (so
    # multiprocessing can pickle them by qualified name) but are not pages —
    # skip them so they don't show up as blank nav entries.
    for filename in sorted(files_in_pages):
        if (
            filename.endswith(".py")
            and not filename.endswith("_worker.py")
            and filename not in excluded_files
        ):
            page_path = f"pages/{filename}"
            page_title = format_title(filename)

            matched_category = None
            # Check the file against the rules from the config
            for category, details in config["categories"].items():
                rules = details.get("rules", [])
                match_type = details.get("match", "all")
                if check_file_match(filename, rules, match_type):
                    matched_category = category
                    break  # Stop at the first matching category

            # Add the page to the matched category or to "Resources" as a fallback
            if matched_category:
                pages[matched_category].append(
                    st.Page(page_path, title=page_title)
                )
            else:
                pages["Resources"].append(st.Page(page_path, title=page_title))

# Remove any empty categories before displaying the navigation
final_pages = {category: pgs for category, pgs in pages.items() if pgs}

# Create and run the navigation
if final_pages:
    pg = st.navigation(final_pages, position="top")
    pg.run()
else:
    st.warning("No pages were found or configured.")
