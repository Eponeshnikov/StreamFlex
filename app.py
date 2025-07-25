import streamlit as st
import os
import yaml  # Required to parse the config file

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
        st.error(f"Configuration file not found at '{path}'. Please create it.")
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
        st.error("The 'pages' directory was not found. Please ensure it exists.")
        files_in_pages = []

    # Scan pages directory and assign files to categories
    for filename in sorted(files_in_pages):
        if filename.endswith(".py"):
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
                pages[matched_category].append(st.Page(page_path, title=page_title))
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