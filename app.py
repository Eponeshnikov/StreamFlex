import streamlit as st
import os

pages = {
    "App": [
        st.Page("streamflex_app.py", title="Streamflex"),
    ],
    "Resources": [],
}

# Scan pages directory and add files with formatted titles
for filename in os.listdir("pages"):
    if filename.endswith(".py"):
        # Remove file extension before formatting title
        name_without_extension = os.path.splitext(filename)[0]
        # Create title by replacing underscores and removing 'private_' prefix if present
        title = name_without_extension.replace("_", " ").title()
        if title.startswith("Private "):
            title = title.replace("Private ", "")
        
        # Add page to navigation
        pages["Resources"].append(st.Page(f"pages/{filename}", title=title))

pg = st.navigation(pages, position="top")
pg.run()