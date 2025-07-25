import json
import os
import re
from copy import deepcopy

import plotly.express as px
import streamlit as st

# --- CONSTANTS & DEFAULTS ---
FONTS = [
    "Arial",
    "Verdana",
    "Helvetica",
    "Times New Roman",
    "Courier New",
    "Georgia",
    "Garamond",
    "Arial Black",
    "Monospace",
]
DEFAULT_THEMES_FILE = "configs/plotly/default_themes.json"


# --- UTILITY FUNCTIONS ---
def hex_to_rgba(h, alpha=1.0):
    h = h.lstrip("#")
    if len(h) != 6:
        return "rgba(0, 0, 0, 1)"
    try:
        rgb = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
        return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"
    except ValueError:
        return "rgba(0, 0, 0, 1)"


def rgba_to_hex_alpha(rgba_string):
    if not isinstance(rgba_string, str):
        return "#000000", 1.0
    if rgba_string.startswith("#"):
        return rgba_string, 1.0
    match = re.search(
        r"rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)", rgba_string
    )
    if match:
        r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
        a = float(match.group(4)) if match.group(4) is not None else 1.0
        return f"#{r:02x}{g:02x}{b:02x}", a
    return "#000000", 1.0


def invert_color(rgba_string):
    if not isinstance(rgba_string, str) or "rgba" not in rgba_string:
        return rgba_string
    match = re.search(
        r"rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)", rgba_string
    )
    if match:
        r, g, b, a = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            float(match.group(4)),
        )
        return f"rgba({255 - r}, {255 - g}, {255 - b}, {a})"
    return rgba_string


def unflatten_dict(d):
    result = {}
    for key, value in d.items():
        if key.startswith("layout."):
            key = key.replace("layout.", "", 1)
        parts = key.split(".")
        nested_dict = result
        for part in parts[:-1]:
            nested_dict = nested_dict.setdefault(part, {})
        nested_dict[parts[-1]] = value
    return result


# --- THEME MANAGEMENT ---
@st.cache_data
def get_master_light_theme_defaults():
    return {
        "layout.paper_bgcolor": "rgba(255, 255, 255, 1)",
        "layout.plot_bgcolor": "rgba(239, 239, 239, 0.95)",
        "layout.autosize": True,
        "layout.template": "plotly_white",
        "layout.margin.l": 80,
        "layout.margin.r": 80,
        "layout.margin.t": 100,
        "layout.margin.b": 80,
        "layout.margin.pad": 0,
        "layout.hoverlabel.bgcolor": "rgba(255, 255, 255, 0.9)",
        "layout.hoverlabel.bordercolor": "rgba(0, 0, 0, 1)",
        "layout.font.family": "Arial",
        "layout.font.size": 12,
        "layout.font.color": "rgba(0, 0, 0, 1)",
        "layout.font.style": "normal",
        "layout.font.weight": 400,
        "layout.font.textcase": "normal",
        "layout.title.font.family": "Arial Black",
        "layout.title.font.size": 24,
        "layout.title.font.color": "rgba(0, 0, 0, 1)",
        "layout.title.font.style": "normal",
        "layout.title.font.weight": 400,
        "layout.title.x": 0.5,
        "layout.title.xanchor": "center",
        "layout.title.pad.t": 10,
        "layout.title.pad.b": 10,
        "layout.title.subtitle.text": "",
        "layout.title.subtitle.font.color": "rgba(0, 0, 0, 1)",
        "layout.showlegend": True,
        "layout.legend.bgcolor": "rgba(255, 255, 255, 0.7)",
        "layout.legend.bordercolor": "rgba(0, 0, 0, 0.5)",
        "layout.legend.borderwidth": 1,
        "layout.legend.font.family": "Arial",
        "layout.legend.font.size": 11,
        "layout.legend.font.color": "rgba(0,0,0,1)",
        "layout.legend.font.style": "normal",
        "layout.legend.font.weight": 400,
        "layout.legend.orientation": "v",
        "layout.legend.title.text": "",
        "layout.legend.x": 1.02,
        "layout.legend.y": 1.0,
        "layout.legend.xanchor": "left",
        "layout.legend.yanchor": "auto",
        "layout.legend.xref": "paper",
        "layout.legend.yref": "paper",
        "layout.xaxis.showgrid": False,
        "layout.xaxis.showline": True,
        "layout.xaxis.mirror": True,
        "layout.xaxis.gridcolor": "rgba(204, 204, 204, 1)",
        "layout.xaxis.linecolor": "rgba(0, 0, 0, 1)",
        "layout.xaxis.linewidth": 2,
        "layout.xaxis.title.font.family": "Arial",
        "layout.xaxis.title.font.size": 14,
        "layout.xaxis.tickfont.family": "Arial",
        "layout.xaxis.tickfont.size": 12,
        "layout.xaxis.tickfont.color": "rgba(0, 0, 0, 1)",
        "layout.xaxis.tickfont.style": "normal",
        "layout.xaxis.tickfont.weight": 400,
        "layout.xaxis.tickangle": 0,
        "layout.xaxis.zeroline": False,
        "layout.xaxis.zerolinecolor": "rgba(0, 0, 0, 1)",
        "layout.yaxis.showgrid": True,
        "layout.yaxis.showline": True,
        "layout.yaxis.mirror": True,
        "layout.yaxis.gridcolor": "rgba(204, 204, 204, 1)",
        "layout.yaxis.linecolor": "rgba(0, 0, 0, 1)",
        "layout.yaxis.linewidth": 2,
        "layout.yaxis.title.font.family": "Arial",
        "layout.yaxis.title.font.size": 14,
        "layout.yaxis.tickfont.family": "Arial",
        "layout.yaxis.tickfont.size": 12,
        "layout.yaxis.tickfont.color": "rgba(0, 0, 0, 1)",
        "layout.yaxis.tickfont.style": "normal",
        "layout.yaxis.tickfont.weight": 400,
        "layout.yaxis.tickangle": 0,
        "layout.yaxis.zeroline": False,
        "layout.yaxis.zerolinecolor": "rgba(0, 0, 0, 1)",
    }


@st.cache_data
def get_master_dark_theme_defaults():
    return {
        "layout.paper_bgcolor": "rgba(17, 17, 17, 1)",
        "layout.plot_bgcolor": "rgba(34, 34, 34, 0.95)",
        "layout.autosize": True,
        "layout.template": "plotly_dark",
        "layout.margin.l": 80,
        "layout.margin.r": 80,
        "layout.margin.t": 100,
        "layout.margin.b": 80,
        "layout.margin.pad": 0,
        "layout.hoverlabel.bgcolor": "rgba(51, 51, 51, 0.9)",
        "layout.hoverlabel.bordercolor": "rgba(255, 255, 255, 1)",
        "layout.font.family": "Arial",
        "layout.font.size": 12,
        "layout.font.color": "rgba(255, 255, 255, 1)",
        "layout.font.style": "normal",
        "layout.font.weight": 400,
        "layout.font.textcase": "normal",
        "layout.title.font.family": "Arial Black",
        "layout.title.font.size": 24,
        "layout.title.font.color": "rgba(255, 255, 255, 1)",
        "layout.title.font.style": "normal",
        "layout.title.font.weight": 400,
        "layout.title.x": 0.5,
        "layout.title.xanchor": "center",
        "layout.title.pad.t": 10,
        "layout.title.pad.b": 10,
        "layout.title.subtitle.text": "",
        "layout.title.subtitle.font.color": "rgba(255, 255, 255, 1)",
        "layout.showlegend": True,
        "layout.legend.bgcolor": "rgba(51, 51, 51, 0.7)",
        "layout.legend.bordercolor": "rgba(255, 255, 255, 0.5)",
        "layout.legend.borderwidth": 1,
        "layout.legend.font.family": "Arial",
        "layout.legend.font.size": 11,
        "layout.legend.font.color": "rgba(255,255,255,1)",
        "layout.legend.font.style": "normal",
        "layout.legend.font.weight": 400,
        "layout.legend.orientation": "v",
        "layout.legend.title.text": "",
        "layout.legend.x": 1.02,
        "layout.legend.y": 1.0,
        "layout.legend.xanchor": "left",
        "layout.legend.yanchor": "auto",
        "layout.legend.xref": "paper",
        "layout.legend.yref": "paper",
        "layout.xaxis.showgrid": False,
        "layout.xaxis.showline": True,
        "layout.xaxis.mirror": True,
        "layout.xaxis.gridcolor": "rgba(85, 85, 85, 1)",
        "layout.xaxis.linecolor": "rgba(255, 255, 255, 1)",
        "layout.xaxis.linewidth": 2,
        "layout.xaxis.title.font.family": "Arial",
        "layout.xaxis.title.font.size": 14,
        "layout.xaxis.tickfont.family": "Arial",
        "layout.xaxis.tickfont.size": 12,
        "layout.xaxis.tickfont.color": "rgba(255, 255, 255, 1)",
        "layout.xaxis.tickfont.style": "normal",
        "layout.xaxis.tickfont.weight": 400,
        "layout.xaxis.tickangle": 0,
        "layout.xaxis.zeroline": False,
        "layout.xaxis.zerolinecolor": "rgba(255, 255, 255, 1)",
        "layout.yaxis.showgrid": True,
        "layout.yaxis.showline": True,
        "layout.yaxis.mirror": True,
        "layout.yaxis.gridcolor": "rgba(85, 85, 85, 1)",
        "layout.yaxis.linecolor": "rgba(255, 255, 255, 1)",
        "layout.yaxis.linewidth": 2,
        "layout.yaxis.title.font.family": "Arial",
        "layout.yaxis.title.font.size": 14,
        "layout.yaxis.tickfont.family": "Arial",
        "layout.yaxis.tickfont.size": 12,
        "layout.yaxis.tickfont.color": "rgba(255, 255, 255, 1)",
        "layout.yaxis.tickfont.style": "normal",
        "layout.yaxis.tickfont.weight": 400,
        "layout.yaxis.tickangle": 0,
        "layout.yaxis.zeroline": False,
        "layout.yaxis.zerolinecolor": "rgba(255, 255, 255, 1)",
    }


def get_themes_from_json():
    if os.path.exists(DEFAULT_THEMES_FILE):
        with open(DEFAULT_THEMES_FILE, "r") as f:
            return json.load(f)
    return {"light": {}, "dark": {}}


@st.cache_data
def get_demo_plots():
    df_scatter = px.data.iris()
    fig_scatter = px.scatter(
        df_scatter,
        x="sepal_width",
        y="sepal_length",
        color="species",
        title="Scatter Plot Example",
    )
    df_line = px.data.stocks()
    fig_line = px.line(
        df_line, x="date", y=["GOOG", "AAPL"], title="Line Chart Example"
    )
    return {"Scatter Plot": fig_scatter, "Line Chart": fig_line}


# --- MAIN APP ---
st.set_page_config(layout="wide")
st.title("🎨 Advanced Plotly Config Generator")

# --- DEFINITIVE STATE INITIALIZATION (RUNS ONCE PER SESSION) ---
if "themes" not in st.session_state:
    st.session_state.themes = {"light": {}, "dark": {}}

    light_master_defaults = get_master_light_theme_defaults()
    dark_master_defaults = get_master_dark_theme_defaults()
    json_themes = get_themes_from_json()
    light_json = json_themes.get("light", {})
    dark_json = json_themes.get("dark", {})

    for key, default_value in light_master_defaults.items():
        value = light_json.get(key, default_value)
        st.session_state.themes["light"][key] = {
            "value": value,
            "include": True,
        }

    for key, default_value in dark_master_defaults.items():
        value = dark_json.get(key, default_value)
        st.session_state.themes["dark"][key] = {
            "value": value,
            "include": True,
        }

# --- GLOBAL STATE ---
demo_plots = get_demo_plots()
active_theme_name = st.sidebar.radio(
    "Select Theme to Edit:",
    ("light", "dark"),
    format_func=lambda x: x.capitalize(),
    key="theme_selector",
)
active_theme_params = st.session_state.themes[active_theme_name]


# --- REUSABLE WIDGET CREATION FUNCTIONS ---
def handle_value_change(theme_name, param_key, widget_key):
    st.session_state.themes[theme_name][param_key]["value"] = st.session_state[
        widget_key
    ]


def handle_checkbox_change(theme_name, param_key, widget_key):
    st.session_state.themes[theme_name][param_key]["include"] = (
        st.session_state[widget_key]
    )


def create_widget(widget_type, label, key, help=None, **kwargs):
    param_data = active_theme_params[key]
    widget_key = f"{active_theme_name}_{key}"
    include_key = f"{widget_key}_include"

    cols = st.columns([0.8, 0.2])
    with cols[0]:
        if widget_type == "number_input":
            st.number_input(
                label,
                value=param_data["value"],
                key=widget_key,
                help=help,
                on_change=handle_value_change,
                args=(active_theme_name, key, widget_key),
                disabled=not param_data["include"],
                **kwargs,
            )
        elif widget_type == "toggle":
            st.toggle(
                label,
                value=bool(param_data["value"]),
                key=widget_key,
                help=help,
                on_change=handle_value_change,
                args=(active_theme_name, key, widget_key),
                disabled=not param_data["include"],
                **kwargs,
            )
        elif widget_type == "selectbox":
            opts = kwargs.pop("options", [])
            idx = (
                opts.index(param_data["value"])
                if param_data["value"] in opts
                else 0
            )
            st.selectbox(
                label,
                options=opts,
                index=idx,
                key=widget_key,
                help=help,
                on_change=handle_value_change,
                args=(active_theme_name, key, widget_key),
                disabled=not param_data["include"],
                **kwargs,
            )
        elif widget_type == "text_input":
            st.text_input(
                label,
                value=str(param_data["value"]),
                key=widget_key,
                help=help,
                on_change=handle_value_change,
                args=(active_theme_name, key, widget_key),
                disabled=not param_data["include"],
                **kwargs,
            )
    with cols[1]:
        st.checkbox(
            "Include",
            value=param_data["include"],
            key=include_key,
            label_visibility="hidden",
            on_change=handle_checkbox_change,
            args=(active_theme_name, key, include_key),
        )


def handle_color_value_change(theme_name, param_key, hex_key, alpha_key):
    new_color = hex_to_rgba(
        st.session_state[hex_key], st.session_state[alpha_key]
    )
    st.session_state.themes[theme_name][param_key]["value"] = new_color


def create_color_widget(label, key, help=None):
    param_data = active_theme_params[key]
    widget_key = f"{active_theme_name}_{key}"
    hex_key = f"{widget_key}_hex"
    alpha_key = f"{widget_key}_alpha"
    include_key = f"{widget_key}_include"

    hex_val, alpha_val = rgba_to_hex_alpha(param_data["value"])

    cols = st.columns([0.6, 0.2, 0.2])
    with cols[0]:
        st.color_picker(
            label,
            value=hex_val,
            key=hex_key,
            help=help,
            on_change=handle_color_value_change,
            args=(active_theme_name, key, hex_key, alpha_key),
            disabled=not param_data["include"],
        )
    with cols[1]:
        st.number_input(
            "Alpha",
            0.0,
            1.0,
            alpha_val,
            0.05,
            key=alpha_key,
            label_visibility="collapsed",
            on_change=handle_color_value_change,
            args=(active_theme_name, key, hex_key, alpha_key),
            disabled=not param_data["include"],
        )
    with cols[2]:
        st.checkbox(
            "Include",
            value=param_data["include"],
            key=include_key,
            label_visibility="hidden",
            on_change=handle_checkbox_change,
            args=(active_theme_name, key, include_key),
        )


def create_font_widget(label, key, help=None):
    param_data = active_theme_params[key]
    widget_key = f"{active_theme_name}_{key}"
    include_key = f"{widget_key}_include"

    font = param_data["value"]
    idx = FONTS.index(font) if font in FONTS else 0

    cols = st.columns([0.8, 0.2])
    with cols[0]:
        st.selectbox(
            label,
            FONTS,
            index=idx,
            key=widget_key,
            help=help,
            on_change=handle_value_change,
            args=(active_theme_name, key, widget_key),
            disabled=not param_data["include"],
        )
    with cols[1]:
        st.checkbox(
            "Include",
            value=param_data["include"],
            key=include_key,
            label_visibility="hidden",
            on_change=handle_checkbox_change,
            args=(active_theme_name, key, include_key),
        )


# --- SIDEBAR UI ---
with st.sidebar:
    st.subheader("Demo Chart")
    selected_plot = st.selectbox(
        "Choose a demo plot:", list(demo_plots.keys())
    )

    st.subheader("Config Management")
    config_filename = st.text_input("Filename", "my_plot_config.json")

    def _save_config(path):
        config_to_save = {}
        for theme_name, all_params in st.session_state.themes.items():
            config_to_save[theme_name] = {
                key: data["value"]
                for key, data in all_params.items()
                if data["include"]
            }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(config_to_save, f, indent=4)
        st.success(f"Config saved to `{path}`")

    if "confirm_overwrite_path" in st.session_state:
        st.warning(
            f"File '{os.path.basename(st.session_state.confirm_overwrite_path)}' already exists."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button(
                "Overwrite", use_container_width=True, type="primary"
            ):
                _save_config(st.session_state.confirm_overwrite_path)
                del st.session_state.confirm_overwrite_path
                st.rerun()
        with c2:
            if st.button("Save New", use_container_width=True):
                base, ext = os.path.splitext(
                    st.session_state.confirm_overwrite_path
                )
                i = 2
                while os.path.exists(f"{base}_v{i}{ext}"):
                    i += 1
                new_path = f"{base}_v{i}{ext}"
                _save_config(new_path)
                del st.session_state.confirm_overwrite_path
                st.rerun()
        with c3:
            if st.button("Cancel", use_container_width=True):
                del st.session_state.confirm_overwrite_path
                st.rerun()
    else:
        if st.button("💾 Save Current Configs"):
            save_path = os.path.join("configs", "plotly", config_filename)
            if os.path.normpath(save_path) == os.path.normpath(
                DEFAULT_THEMES_FILE
            ):
                st.error(
                    "Cannot overwrite the default theme file. Please choose a different name."
                )
            elif os.path.exists(save_path):
                st.session_state.confirm_overwrite_path = save_path
                st.rerun()
            else:
                _save_config(save_path)

    uploaded_file = st.file_uploader("📂 Load Config File", type="json")
    if uploaded_file and st.button(
        "Apply Loaded Config", use_container_width=True, type="primary"
    ):
        try:
            loaded_data = json.load(uploaded_file)
            loaded_light = loaded_data.get("light", {})
            loaded_dark = loaded_data.get("dark", {})

            for key in st.session_state.themes["light"]:
                if key in loaded_light:
                    st.session_state.themes["light"][key]["value"] = (
                        loaded_light[key]
                    )
                    st.session_state.themes["light"][key]["include"] = True
                else:
                    st.session_state.themes["light"][key]["include"] = False

            for key in st.session_state.themes["dark"]:
                if key in loaded_dark:
                    st.session_state.themes["dark"][key]["value"] = (
                        loaded_dark[key]
                    )
                    st.session_state.themes["dark"][key]["include"] = True
                else:
                    st.session_state.themes["dark"][key]["include"] = False

            st.success("Configuration loaded!")
            st.rerun()
        except Exception as e:
            st.error(f"Invalid config file. Error: {e}")

    if st.button(f"🔄 Reset {active_theme_name.capitalize()} Theme"):
        json_themes = get_themes_from_json()
        if active_theme_name == "light":
            master_defaults = get_master_light_theme_defaults()
            json_defaults = json_themes.get("light", {})
        else:
            master_defaults = get_master_dark_theme_defaults()
            json_defaults = json_themes.get("dark", {})

        for key, master_value in master_defaults.items():
            value = json_defaults.get(key, master_value)
            st.session_state.themes[active_theme_name][key]["value"] = value
            st.session_state.themes[active_theme_name][key]["include"] = True
        st.rerun()

    st.subheader("Theme Synchronization")
    target_theme_name = "dark" if active_theme_name == "light" else "light"

    if st.button(f"Copy Style Values to {target_theme_name.capitalize()}"):
        for key, data in st.session_state.themes[active_theme_name].items():
            st.session_state.themes[target_theme_name][key]["value"] = data[
                "value"
            ]
        st.rerun()

    if st.button(f'Copy "Include" States to {target_theme_name.capitalize()}'):
        for key, data in st.session_state.themes[active_theme_name].items():
            st.session_state.themes[target_theme_name][key]["include"] = data[
                "include"
            ]
        st.rerun()

    if st.button(f"Copy & Invert Colors to {target_theme_name.capitalize()}"):
        for key, data in st.session_state.themes[active_theme_name].items():
            new_value = (
                invert_color(data["value"])
                if "color" in key.lower()
                else data["value"]
            )
            st.session_state.themes[target_theme_name][key]["value"] = (
                new_value
            )
        st.rerun()

# --- MAIN UI ---
col1, col2 = st.columns([0.4, 0.6])
with col1:
    st.header(f"Styling Options ({active_theme_name.capitalize()} Theme)")
    tabs = st.tabs(
        ["General", "Typography", "Axes", "Legend", "Interactivity"]
    )
    with tabs[0]:
        with st.expander("🎨 Background & Template", expanded=True):
            create_color_widget("Paper Background", "layout.paper_bgcolor")
            create_color_widget("Plot Area Background", "layout.plot_bgcolor")
            create_widget(
                "selectbox",
                "Plotly Template",
                "layout.template",
                options=[
                    "plotly",
                    "plotly_white",
                    "plotly_dark",
                    "ggplot2",
                    "seaborn",
                    "simple_white",
                    "none",
                ],
            )
            create_widget("toggle", "Autosize", "layout.autosize")
        with st.expander("📐 Margins & Padding", expanded=True):
            create_widget(
                "number_input", "Left Margin", "layout.margin.l", min_value=0
            )
            create_widget(
                "number_input", "Right Margin", "layout.margin.r", min_value=0
            )
            create_widget(
                "number_input", "Top Margin", "layout.margin.t", min_value=0
            )
            create_widget(
                "number_input", "Bottom Margin", "layout.margin.b", min_value=0
            )
            create_widget(
                "number_input", "Padding", "layout.margin.pad", min_value=0
            )
    with tabs[1]:
        with st.expander("🔤 Global Font", expanded=True):
            create_font_widget("Font Family", "layout.font.family")
            create_widget(
                "number_input", "Font Size", "layout.font.size", min_value=1
            )
            create_color_widget("Font Color", "layout.font.color")
            create_widget(
                "selectbox",
                "Font Style",
                "layout.font.style",
                options=["normal", "italic"],
            )
            create_widget(
                "number_input",
                "Font Weight",
                "layout.font.weight",
                min_value=1,
                max_value=1000,
                step=100,
            )
            create_widget(
                "selectbox",
                "Text Case",
                "layout.font.textcase",
                options=["normal", "word caps", "upper", "lower"],
            )
        with st.expander("✒️ Main Title", expanded=True):
            create_font_widget("Title Font Family", "layout.title.font.family")
            create_widget(
                "number_input",
                "Title Font Size",
                "layout.title.font.size",
                min_value=1,
            )
            create_color_widget("Title Font Color", "layout.title.font.color")
            create_widget(
                "number_input",
                "Title X Position",
                "layout.title.x",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
            )
            create_widget(
                "selectbox",
                "Title X Anchor",
                "layout.title.xanchor",
                options=["auto", "left", "center", "right"],
            )
        with st.expander("✏️ Subtitle", expanded=True):
            create_widget(
                "text_input", "Subtitle Text", "layout.title.subtitle.text"
            )
            create_color_widget(
                "Subtitle Font Color", "layout.title.subtitle.font.color"
            )
    with tabs[2]:
        with st.expander("📏 X-Axis", expanded=True):
            create_widget("toggle", "Show Grid", "layout.xaxis.showgrid")
            create_color_widget("Grid Color", "layout.xaxis.gridcolor")
            create_widget("toggle", "Show Line", "layout.xaxis.showline")
            create_color_widget("Line Color", "layout.xaxis.linecolor")
            create_widget(
                "number_input",
                "Line Width",
                "layout.xaxis.linewidth",
                min_value=0,
            )
            create_widget("toggle", "Mirror Ticks", "layout.xaxis.mirror")
            st.subheader("Tick Labels")
            create_font_widget(
                "Tick Font Family", "layout.xaxis.tickfont.family"
            )
            create_widget(
                "number_input",
                "Tick Font Size",
                "layout.xaxis.tickfont.size",
                min_value=1,
            )
            create_color_widget(
                "Tick Font Color", "layout.xaxis.tickfont.color"
            )
            create_widget(
                "number_input",
                "Tick Angle",
                "layout.xaxis.tickangle",
                min_value=-180,
                max_value=180,
            )
        with st.expander("📐 Y-Axis", expanded=True):
            create_widget("toggle", "Show Grid", "layout.yaxis.showgrid")
            create_color_widget("Grid Color", "layout.yaxis.gridcolor")
            create_widget("toggle", "Show Line", "layout.yaxis.showline")
            create_color_widget("Line Color", "layout.yaxis.linecolor")
            create_widget(
                "number_input",
                "Line Width",
                "layout.yaxis.linewidth",
                min_value=0,
            )
            create_widget("toggle", "Mirror Ticks", "layout.yaxis.mirror")
            st.subheader("Tick Labels")
            create_font_widget(
                "Tick Font Family", "layout.yaxis.tickfont.family"
            )
            create_widget(
                "number_input",
                "Tick Font Size",
                "layout.yaxis.tickfont.size",
                min_value=1,
            )
            create_color_widget(
                "Tick Font Color", "layout.yaxis.tickfont.color"
            )
            create_widget(
                "number_input",
                "Tick Angle",
                "layout.yaxis.tickangle",
                min_value=-180,
                max_value=180,
            )
    with tabs[3]:
        with st.expander("📜 Legend Settings", expanded=True):
            st.subheader("Layout & Font")
            create_widget("toggle", "Show Legend", "layout.showlegend")
            create_widget(
                "selectbox",
                "Orientation",
                "layout.legend.orientation",
                options=["v", "h"],
            )
            create_color_widget("Background Color", "layout.legend.bgcolor")
            create_widget(
                "number_input",
                "Border Width",
                "layout.legend.borderwidth",
                min_value=0,
            )
            create_color_widget("Border Color", "layout.legend.bordercolor")
            create_font_widget("Font Family", "layout.legend.font.family")
            create_widget(
                "number_input",
                "Font Size",
                "layout.legend.font.size",
                min_value=1,
            )
            create_color_widget("Font Color", "layout.legend.font.color")

            st.subheader("Position & Anchoring")
            c1, c2 = st.columns(2)
            with c1:
                create_widget(
                    "number_input",
                    "X Position",
                    "layout.legend.x",
                    min_value=-2.0,
                    max_value=3.0,
                    step=0.1,
                    help="Value between -2 and 3",
                )
                create_widget(
                    "selectbox",
                    "X Anchor",
                    "layout.legend.xanchor",
                    options=["auto", "left", "center", "right"],
                )
                create_widget(
                    "selectbox",
                    "X Reference",
                    "layout.legend.xref",
                    options=["container", "paper"],
                )
            with c2:
                create_widget(
                    "number_input",
                    "Y Position",
                    "layout.legend.y",
                    min_value=-2.0,
                    max_value=3.0,
                    step=0.1,
                    help="Value between -2 and 3",
                )
                create_widget(
                    "selectbox",
                    "Y Anchor",
                    "layout.legend.yanchor",
                    options=["auto", "top", "middle", "bottom"],
                )
                create_widget(
                    "selectbox",
                    "Y Reference",
                    "layout.legend.yref",
                    options=["container", "paper"],
                )

        with st.expander("✨ Legend Title", expanded=True):
            create_widget(
                "text_input", "Title Text", "layout.legend.title.text"
            )
    with tabs[4]:
        with st.expander("🖱️ Hover Labels", expanded=True):
            create_color_widget(
                "Hover Background", "layout.hoverlabel.bgcolor"
            )
            create_color_widget(
                "Hover Border", "layout.hoverlabel.bordercolor"
            )

# --- LIVE PREVIEW & CODE GENERATION ---
with col2:
    st.header("Live Preview")
    included_style_dict = {
        key: data["value"]
        for key, data in active_theme_params.items()
        if data["include"]
    }
    fig_to_display = deepcopy(demo_plots[selected_plot])
    nested_style_dict = unflatten_dict(included_style_dict)

    if nested_style_dict:
        fig_to_display.update_layout(nested_style_dict)

    st.plotly_chart(fig_to_display, use_container_width=True, theme=None)

    with st.expander("Generated Config", expanded=True):
        st.code(
            f"style_dict = {json.dumps(included_style_dict, indent=4)}",
            language="python",
        )
