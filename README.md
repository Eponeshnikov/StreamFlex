# 🎛️ StreamFlex

A flexible application framework with plugin support, state snapshots, and dynamic widget management. Built with Streamlit for rapid UI development.

## ✨ Features

- **🔌 Plugin System**  
  Dynamically load/unload plugins from the `plugins` directory. Each plugin can:
  - Maintain its own UI widgets
  - Persist state between sessions
  - Define custom data processing logic

- **📸 Snapshot Management**  
  Save/Load complete application states including:
  - Selected plugins
  - Widget configurations
  - Application data
  - (Auto-saved in `snapshots` directory)

- **🛠️ Developer Tools**  
  Built-in debugging features:
  - Real-time session state inspection
  - Colored log viewer with auto-scroll
  - One-click state refresh
  - Log cleaning utility

- **🧩 Extensible Architecture**  
  Core components:
  - `PluginManager`: Handles plugin lifecycle
  - `WidgetManager`: Persistent widget states
  - `DataManager`: Shared data storage
  - `StateManager`: Snapshot operations

- **📂 Page Categorization**
  Dynamic page categorization is managed through the `configs/page_categories.yaml` file. This configuration:
  - Defines categories for organizing pages in the navigation bar
  - Specifies matching rules (startswith, contains, endswith) for assigning pages to categories
  - Uses a YAML format with 'match' type ('all' or 'any') and rules list
  - Example configuration:
    ```yaml
    categories:
      "Peaks Processing":
        match: all
        rules:
          - type: startswith
          value: "private_"
          - type: contains
          value: "peak"
      
      "Analytics":
        match: any
        rules:
          - type: contains
          value: "analysis"
          - type: endswith
          value: "_analytics.py"
    ```
  - Pages not matching any category are placed in the default "Resources" category

## 🚀 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Plugin Dashboard**
   - Select active plugins from multi-select dropdown
   - Each plugin renders its UI in collapsible sections
   - Interactive widgets persist state automatically

3. **Snapshot Controls (Sidebar)**
   - **Save**: Capture current state with name
   - **Load**: Restore from previous snapshot
   - **Delete**: Remove unwanted snapshots

4. **Debug Console (Sidebar)**
   - View real-time session state
   - Monitor colored log output
   - Refresh app state or clear logs

## Execution & Deployment

### Installing Dependencies

Firstly, you need install [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

To install dependencies for a project, use the `uv sync` command:

```bash
uv sync
```

This command installs all dependencies listed in `pyprojects.toml`.

### Launching the Application

Once the dependencies are installed, launch the application using Streamlit with the following command:

```bash
streamlit run app.py
```