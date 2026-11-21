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

### Installation

Install Git, Python 3.10 or newer for the bootstrap script, and [uv](https://docs.astral.sh/uv/getting-started/installation/). The application environment uses the Python version in `.python-version`. GitLab access is required for private repositories.

From a new machine, clone the core and install the selected components in one command.

```bash
git clone https://gitlab.epohub.netcraze.pro/StreamFlex/StreamFlex.git && python3 StreamFlex/scripts/bootstrap.py --plugin CIR-Generator --plugin ResultsSaverPlugin --page private_peaks_processing
```

- Omit the flags for the core and its two bundled plotting pages. Repeat `--plugin NAME` to select plugins, or use `--plugin all`.
- Use `--pages` for every private page or repeat `--page NAME` to select entries in the navigation. The Pages checkout includes its shared modules. CIR-Generator and OptiReceiver also need that checkout.
- Add `--research` to clone the separate Research repository. `--skip-sync` fetches source only. `--list` lists plugin names.
- The script keeps existing checkouts and installs from the root lockfile. Pages and OptiReceiver use `research/rtw8-fourway-20261117`; the other components use `master`. Set `--gitlab-root` to use an alternate mirror.

Existing measurements, scenes, models, results and tracing caches can be copied separately. See the [data layout](docs/DATA_LAYOUT.md).

### Launch

Run from the project root.

```bash
uv run --no-sync streamlit run app.py
```

[Core documentation](docs/README.md) · [Pages documentation](pages/docs/README.md) · [Research](research/README.md)
