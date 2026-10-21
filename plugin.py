import importlib.util
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Literal

import streamlit as st
import toml
from loguru import logger
from streamlit.errors import StreamlitAPIException


def load_sibling(anchor: str, filename: str) -> ModuleType:
    """The module ``filename`` in the folder of ``anchor`` (a ``__file__``).

    A plugin folder name is not importable (``CIR-Generator``), so its
    modules are loaded by path. They are deliberately kept out of
    ``sys.modules``: cloudpickle then ships their functions and classes to
    loky workers by value, and a worker never has to import the plugin.

    The plugin manager executes ``plugin.py`` afresh on every rerun, and
    module-level values (the run timestamp, scratch roots) are read then.
    So the modules are cached per load of the calling plugin, not per
    process: a fresh ``plugin.py`` gets fresh siblings, and within one load
    every caller shares one module object, so classes keep one identity.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(anchor)), filename)
    caller = sys._getframe(1).f_globals
    loaded = caller.setdefault("__plugin_siblings__", {})
    module = loaded.get(path)
    if module is None:
        folder = os.path.basename(os.path.dirname(path))
        name = f"{folder}.{os.path.splitext(filename)[0]}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        # The sibling's own load_sibling calls share this load's modules.
        module.__dict__["__plugin_siblings__"] = loaded
        loaded[path] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del loaded[path]
            raise
    return module


class Plugin:
    def __init__(self):
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.global_rerun_scope: Literal["fragment", "app"] = "fragment"
        self.rerun_on_err = True
        self.max_retries = 3
        try:
            self.file = self.file if self.file is not None else __file__
            self.logger.debug(
                f"Determined plugin file path: {self.file}", file=self.file
            )
            self.default_path = False
        except AttributeError:
            self.file = __file__
            self.logger.warning(
                f"Unable to determine plugin file path, use default {self.file}",
                file=self.file,
            )
            self.default_path = True

        if self.__class__ is Plugin:
            self.logger.error("Cannot instantiate base Plugin class")
            raise RuntimeError("Base Plugin class cannot be instantiated")

        self._load_config()

    def _load_config(self):
        """Robust config loading with project-level fallback"""
        # Get project root relative to this file location
        project_root = Path(__file__).resolve().parent
        plugin_dir = (
            project_root / "plugins" / self.__class__.__name__
            if self.default_path
            else Path(os.path.dirname(self.file)).resolve()
        )
        config_dir = plugin_dir / "configs"
        config_path = config_dir / "config.toml"

        # Default values
        default_name = (
            self.__class__.__name__
            if self.default_path
            else os.path.basename(os.path.dirname(self.file))
        )
        default_version = "0.0.0"

        # Ensure directory structure exists
        config_dir.mkdir(parents=True, exist_ok=True)

        # Load or create config
        if config_path.exists():
            with open(config_path, "r") as f:
                config = toml.load(f)
            self._name = config.get("name", default_name)
            self._version = config.get("version", default_version)
            self.logger.debug(f"Loaded config from {config_path}")
        else:
            self._name = default_name
            self._version = default_version
            with open(config_path, "w") as f:
                toml.dump({"name": self._name, "version": self._version}, f)
            self.logger.info(f"Created new config at {config_path}")

        self.logger.success(
            f"Initialized plugin: {self.get_name()} v{self.get_version()}"
        )

    def get_name(self):
        return self._name

    def get_version(self):
        return self._version

    @st.fragment
    def run_notification(self, data_manager, widget_manager):
        start_time = time.time()
        retry_key = f"{self.get_name()}_retry_count"

        # Initialize retry counter in session state if not present
        if retry_key not in st.session_state:
            st.session_state[retry_key] = 0

        try:
            self.run(data_manager, widget_manager)
            # st.toast(f"✅ Plugin `{self.get_name()}` executed successfully.")
            # Reset retry count on success
            st.session_state[retry_key] = 0

        except StreamlitAPIException:
            self.logger.exception(f"Error in {self.get_name()} plugin")
            if self.rerun_on_err:
                current_retries = st.session_state[retry_key]
                if current_retries < self.max_retries:
                    st.session_state[retry_key] += 1
                    st.toast(
                        f"❌ Error in {self.get_name()} plugin (retry {current_retries + 1}/{self.max_retries}), rerunning..."
                    )
                    st.rerun()
                else:
                    st.toast(
                        f"❌ {self.get_name()} failed after {self.max_retries} reruns"
                    )
                    self.logger.exception(
                        f"{self.get_name()} plugin failed after {self.max_retries} reruns"
                    )
                    st.session_state[retry_key] = 0  # Reset for future use
            else:
                st.toast(f"❌ Error in {self.get_name()} plugin")
                self.logger.exception(f"{self.get_name()} plugin error")

        finally:
            st.toast(
                f"🚀 Plugin `{self.get_name()}` executed in {time.time() - start_time: .2f} sec."
            )

    def run(self, data_manager, widget_manager):
        pass

    def create_widget(
        self,
        widget_manager,
        widget_type,
        widget_name,
        default_value=None,
        value_param="value",
        args=(),
        kwargs=None,
        value_serializer=lambda x: x,
        value_deserializer=lambda x: x,
        rerun_scope: Literal["app", "fragment"] | None = None,
        ui_key_suffix: str = "",
    ):
        """
        Universal widget creator with state persistence.

        This function creates a widget of the specified type and persists its state.
        It loads the saved state, prepares the widget parameters, creates the widget,
        serializes and saves the widget value if it changes, and finally returns the widget value.

        Parameters
        ----------
        - widget_manager: An instance of the widget manager responsible for state persistence.
        - widget_type: The type of the widget to be created.
        - widget_name: The name of the widget.
        - default_value: The default value of the widget.
        - value_param (optional): The parameter name for the widget value. Default is 'value'.
        - args (optional): Additional positional arguments for the widget creation. Default is an empty tuple.
        - kwargs (optional): Additional keyword arguments for the widget creation. Default is None, treated as an empty dictionary.
        - value_serializer (optional): A function to serialize the widget value. Default is a lambda function that returns the input value.
        - value_deserializer (optional): A function to deserialize the saved widget value. Default is a lambda function that returns the input value.

        Returns
        -------
        The created and potentially updated widget value.
        """
        kwargs = {} if kwargs is None else kwargs
        local_rerun_scope: Literal["app", "fragment"] = (
            rerun_scope if rerun_scope is not None else self.global_rerun_scope
        )
        # ``widget_name`` remains the stable persistence key. ``ui_key_suffix``
        # may rotate Streamlit's frontend identity when an external control
        # (for example, a hardware preset) replaces the widget value.
        full_key = f"{self.get_name()}_{widget_name}{ui_key_suffix}"

        # Load and deserialize saved state
        saved_raw_value, _is_new_object = widget_manager.load_widget_state(
            self.get_name(), widget_name, default_value
        )
        saved_value = value_deserializer(saved_raw_value)

        # Prepare widget parameters
        if value_param is not None:
            widget_params = {
                **kwargs,
                value_param: saved_value,
                "key": full_key,
            }
        else:
            widget_params = {**kwargs, "key": full_key}
        # Create widget
        try:
            widget_value = widget_type(*args, **widget_params)
        except:  # noqa: E722
            widget_params = {**kwargs, "key": full_key}
            widget_value = widget_type(*args, **widget_params)

        # Serialize and save if value changes
        serialized_value = value_serializer(widget_value)
        if serialized_value != saved_raw_value:
            widget_manager.save_widget_state(
                self.get_name(), widget_name, serialized_value
            )
            # An ``on_change="ignore"`` widget reaches Python only on a rerun
            # something else started -- very often a button click. Rerunning
            # again here would drop that click (a button is True for one run
            # only), and nothing on screen depends on such a widget, which is
            # the reason it was made to ignore changes in the first place.
            if (
                local_rerun_scope == "app"
                and kwargs.get("on_change") != "ignore"
            ):
                st.rerun(scope=local_rerun_scope)

        return widget_value

    def create_file_input(
        self, widget_manager, widget_name, label, **file_input_kwargs
    ):
        """``utils.file_input`` whose server-folder choice is persisted.

        ``file_input`` is built from raw Streamlit widgets, so on its own
        the chosen file never reaches the WidgetManager and a snapshot
        restores everything but it. Here the folder and the file are stored
        under ``widget_name`` as ``{"folder", "path"}`` and seeded back into
        the raw keys when those are missing, which is the case after a
        snapshot load: the keys carry the plugin prefix, so
        ``WidgetManager.import_state`` clears them with the rest. A browser
        upload has no path to keep and is not persisted. A stored file that
        is gone is not seeded (the picker starts empty).
        """
        from utils import file_input

        key = f"{self.get_name()}_{widget_name}"
        # The store itself, not ``load_widget_state``: its read-through
        # cache keeps the first value it returned, and the choice saved
        # below would then look unsaved on every later run.
        saved = st.session_state.widget_states.get(self.get_name(), {}).get(
            widget_name
        )
        if (
            isinstance(saved, dict)
            and f"{key}__selectbox" not in st.session_state
            and f"{key}__multiselect" not in st.session_state
            and saved.get("path")
            and os.path.isfile(saved["path"])
        ):
            st.session_state[f"{key}__source_is_upload"] = False
            st.session_state[f"{key}__folder"] = saved["folder"]
            st.session_state[f"{key}__selectbox"] = saved["path"]

        chosen = file_input(label, key=key, **file_input_kwargs)

        if not st.session_state.get(f"{key}__source_is_upload"):
            current = {
                "folder": st.session_state.get(f"{key}__folder"),
                "path": getattr(chosen, "path", None),
            }
            if current != saved:
                widget_manager.save_widget_state(
                    self.get_name(), widget_name, current
                )
        return chosen
