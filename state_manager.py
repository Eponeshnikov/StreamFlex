import json
import os
import pickle
import time
from datetime import datetime

import streamlit as st
from loguru import logger

# Names reserved for the autosave. A leading underscore keeps them out of
# `list_snapshots`, so they never show up in the manual snapshot picker.
AUTOSAVE_NAME = "_autosave"
AUTOSAVE_WIDGETS_NAME = "_autosave.widgets"
AUTOSAVE_SETTINGS_FILE = ".autosave.json"

# The data bus is pickled at most this often. Widgets are written on every
# run (kilobytes); `shared_data` carries whole pipeline payloads and can run
# to gigabytes, so it is throttled and backs off further when a write turns
# out to be slow — see `StateManager.autosave`.
AUTOSAVE_DATA_INTERVAL_S = 300
AUTOSAVE_BACKOFF_FACTOR = 20


class StateManager:
    def __init__(self, snapshots_dir="snapshots"):
        """
        Initialize a StateManager instance with a specified snapshots directory.

        Parameters:
        -----------
            snapshots_dir (str): The directory where snapshots will be stored. Defaults to "snapshots".

        Returns:
        --------
            None
        """
        self.snapshots_dir = snapshots_dir
        os.makedirs(self.snapshots_dir, exist_ok=True)
        logger.info(
            "Initializing StateManager with snapshots directory: '{}'",
            self.snapshots_dir,
        )

    def save_snapshot(
        self, snapshot_name, data_mgr, widget_mgr, selected_plugins
    ):
        """
        Save current application state to a named snapshot.

        Parameters:
        -----------
            snapshot_name (str): The name of the snapshot to be created.
            data_mgr (DataManager): An instance of the DataManager class responsible for managing data.
            widget_mgr (WidgetManager): An instance of the WidgetManager class responsible for managing widgets.
            selected_plugins (list): A list of selected plugin names.

        Returns:
        --------
            bool: True if the snapshot is saved successfully, False otherwise.
        """
        try:
            # Prepare snapshot data
            creation_time = datetime.now().astimezone()
            snapshot_data = {
                "metadata": {
                    "created_at": creation_time,
                    "selected_plugins": selected_plugins,
                },
                "data": data_mgr.export_state(),
                "widgets": widget_mgr.export_state(),
            }

            logger.debug(
                "Preparing snapshot '{}' with {} plugins (created at {})",
                snapshot_name,
                len(selected_plugins),
                creation_time.isoformat(),
            )

            # Save to file
            snapshot_path = self._snapshot_path(snapshot_name)
            self._write_pickle(snapshot_path, snapshot_data)

            # Log success details
            file_size = os.path.getsize(snapshot_path)
            logger.success(
                "Saved snapshot '{}' successfully ({} bytes) to: {}",
                snapshot_name,
                file_size,
                snapshot_path,
            )
            return True

        except Exception as e:
            logger.exception(
                "Failed to save snapshot '{}'! Error: {}", snapshot_name, e
            )
            return False

    def load_snapshot(self, snapshot_name, data_mgr, widget_mgr):
        """
        Load application state from a named snapshot.

        Parameters:
        -----------
            snapshot_name (str): The name of the snapshot to be loaded.
            data_mgr (DataManager): An instance of the DataManager class responsible for managing data.
            widget_mgr (WidgetManager): An instance of the WidgetManager class responsible for managing widgets.

        Returns:
        --------
            list: A list of selected plugin names if the snapshot is loaded successfully, None otherwise.
        """
        try:
            st.session_state["current_value"] = {}
            snapshot_path = self._snapshot_path(snapshot_name)
            logger.debug("Attempting to load snapshot from: {}", snapshot_path)

            with open(snapshot_path, "rb") as f:
                snapshot_data = pickle.load(f)

            # Log metadata details
            metadata = snapshot_data["metadata"]
            logger.info(
                "Loading snapshot '{}' created at {} with {} plugins",
                snapshot_name,
                metadata["created_at"].isoformat(),
                len(metadata["selected_plugins"]),
            )

            # Restore state
            data_mgr.import_state(snapshot_data["data"])
            logger.debug("Restored data state from snapshot")

            widget_mgr.import_state(snapshot_data["widgets"])
            logger.debug("Restored widget state from snapshot")

            return metadata["selected_plugins"]

        except Exception as e:
            logger.exception(
                "Failed to load snapshot '{}'! Error: {}", snapshot_name, e
            )
            return None

    def list_snapshots(self):
        """
        List all available snapshots.

        This function retrieves a list of all available snapshots in the snapshots directory.
        It scans the directory for files with a .pkl extension, indicating that they are snapshot files.
        The function then extracts the snapshot names by removing the .pkl extension from each file name.

        Returns:
        --------
            list: A list of strings representing the names of available snapshots.
                If an error occurs during the listing process, an empty list is returned.
        """
        try:
            files = [
                f
                for f in os.listdir(self.snapshots_dir)
                if f.endswith(".pkl") and not f.startswith("_")
            ]
            snapshots = [f[:-4] for f in files]  # Remove .pkl extension
            logger.info(
                "Found {} snapshots in directory '{}'",
                len(snapshots),
                self.snapshots_dir,
            )
            return snapshots

        except Exception as e:
            logger.exception(
                "Failed to list snapshots! Directory: '{}' Error: {}",
                self.snapshots_dir,
                e,
            )
            return []

    def delete_snapshot(self, snapshot_name: str) -> bool:
        """
        Delete a specific snapshot.

        This function deletes a snapshot file with the given name from the snapshots directory.
        If the snapshot file exists, it is removed and a success message is logged.
        If the snapshot file does not exist, a warning message is logged.
        In case of any exceptions during the deletion process, an error message is logged.

        Parameters:
        -----------
            snapshot_name (str): The name of the snapshot to be deleted.

        Returns:
        --------
            bool: True if the snapshot is deleted successfully, False otherwise.
        """
        try:
            snapshot_path = self._snapshot_path(snapshot_name)

            if os.path.exists(snapshot_path):
                os.remove(snapshot_path)
                logger.success(
                    "Deleted snapshot '{}' from: {}",
                    snapshot_name,
                    snapshot_path,
                )
                return True

            logger.warning(
                "Snapshot '{}' not found at: {}", snapshot_name, snapshot_path
            )
            return False

        except Exception as e:
            logger.exception(
                "Failed to delete snapshot '{}'! Error: {}", snapshot_name, e
            )
            return False

    # ------------------------------------------------------------------
    # Autosave / resume
    #
    # A browser reload always lands in a brand-new session: streamlit's
    # reconnect id lives in the page's memory, so nothing the server does can
    # make a fresh tab claim the old session. `session_guard` keeps a running
    # script alive across the disconnect; this is the other half — the state
    # that run leaves in `session_state` is written to disk so the next
    # session can pick it back up.
    #
    # Two files, because the two halves cost wildly different amounts:
    #   `_autosave.widgets.pkl`  widgets + plugin selection, kilobytes, every run
    #   `_autosave.pkl`          the same plus the data bus, throttled
    # On restore the data file is loaded first and the widgets file is
    # overlaid when it is newer, so the widget you touched a second before
    # closing the tab survives even if the data snapshot is minutes older.
    # ------------------------------------------------------------------

    def _snapshot_path(self, snapshot_name):
        """Absolute path of a snapshot file (no existence check)."""
        return os.path.join(self.snapshots_dir, f"{snapshot_name}.pkl")

    def _write_pickle(self, path, payload):
        """Pickle `payload` to `path` without ever truncating a good file.

        Autosaves overwrite the same two names over and over, and one of
        those writes will eventually be interrupted (a stop, a crash, a full
        disk). Writing beside the target and renaming makes the replacement
        atomic, so a failed write leaves the previous snapshot intact instead
        of a half-written pickle that fails to load.
        """
        tmp_path = f"{path}.tmp"
        try:
            with open(tmp_path, "wb") as handle:
                pickle.dump(payload, handle)
            os.replace(tmp_path, path)
        except BaseException:
            # BaseException on purpose: a StopException mid-write must not
            # leave the scratch file behind either.
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    def autosave(
        self, data_mgr, widget_mgr, selected_plugins, *, include_data
    ):
        """Write the autosave files. Returns the seconds the write took.

        `include_data=False` writes only the widgets file. Callers decide when
        the expensive one is due; `autosave_data_due` holds that policy.
        """
        started = time.perf_counter()
        creation_time = datetime.now().astimezone()
        payload = {
            "metadata": {
                "created_at": creation_time,
                "selected_plugins": selected_plugins,
            },
            "widgets": widget_mgr.export_state(),
        }

        self._write_pickle(self._snapshot_path(AUTOSAVE_WIDGETS_NAME), payload)

        if include_data:
            payload = dict(payload, data=data_mgr.export_state())
            path = self._snapshot_path(AUTOSAVE_NAME)
            self._write_pickle(path, payload)
            duration = time.perf_counter() - started
            logger.info(
                "Autosaved session state ({:.1f} MB, {:.2f} s)",
                os.path.getsize(path) / 1024**2,
                duration,
            )
            return duration

        return time.perf_counter() - started

    def autosave_data_due(self, last_duration, last_written):
        """Whether the data half of the autosave should be rewritten now.

        The interval is the floor; a write that took long pushes the next one
        proportionally further out, so a multi-gigabyte data bus is not
        pickled every five minutes just because a widget was clicked.
        """
        if last_written is None:
            return True
        interval = max(
            AUTOSAVE_DATA_INTERVAL_S,
            AUTOSAVE_BACKOFF_FACTOR * (last_duration or 0.0),
        )
        return (time.time() - last_written) >= interval

    def autosave_status(self):
        """`{name: (mtime, size_bytes)}` for whichever autosave files exist."""
        status = {}
        for name in (AUTOSAVE_NAME, AUTOSAVE_WIDGETS_NAME):
            path = self._snapshot_path(name)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            status[name] = (stat.st_mtime, stat.st_size)
        return status

    def restore_autosave(self, data_mgr, widget_mgr):
        """Restore the autosaved state. Returns the plugin selection or None.

        Mirrors `load_snapshot`: the `current_value` cache has to be dropped
        along with the widget states, or a stale cached value wins over the
        restored one.
        """
        status = self.autosave_status()
        if not status:
            return None

        try:
            st.session_state["current_value"] = {}
            selected_plugins = None

            data_entry = status.get(AUTOSAVE_NAME)
            if data_entry:
                with open(self._snapshot_path(AUTOSAVE_NAME), "rb") as handle:
                    payload = pickle.load(handle)
                data_mgr.import_state(payload.get("data", {}))
                widget_mgr.import_state(payload["widgets"])
                selected_plugins = payload["metadata"]["selected_plugins"]
                logger.info(
                    "Restored autosaved data bus and widgets from {}",
                    datetime.fromtimestamp(data_entry[0]).astimezone(),
                )

            widgets_entry = status.get(AUTOSAVE_WIDGETS_NAME)
            if widgets_entry and (
                not data_entry or widgets_entry[0] > data_entry[0]
            ):
                path = self._snapshot_path(AUTOSAVE_WIDGETS_NAME)
                with open(path, "rb") as handle:
                    payload = pickle.load(handle)
                widget_mgr.import_state(payload["widgets"])
                selected_plugins = payload["metadata"]["selected_plugins"]
                logger.info(
                    "Overlaid newer autosaved widgets from {}",
                    datetime.fromtimestamp(widgets_entry[0]).astimezone(),
                )

            return selected_plugins

        except Exception as e:
            logger.exception("Failed to restore autosave! Error: {}", e)
            return None

    def clear_autosave(self):
        """Delete both autosave files. Returns True if anything was removed."""
        removed = False
        for name in (AUTOSAVE_NAME, AUTOSAVE_WIDGETS_NAME):
            path = self._snapshot_path(name)
            if os.path.exists(path):
                try:
                    os.remove(path)
                    removed = True
                except OSError as e:
                    logger.warning("Could not delete {}: {}", path, e)
        if removed:
            logger.info("Cleared autosave files")
        return removed

    def autosave_settings(self):
        """Read the autosave preferences. Survives restarts, unlike state."""
        path = os.path.join(self.snapshots_dir, AUTOSAVE_SETTINGS_FILE)
        settings = {"enabled": True}
        try:
            with open(path) as handle:
                stored = json.load(handle)
            if isinstance(stored, dict):
                settings.update(stored)
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as e:
            logger.warning("Ignoring unreadable autosave settings: {}", e)
        return settings

    def save_autosave_settings(self, settings):
        """Persist the autosave preferences."""
        path = os.path.join(self.snapshots_dir, AUTOSAVE_SETTINGS_FILE)
        try:
            with open(path, "w") as handle:
                json.dump(settings, handle)
        except OSError as e:
            logger.warning("Could not save autosave settings: {}", e)
