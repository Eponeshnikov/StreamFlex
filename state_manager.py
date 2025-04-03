import os
import pickle
from datetime import datetime

from loguru import logger


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
            creation_time = datetime.now()
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
            snapshot_path = os.path.join(
                self.snapshots_dir, f"{snapshot_name}.pkl"
            )
            with open(snapshot_path, "wb") as f:
                pickle.dump(snapshot_data, f)

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
            snapshot_path = os.path.join(
                self.snapshots_dir, f"{snapshot_name}.pkl"
            )
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
                f for f in os.listdir(self.snapshots_dir) if f.endswith(".pkl")
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
            snapshot_path = os.path.join(
                self.snapshots_dir, f"{snapshot_name}.pkl"
            )

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
