import os
import json
import logging
from datetime import datetime

class StateManager:
    def __init__(self, snapshots_dir="snapshots"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.snapshots_dir = snapshots_dir
        os.makedirs(self.snapshots_dir, exist_ok=True)

    def save_snapshot(self, snapshot_name, data_mgr, config_mgr, widget_mgr, selected_plugins):
        try:
            snapshot_data = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "selected_plugins": selected_plugins
                },
                "data": data_mgr.export_state(),
                "configs": config_mgr.export_state(),
                "widgets": widget_mgr.export_state()
            }
            
            snapshot_path = os.path.join(self.snapshots_dir, f"{snapshot_name}.json")
            with open(snapshot_path, 'w') as f:
                json.dump(snapshot_data, f, indent=2)
            
            self.logger.info(f"Saved snapshot: {snapshot_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving snapshot: {str(e)}")
            return False

    def load_snapshot(self, snapshot_name, data_mgr, config_mgr, widget_mgr):
        try:
            snapshot_path = os.path.join(self.snapshots_dir, f"{snapshot_name}.json")
            with open(snapshot_path, 'r') as f:
                snapshot_data = json.load(f)
            
            data_mgr.import_state(snapshot_data["data"])
            config_mgr.import_state(snapshot_data["configs"])
            widget_mgr.import_state(snapshot_data["widgets"])
            
            self.logger.info(f"Loaded snapshot: {snapshot_name}")
            return snapshot_data["metadata"]["selected_plugins"]
        except Exception as e:
            self.logger.error(f"Error loading snapshot: {str(e)}")
            return None

    def list_snapshots(self):
        try:
            return [f.replace(".json", "") for f in os.listdir(self.snapshots_dir)
                    if f.endswith(".json")]
        except Exception as e:
            self.logger.error(f"Error listing snapshots: {str(e)}")
            return []

    def delete_snapshot(self, snapshot_name):
        try:
            snapshot_path = os.path.join(self.snapshots_dir, f"{snapshot_name}.json")
            if os.path.exists(snapshot_path):
                os.remove(snapshot_path)
                self.logger.info(f"Deleted snapshot: {snapshot_name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting snapshot: {str(e)}")
            return False