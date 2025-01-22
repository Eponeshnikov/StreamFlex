import os
import importlib
import subprocess
import logging
from git import Repo
from plugin import Plugin

class PluginManager:
    def __init__(self, plugins_dir="plugins"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.plugins_dir = plugins_dir
        self.plugins = {}
        os.makedirs(self.plugins_dir, exist_ok=True)

    def load_plugins(self):
        self.plugins.clear()
        for plugin_name in os.listdir(self.plugins_dir):
            plugin_path = os.path.join(self.plugins_dir, plugin_name)
            if os.path.isdir(plugin_path):
                try:
                    module_path = os.path.join(plugin_path, "plugin.py")
                    spec = importlib.util.spec_from_file_location(plugin_name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    for attr in dir(module):
                        cls = getattr(module, attr)
                        if (isinstance(cls, type) and 
                            issubclass(cls, Plugin) and 
                            cls != Plugin):
                            plugin = cls()
                            self.plugins[plugin.get_name()] = plugin
                            self.logger.info(f"Loaded plugin: {plugin.get_name()}")
                except Exception as e:
                    self.logger.error(f"Error loading plugin {plugin_name}: {str(e)}")

    def install_plugin(self, repo_url):
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        plugin_dir = os.path.join(self.plugins_dir, repo_name)
        
        if os.path.exists(plugin_dir):
            self.logger.warning(f"Plugin already exists: {repo_name}")
            return

        try:
            Repo.clone_from(repo_url, plugin_dir)
            self.logger.info(f"Cloned repository: {repo_name}")
            
            # Install requirements
            req_file = os.path.join(plugin_dir, "requirements.txt")
            if os.path.exists(req_file):
                subprocess.run(["pip", "install", "-r", req_file], check=True)
            
            self.load_plugins()
        except Exception as e:
            self.logger.error(f"Error installing plugin: {str(e)}")

    def get_plugins(self):
        return list(self.plugins.values())