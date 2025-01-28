from typing import List
import importlib
import subprocess
from git import Repo
from loguru import logger
from pathlib import Path
from plugin import Plugin

class PluginManager:
    def __init__(self, plugins_dir="plugins"):
        """
        Initialize a PluginManager instance.

        This class is responsible for managing and loading plugins from a specified directory.

        Parameters:
        -----------
            plugins_dir (str): The path to the directory where plugins are stored. Defaults to "plugins".

        Returns:
        --------
            None
        """
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.plugins_dir = Path(plugins_dir)
        self.plugins = {}

        self.logger.info(f"Initializing PluginManager", plugins_dir=str(self.plugins_dir))
        self.logger.info(f"Pluginsdir: {self.plugins_dir}")
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug("Created plugins directory if not exists")

    def load_plugins(self):
        """
        Load and register all available plugins from the specified directory.

        This method iterates through the plugins directory, identifies valid plugin directories,
        loads the plugin module, and registers the plugin instances in the PluginManager.

        Returns:
        --------
            None

        Raises:
        -------
            Exception: If an error occurs during the plugin loading process.
        """
        self.logger.info("Starting plugin loading process")
        self.plugins.clear()
        self.logger.debug("Cleared existing plugins cache")

        plugin_count = 0
        for entry in self.plugins_dir.iterdir():
            plugin_name = entry.name
            plugin_path = entry.resolve()

            self.logger.debug("Processing directory entry", entry=plugin_name)

            if not entry.is_dir():
                self.logger.warning("Skipping non-directory plugin", entry=plugin_name)
                continue

            module_path = plugin_path / "plugin.py"
            if not module_path.exists():
                self.logger.warning("Plugin directory missing plugin.py", plugin=plugin_name)
                continue

            try:
                self.logger.info(f"Loading plugin {plugin_name}", plugin=plugin_name)
                spec = importlib.util.spec_from_file_location(plugin_name, str(module_path))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.logger.debug(f"Module {module.__name__} loaded successfully", module=module.__name__)

                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, Plugin) and 
                        attr != Plugin):
                        self.logger.debug(f"Found Plugin subclass {attr.__name__}", class_name=attr.__name__)
                        plugin = attr()
                        plugin_name = plugin.get_name()

                        if plugin_name in self.plugins:
                            self.logger.warning("Duplicate plugin name detected", 
                                             existing=plugin_name, new=attr.__name__)
                            continue

                        self.plugins[plugin_name] = plugin
                        plugin_count += 1
                        self.logger.success("Plugin registered successfully", 
                                          plugin=plugin_name, version=plugin.get_version())

            except Exception as e:
                self.logger.opt(exception=True).error("Plugin loading failed", 
                    plugin=plugin_name, error=str(e))

        self.logger.info("Completed plugin loading", total_plugins=plugin_count)

    def install_plugin(self, repo_url: str) -> bool:
        """
        Install a new plugin from a given repository URL.

        This function clones the repository into the plugins directory, installs any required dependencies,
        and reloads the plugin manager to make the new plugin available.

        Parameters:
        -----------
            repo_url (str): The URL of the Git repository containing the plugin.

        Returns:
        --------
            bool: True if the plugin installation was successful, False otherwise.

        Raises:
        -------
            subprocess.CalledProcessError: If there is an error during the dependency installation process.
        """
        self.logger.info("Starting plugin installation", repo_url=repo_url)
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        plugin_dir = self.plugins_dir / repo_name

        if plugin_dir.exists():
            self.logger.warning("Plugin directory already exists", 
                              plugin=repo_name, path=str(plugin_dir))
            return False

        try:
            self.logger.debug("Cloning repository", repo=repo_name)
            Repo.clone_from(repo_url, str(plugin_dir))
            self.logger.success("Repository cloned successfully", path=str(plugin_dir))

            req_file = plugin_dir / "requirements.txt"
            if req_file.exists():
                self.logger.info("Installing dependencies", requirements=str(req_file))
                result = subprocess.run(
                    ["pip", "install", "-r", str(req_file)],
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:
                    self.logger.success("Dependencies installed successfully",
                                      output=result.stdout)
                else:
                    self.logger.error("Dependency installation failed",
                                   stderr=result.stderr,
                                   returncode=result.returncode)
                    raise subprocess.CalledProcessError(
                        result.returncode, 
                        result.args,
                        output=result.stdout,
                        stderr=result.stderr
                    )

            self.logger.debug("Reloading plugins after installation")
            self.load_plugins()
            return True

        except Exception as e:
            self.logger.opt(exception=True).critical("Plugin installation failed",
                repo_url=repo_url, error=str(e))
            if plugin_dir.exists():
                self.logger.debug("Cleaning up failed installation", path=str(plugin_dir))
                import shutil
                shutil.rmtree(plugin_dir)
            return False

    def get_plugins(self) -> List[Plugin]:
        """
        Retrieve a list of all registered plugins.

        This method returns a list of all the plugins currently loaded and registered in the PluginManager.

        Returns:
        --------
            List[Plugin]: A list of Plugin instances. Each Plugin instance represents a loaded and registered plugin.
        """
        self.logger.debug("Retrieving registered plugins", count=len(self.plugins))
        return list(self.plugins.values())