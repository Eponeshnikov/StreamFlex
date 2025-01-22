import logging
from abc import ABC, abstractmethod

class Plugin(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def run(self, data_manager, config_manager, widget_manager):
        pass
    
    @abstractmethod
    def get_name(self):
        pass
    
    @abstractmethod
    def load_config(self, config_path):
        pass