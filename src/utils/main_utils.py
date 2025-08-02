import sys
import os
import pickle
import yaml
import pandas as pd
from typing import Dict, Tuple
from src.constant import *
from src.exception import CustomException
from src.logger import logging

class MainUtils:
    def __init__(self) -> None:
         pass
    def read_yaml_file(self, filename: str) -> Dict:
        """Read a YAML file and return its contents as a dictionary."""
        
        try:
            with open(filename, 'rb') as yaml_file:
                return yaml.safe_load(yaml_file)
        except  Exception as e:
            logging.error(f'Error reading YAML file: {filename}')
            raise CustomException(e, sys) from e
        
    def read_schema_config_file(self) -> Dict:
        """Read a schema.YAML config from the config directory."""
        
        try:
            schema_config = self.read_yaml_file(os.path.join('config', 'schema.yaml'))
            return schema_config
        except Exception as e :
            logging.error("Error reading schema config file")
            raise CustomException(e, sys) from e
        
    @staticmethod
    def save_object(file_path: str, obj: object) -> None:
        """Save any Python object to a file using pickle."""
        logging.info(f'Saving object to {file_path}')
        
        try:
            with open(file_path, 'wb') as file_obj:
                pickle.dump(obj, file_obj)
            logging.info(f'Successfully saved object to {file_path}')
        except Exception as e:
            logging.error(f'Faild to save object to {file_path}')
            raise CustomException(e, sys) from e
    
    @staticmethod
    def load_object(file_path: str) -> None:
        """Load any Pickled Python object from a file."""
        logging.info(f'Loading object from {file_path}')
        
        try:
            with open(file_path, 'rb') as file_obj:
                obj = pickle.load(file_obj)
            logging.info(f'Successfully loaded object from {file_path}')
            return obj
        except Exception as e:
            logging.error(f'Faild to load object from {file_path}')
            raise CustomException(e, sys) from e