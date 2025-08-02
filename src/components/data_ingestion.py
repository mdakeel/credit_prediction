import os 
import pandas as pd
import sys 

from src.constant import APPLICATION_TRAIN_PATH
from src.logger import logging
from dataclasses import dataclass #remove init , if we have dataclass , we dont need to writhe init
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    artifacts_folder: str = 'artifacts'
    train_file_name: str = 'application_train.csv'
    
class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")
        
        try:
            os.makedirs(self.config.artifacts_folder, exist_ok=True)
            logging.info(f'Artifacts folder created at: {self.config.artifacts_folder}')
            destination_path = os.path.join(self.config.artifacts_folder, self.config.train_file_name)
            logging.info(f'Copying data from {APPLICATION_TRAIN_PATH} to {destination_path}')
            df = pd.read_csv(APPLICATION_TRAIN_PATH)
            df.to_csv(destination_path, index=False)
            logging.info(f'Data saved to {destination_path}')
            logging.info("Data Ingestion completed successfully")
        except Exception as e:
            logging.error(f'Error occured during data ingestion: {str(e)}')
            raise CustomException(e, sys)