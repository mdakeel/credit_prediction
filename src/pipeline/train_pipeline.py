import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
from src.utils.extract_top_features import extract_and_save_top_features

class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            ingestion = DataIngestion()
            ingestion.initiate_data_ingestion()
            logging.info("✅ Data ingestion completed.")
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self):
        try:
            transformer = DataTransformation()
            train_arr_path, test_arr_path, preprocessor, train_csv, test_csv = transformer.initiate_data_transformation()
            logging.info(f"✅ Data transformation completed. Train: {train_arr_path}, Test: {test_arr_path}")
            return train_arr_path, test_arr_path
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_training(self, train_arr_path, test_arr_path):
        try:
            train_data = np.load(train_arr_path, allow_pickle=True).item()
            test_data = np.load(test_arr_path, allow_pickle=True).item()
            x_train, y_train = train_data['X'], train_data['y']
            x_test, y_test = test_data['X'], test_data['y']

            # Optional: use smaller subset for faster training
            x_train, y_train = x_train[:1000], y_train[:1000]
            x_test, y_test = x_test[:200], y_test[:200]

            trainer = ModelTrainer()
            model_path = trainer.initiate_model_trainer(x_train, y_train, x_test, y_test)
            logging.info(f"✅Model training completed. Model saved at: {model_path}")
            return model_path
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            print("Starting data ingestion...")
            self.start_data_ingestion()
            print("Starting data transformation...")
            train_arr_path, test_arr_path = self.start_data_transformation()
            print("Starting model trining...")
            model_path = self.start_model_training(train_arr_path, test_arr_path)
            print("Extracting top features...")
            extract_and_save_top_features()
            print("✅Training pipeline completed. Model saved at: ", model_path)
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    TrainingPipeline().run_pipeline()
