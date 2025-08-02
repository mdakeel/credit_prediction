import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components.data_ingestion import DataIngestion

if __name__ == '__main__':
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()
    print("Data ingestion process finished.")