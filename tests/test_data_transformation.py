import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components.data_transformation import DataTransformation

if __name__ == '__main__':
    transformer = DataTransformation()
    print("Starting data transformation process...")
    train_path, test_path, preprocessor, train_csv, test_csv = transformer.initiate_data_transformation()
    print(f"Transformation Trian data saved at: {train_path}")
    print(f"Transformation Test data saved at: {test_path}")
    print(f"Transformation Object data saved at: {preprocessor}")
    print(f"Transformation Trian csv data saved at: {train_csv}")
    print(f"Transformation Test csv data saved at: {test_csv}")
    print("Data Transformation process finished.")