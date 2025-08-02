import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from src.utils.main_utils import MainUtils

@dataclass

class DataTransformationConfig:
    artifacts_dir = os.path.join('artifacts')
    ingested_train_path: str = os.path.join(artifacts_dir, 'application_train.csv')
    transformed_trian_file_path: str = os.path.join(artifacts_dir, 'train.npy')
    transformed_test_file_path: str = os.path.join(artifacts_dir, 'test.npy') #npy is binary form
    transformed_train_csv_path: str = os.path.join(artifacts_dir, 'transformed_train.csv') 
    transformed_test_csv_path: str = os.path.join(artifacts_dir, 'transformed_test.csv') 
    transformed_object_file_path: str = os.path.join(artifacts_dir, 'preprocessor.pkl') 
    
class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.utils = MainUtils()
        
    def initiate_data_transformation(self):
        logging.info("Data Transformation Started")
        
        try:
            df = pd.read_csv(self.config.ingested_train_path)
            logging.info(f'Data loaded from {self.config.ingested_train_path}')
            
            if 'SK_ID_CURR' in df.columns:
                df.drop(columns=['SK_ID_CURR'], inplace=True)
                logging.info("Dropped 'SK_ID_CURR' column from the datasete")
                
            X = df.drop(columns=['TARGET'])
            y = df['TARGET']
            logging.info('Separated features and target variables') 
            
            #train-test-split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            logging.info("Train-test split completed")
            logging.info(f"X_train_shape: {X_train.shape}, X_test_shape: {X_test.shape}")
            
            #separat categorical and numerical data
            categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
            numerical_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
            logging.info(f"Categorical Columns: {categorical_cols}")
            logging.info(f"Numerical Columns: {numerical_cols}")
            
            #Numerical pipeline
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy= 'median')),
                ('scaler', RobustScaler())
            ])
            
            #fit and transforming data
            X_train_num = numerical_pipeline.fit_transform(X_train[numerical_cols])
            X_test_num = numerical_pipeline.transform(X_test[numerical_cols])
            logging.info("Numerical features scaled")
            
            #Categorical pipeline > encoding
                        
            #
            X_train_cat = pd.get_dummies(X_train[categorical_cols], drop_first=True)
            X_test_cat = pd.get_dummies(X_test[categorical_cols], drop_first=True)
            logging.info("Categorical features one hot encoded")
            
            #Align the columns of train and test sets
            X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join='left', axis=1, fill_value=0)
            logging.info("Aligned categorical columns between train and test sets")
            
            #Combine numerical and categorical features
            X_train_processed = np.hstack([X_train_num, X_train_cat.values])
            X_test_processed = np.hstack([X_test_num, X_test_cat.values])
            logging.info("Combined numerical and categorical features")
            
            #save the processed data
            np.save(self.config.transformed_trian_file_path, {'X': X_train_processed, 'y': y_train.values })
            np.save(self.config.transformed_test_file_path, {'X': X_test_processed, 'y': y_test.values })
            
            #Save the processed data as CSV > Option no need to save data to csv we save data only in one file
            all_feature_names = numerical_cols + list(X_train_cat.columns)
            train_df_out = pd.DataFrame(X_train_processed, columns=all_feature_names)
            test_df_out = pd.DataFrame(X_test_processed, columns=all_feature_names)
            train_df_out['TARGET'] = y_train.values
            test_df_out['TARGET'] = y_test.values
            train_df_out.to_csv(self.config.transformed_train_csv_path, index=True)
            test_df_out.to_csv(self.config.transformed_test_csv_path, index=True)
            
            #Save the processor object
            preprocessor = {
                'numerical_pipeline': numerical_pipeline,
                'categorical_cols': categorical_cols,
                'numerical_cols': numerical_cols,
                'categorical_cols': X_train_cat.columns.tolist()
            }
            self.utils.save_object(self.config.transformed_object_file_path, preprocessor)
            logging.info(f"Preprocessor object saved to {self.config.transformed_object_file_path}")
            
            logging.info("Data transformation completed successfully")
            
            return (
                self.config.transformed_trian_file_path,
                self.config.transformed_test_file_path,
                self.config.transformed_object_file_path,
                self.config.transformed_train_csv_path,
                self.config.transformed_test_csv_path
            )
        
        except Exception as e:
            logging.error(f'Error occured during data transformation: {str(e)}') 
            raise CustomException(e, sys)