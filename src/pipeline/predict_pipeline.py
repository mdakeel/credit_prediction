import os
import sys
import pandas as pd
import numpy as np
import json
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import MainUtils
from src.constant import TARGET_COLUMN

class PredictPipeline:
    def __init__(self):
        self.utils = MainUtils()
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.top_features_path = os.path.join('artifacts', 'top_features.json')
        
    def load_artifacts(self):
        try:
            model = self.utils.load_object(self.model_path)
            preprocessor = self.utils.load_object(self.preprocessor_path)
            with open(self.top_features_path, 'r') as f:
                features_info = json.load(f)
            top_features = features_info['top_features']
            feature_means = features_info['feature_means']
            return model, preprocessor, top_features, feature_means
        except Exception as e:
            logging.error("Error occured while loading artifacts...")
            raise CustomException("Error occured while loading artifacts...") from e
        
    def preprocess_input(self, input_df, preprocessor, feature_means):
        try:
            #Drop columns as in training
            drop_cols = ['SK_ID_CURR']
            for col in drop_cols:
                if col in input_df.columns:
                    input_df = input_df.drop(columns=[col])
                    
            #fill missing features with means using reindex for performance 
            all_features = preprocessor['numerical_cols'] + preprocessor["categorical_cols"]
            input_df = input_df.reindex(columns=all_features, fill_value = np.nan)
            input_df = input_df.fillna(feature_means)
            
            #Numeric and categorical
            numerical_cols = preprocessor['numerical_cols']
            categorical_cols = preprocessor['categorical_cols']
            X_num = preprocessor['numerical_pipeline'].transform(input_df[numerical_cols])
            X_cat = pd.get_dummies(input_df[categorical_cols], drop_first=True)
            X_cat = X_cat.reindex(columns=preprocessor['categorical_cols'], fill_value= np.nan)
            X_processed = np.hstack([X_num, X_cat.values])
            return X_processed
        except Exception as e:
            logging.error("Error in preprocessing input.")
            raise CustomException(e, sys)
        
    def predict_from_csv(self, csv_path):
        try:
            model, preprocessor, top_features, feature_means = self.load_artifacts()
            input_df = pd.read_csv(csv_path)
            
            #if unnamed: 0 exists, drop it
            if "Unnamed: 0" in input_df.columns:
                input_df = input_df.drop(columns="Unnamed: 0")
            
            X_processed = self.preprocess_input(input_df, preprocessor, feature_means)
            preds = model.predict(X_processed)
            input_df[TARGET_COLUMN] = preds
            
            #Map to labels
            target_column_mapping = {0: 'bad', 1: 'good'}
            input_df[TARGET_COLUMN] = input_df[TARGET_COLUMN].map(target_column_mapping)
            
            #print counts of each prediction
            counts = input_df[TARGET_COLUMN].value_counts()
            print(f"Prediction counts: \n{counts.to_string()}")
            logging.info(f"Prediction counts: \n{counts.to_string()}")
            
            
            #save prediction
            output_dir = os.path.join('artifacts', 'predictions')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "prediction_file.csv")
            input_df.to_csv(output_path, index=False)
            logging.info(f"Prediction saved to {output_path}")
            return output_path
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_from_dic(self, user_input_dict):
        try:
            model, preprocessor, top_features, feature_means = self.load_artifacts()
            all_features = preprocessor['numerical_cols'] + preprocessor["categorical_cols"]
            
            #fill missing features with mean 
            fill_input = {f: user_input_dict.get(f, feature_means[f]) for f in all_features}
            input_df = pd.DataFrame([fill_input])
            X_processed = self.preprocess_input(input_df, preprocessor, feature_means)
            pred = model.predict(X_processed)[0]
            return pred
        except Exception as e:
            raise CustomException(e, sys)
            
            
            
