import sys
import numpy as np
import pandas as pd
import json
from src.utils.main_utils import MainUtils
from src.logger import logging
from src.exception import CustomException

def extract_and_save_top_features(
    preprocessor_path = "artifacts/preprocessor.pkl",
    model_path = "artifacts/model.pkl",
    train_csv_path = "artifacts/transformed_train.csv",
    output_json_path = "artifacts/top_features.json",
    top_n = 10
):
    try:
        logging.info("Loading preprocessor and model objects.")
        preprocessor = MainUtils().load_object(preprocessor_path)
        model = MainUtils().load_object(model_path)
        train_df = pd.read_csv(train_csv_path)
        
        feature_names = preprocessor['numerical_cols'] + preprocessor["categorical_cols"]
        importance = model.feature_importances_
        top_indices = np.argsort(importance)[::-1][:top_n]
        top_features = [feature_names[i] for i in top_indices]
        logging.info(f"Top {top_n} features: {top_features}")
        
        feature_means = train_df[feature_names].mean().to_dict()
        
        with open(output_json_path, 'w') as f:
            json.dump({"top_features": top_features, "feature_means": feature_means}, f)
        logging.info(f"Top features and means save to : {output_json_path}")
        return top_features, feature_means
    except Exception as e:
        logging.error(f"Error occured while extracting and saving top features.  {str(e)}")
        raise CustomException(e, sys) from e