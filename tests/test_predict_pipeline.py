import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline.predict_pipeline import PredictPipeline

if __name__ == '__main__':
    print('Testing Batch Predict Pipeline...')
    #Make sure 'notebooks/data/application_train.csv' exists in your project root 
    pipeline = PredictPipeline()
    output_path = pipeline.predict_from_csv('notebooks/data/application_train.csv')
    print(f'Prediction batch file created at: {output_path}')
    print('Predict pipeline test completed successfully.')