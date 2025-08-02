from flask import Flask, render_template, request, send_from_directory
import os
import pandas as pd
import numpy as np
import json
from src.utils.main_utils import MainUtils
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

#landing page 
@app.route('/')
def landing():
    return render_template('index.html')

#Load top features and means
with open('artifacts/top_features.json') as f:
    feature_info = json.load(f)
top_features = feature_info['top_features']
feature_means = feature_info['feature_means']

#Load preprocessor and model
preprocessor = MainUtils().load_object('artifacts/preprocessor.pkl')
model = MainUtils().load_object('artifacts/model.pkl')
all_features = preprocessor['numerical_cols'] + preprocessor["categorical_cols"]

@app.route('/form', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        #Get user input for top features
        user_input = {f: float(request.form[f]) for f in top_features}
        
        #fill in the rest with mean values
        fill_input = {f: user_input.get(f, feature_means[f]) for f in all_features}
        
        #Convert to DataFrame
        input_df = pd.DataFrame([fill_input])
        
        #Numeric and categorical columns
        numerical_cols = preprocessor['numerical_cols']
        categorical_cols = preprocessor['categorical_cols']
        
        #Numerical pipline 
        X_num = preprocessor['numerical_pipeline'].transform(input_df[numerical_cols])
        
        #Categorical encoding
        X_cat = pd.get_dummies(input_df[categorical_cols], drop_first=True)
        
        #Align columns
        X_cat = X_cat.reindex(columns=preprocessor['categorical_cols'], fill_value= np.nan)
        
        #Combine
        X_processed = np.hstack([X_num, X_cat.values])
        
        #predict
        prediction = model.predict(X_processed)[0]
    return render_template('form.html', top_features = top_features, prediction= prediction)

@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    output_path = None
    if request.method == 'POST':
        file = request.files['file']
        upload_dir = os.path.join('artifacts', 'predictions_artifacts')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)

        pipeline = PredictPipeline()
        output_path = pipeline.predict_from_csv(file_path)

    return render_template('upload.html', output_path=output_path)



@app.route('/download/<filename>')
def download_file(filename):
    predictions_dir = os.path.join('artifacts', 'predictions')
    return send_from_directory(predictions_dir, filename, as_attachment=True, mimetype='text/csv')

if __name__ == '__main__':
    app.run(debug=True)