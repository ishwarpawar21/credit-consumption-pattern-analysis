from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models
with open('../train_model/models.pkl', 'rb') as f:
    models = pickle.load(f)
    
sarima_model = models['sarima_model']
prophet_model = models['prophet_model']
random_forest_model = models['random_forest_model']
gradient_boosting_model = models['gradient_boosting_model']
lstm_model = models['lstm_model']

def create_dataset(dataset, look_back=1):
    X = []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
    return np.array(X)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_name = data['model']
    input_data = pd.DataFrame(data['data'])
    input_data['Timestamp'] = pd.to_datetime(input_data['Timestamp'])
    input_data.set_index('Timestamp', inplace=True)
    input_data['WeekOfYear'] = input_data.index.isocalendar().week
    input_data['Year'] = input_data.index.year

    if model_name == 'sarima':
        forecast = sarima_model.get_forecast(steps=1)
        prediction = forecast.predicted_mean.iloc[0]
        
    elif model_name == 'prophet':
        future_dates = prophet_model.make_future_dataframe(periods=1, freq='W')
        forecast = prophet_model.predict(future_dates)
        prediction = forecast['yhat'].iloc[-1]
        
    elif model_name == 'random_forest':
        features = input_data[['WeekOfYear', 'Year']]
        prediction = random_forest_model.predict(features)[-1]
        
    elif model_name == 'gradient_boosting':
        features = input_data[['WeekOfYear', 'Year']]
        prediction = gradient_boosting_model.predict(features)[-1]
        
    elif model_name == 'lstm':
        # Directly use the amount values without scaling
        look_back = 1
        raw_data = input_data['Amount'].values.reshape(-1, 1)
        testX = create_dataset(raw_data, look_back)
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        lstm_pred = lstm_model.predict(testX)
        prediction = lstm_pred[-1][0]  # Directly use the prediction without inverse scaling
        
    else:
        return jsonify({"error": "Invalid model name"}), 400

    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)
