import os
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from keras.models import load_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf

# Load LSTM model if available
def load_lstm_model():
    lstm_model = None
    if os.path.exists('lstm_model.h5'):
        lstm_model = load_model('lstm_model.h5')
    else:
        st.warning("LSTM model file not found.")
    return lstm_model

import os
import streamlit as st
import pandas as pd
import numpy as np

# Load data with debugging info
@st.cache_data
def load_data():
    # Load dataset
    data = pd.read_csv('Complete_South_African_Energy_Consumption.csv')

    st.write("Original Data:")
    st.write(data.head())

    # Check if 'DateTime' or 'Building' columns exist and drop them
    if 'DateTime' in data.columns:
        st.write("Dropping 'DateTime' column")
        data = data.drop(columns=['DateTime'])
    if 'Building' in data.columns:
        st.write("Dropping 'Building' column")
        data = data.drop(columns=['Building'])

    st.write("Data After Dropping Non-Numeric Columns:")
    st.write(data.head())

    # Ensure all columns are numeric
    data = data.apply(pd.to_numeric, errors='coerce')

    st.write("Data After Conversion to Numeric:")
    st.write(data.head())

    # Drop rows with NaN values
    data = data.dropna()
    
    st.write("Data After Dropping Rows with NaN:")
    st.write(data.head())

    if data.empty:
        st.error("The dataset is empty after preprocessing. Please check your data.")
    return data

# Streamlit app layout
st.title("Debugging: Energy Consumption Forecasting")

# Load dataset
data = load_data()

# Show the final dataset
if not data.empty:
    st.write("Final Data:")
    st.write(data.head())
else:
    st.error("Final dataset is empty. Please check the dataset.")


# ARIMA Model Prediction
def arima_forecast(data, steps):
    if len(data) == 0:
        st.error("Data is empty for ARIMA prediction.")
        return np.array([])

    model = ARIMA(data['Consumption'], order=(5,1,0))  # Adjust based on your analysis
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# XGBoost Prediction
def xgboost_forecast(data):
    X = data.drop(columns=['Consumption'])
    y = data['Consumption']

    if X.empty or y.empty:
        st.error("Data is empty for XGBoost prediction.")
        return np.array([])

    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(X, y)
    return xgb_model.predict(X)

# Decision Tree Prediction
def decision_tree_forecast(data):
    X = data.drop(columns=['Consumption'])
    y = data['Consumption']

    if X.empty or y.empty:
        st.error("Data is empty for Decision Tree prediction.")
        return np.array([])

    dt_model = DecisionTreeRegressor()
    dt_model.fit(X, y)
    return dt_model.predict(X)

# Random Forest Prediction
def random_forest_forecast(data):
    X = data.drop(columns=['Consumption'])
    y = data['Consumption']

    if X.empty or y.empty:
        st.error("Data is empty for Random Forest prediction.")
        return np.array([])

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    return rf_model.predict(X)

# LSTM Model Prediction
def lstm_forecast(data, time_steps=60):
    lstm_model = load_lstm_model()
    if lstm_model is None:
        return np.array([])

    # Preprocess data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Consumption'].values.reshape(-1, 1))
    X = []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    if X.shape[0] == 0:
        st.error("Not enough data for LSTM prediction.")
        return np.array([])

    # Make predictions
    predictions = lstm_model.predict(X)
    return scaler.inverse_transform(predictions)

# Metrics Calculation
def calculate_metrics(y_true, y_pred):
    if len(y_pred) == 0:
        st.error("No predictions to calculate metrics.")
        return None, None, None

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

# Streamlit app layout
st.title("Energy Consumption Forecasting")

# Select model
model_choice = st.selectbox("Select a model", ["ARIMA", "LSTM", "XGBoost", "Decision Tree", "Random Forest"])

# Forecasting period (days)
forecast_days = st.slider("Forecasting Days", 1, 30)

# Load dataset
data = load_data()

# Show dataset
if st.checkbox("Show Data"):
    st.write(data.head())

# Run Forecast
if st.button("Predict"):
    if model_choice == "ARIMA":
        forecast = arima_forecast(data, steps=forecast_days)
        if forecast is not None and len(forecast) > 0:
            st.line_chart(forecast)
            mae, mse, r2 = calculate_metrics(data['Consumption'][:len(forecast)], forecast)
            if mae and mse and r2:
                st.write(f"ARIMA Metrics - MAE: {mae}, MSE: {mse}, R²: {r2}")
        
    elif model_choice == "LSTM":
        forecast = lstm_forecast(data, time_steps=60)
        if len(forecast) > 0:
            st.line_chart(forecast)
            mae, mse, r2 = calculate_metrics(data['Consumption'][:len(forecast)], forecast)
            if mae and mse and r2:
                st.write(f"LSTM Metrics - MAE: {mae}, MSE: {mse}, R²: {r2}")
        else:
            st.error("LSTM model could not be loaded.")
    
    elif model_choice == "XGBoost":
        forecast = xgboost_forecast(data)
        if len(forecast) > 0:
            st.line_chart(forecast)
            mae, mse, r2 = calculate_metrics(data['Consumption'], forecast)
            if mae and mse and r2:
                st.write(f"XGBoost Metrics - MAE: {mae}, MSE: {mse}, R²: {r2}")
    
    elif model_choice == "Decision Tree":
        forecast = decision_tree_forecast(data)
        if len(forecast) > 0:
            st.line_chart(forecast)
            mae, mse, r2 = calculate_metrics(data['Consumption'], forecast)
            if mae and mse and r2:
                st.write(f"Decision Tree Metrics - MAE: {mae}, MSE: {mse}, R²: {r2}")
    
    elif model_choice == "Random Forest":
        forecast = random_forest_forecast(data)
        if len(forecast) > 0:
            st.line_chart(forecast)
            mae, mse, r2 = calculate_metrics(data['Consumption'], forecast)
            if mae and mse and r2:
                st.write(f"Random Forest Metrics - MAE: {mae}, MSE: {mse}, R²: {r2}")