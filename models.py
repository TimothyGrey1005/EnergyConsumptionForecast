import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Load and preprocess the data
data = pd.read_csv('Updated_Energy_Consumption_Price.csv')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.sort_values('DateTime', inplace=True)

# Normalize the features
features_to_scale = ['Production', 'Water', 'Wind', 'Hydroelectric', 'Oil and Gas', 'Coal', 'Solar']
scaler = MinMaxScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Split data into training and testing
X = data[features_to_scale]
y = data['Consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model Selection
model_option = st.selectbox(
    "Select a model to visualize predictions:",
    ("XGBoost", "Decision Tree", "Random Forest", "KNN", "LSTM")
)

# Train and predict based on the selected model
if model_option == 'XGBoost':
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
elif model_option == 'Decision Tree':
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
elif model_option == 'Random Forest':
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
elif model_option == 'KNN':
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
else:  # LSTM
    # Reshape input for LSTM [samples, time steps, features]
    X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(1, X_train.shape[1])),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
    y_pred = model.predict(X_test_lstm).flatten()

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit App Title
st.title('Real-Time Energy Monitoring Dashboard')

# Display evaluation metrics
st.write(f"**Model:** {model_option}")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Display Total and Peak Consumption
total_consumption = data['Consumption'].sum()
average_consumption = data['Consumption'].mean()
peak_consumption = data['Consumption'].max()
peak_time = data.loc[data['Consumption'].idxmax(), 'DateTime']

st.metric("Total Energy Consumption", f"{total_consumption:.2f} kWh")
st.metric("Average Energy Consumption", f"{average_consumption:.2f} kWh")
st.metric("Peak Energy Consumption", f"{peak_consumption:.2f} kWh at {peak_time}")

# Plot Energy Consumption over Time
st.subheader('Energy Consumption Over Time')
plt.figure(figsize=(12, 6))
plt.plot(data['DateTime'], data['Consumption'], label='Actual Consumption', color='blue')
plt.xlabel('DateTime')
plt.ylabel('Consumption (kWh)')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Plot Predictions vs Actual
st.subheader(f'{model_option} Predictions vs Actual Values')
plt.figure(figsize=(12, 6))
plt.plot(y_test.reset_index(drop=True), label='Actual Values', color='blue')
plt.plot(y_pred, label=f'{model_option} Predicted Values', color='red', linestyle='dashed')
plt.xlabel('Time Steps')
plt.ylabel('Consumption (kWh)')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Dynamic threshold adjustment and prediction view
threshold = st.slider("Set Consumption Threshold", min_value=int(np.min(data['Consumption'])), max_value=int(np.max(data['Consumption'])), value=int(average_consumption))

# Highlight values exceeding the threshold
exceeding_consumption = data[data['Consumption'] > threshold]

if not exceeding_consumption.empty:
    st.warning(f"Warning! Consumption exceeded {threshold} kWh at the following times:")
    st.write(exceeding_consumption[['DateTime', 'Consumption']])
else:
    st.success(f"No consumption values exceed {threshold} kWh.")