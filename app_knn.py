import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv('Updated_Energy_Consumption_Price.csv')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.sort_values('DateTime', inplace=True)

# Normalize the features
features_to_scale = ['Production', 'Water', 'Wind', 'Hydroelectric', 'Oil and Gas', 'Coal', 'Solar']
scaler = MinMaxScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Split the data into training and testing sets
X = data[features_to_scale]
y = data['Consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the KNN model
knn_model = KNeighborsRegressor(n_neighbors=5)  # You can adjust n_neighbors as needed
knn_model.fit(X_train, y_train)

# Make predictions
y_pred_knn = knn_model.predict(X_test)

# Streamlit App Title
st.title('Real-Time Energy Monitoring Dashboard with KNN')

# Display Total and Peak Consumption
total_consumption = data['Consumption'].sum()
average_consumption = data['Consumption'].mean()
peak_consumption = data['Consumption'].max()
peak_time = data.loc[data['Consumption'].idxmax(), 'DateTime']

st.metric("Total Energy Consumption", f"{total_consumption} kWh")
st.metric("Average Energy Consumption", f"{average_consumption} kWh")
st.metric("Peak Energy Consumption", f"{peak_consumption} kWh at {peak_time}")

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
st.subheader('KNN Predictions vs Actual Values')
plt.figure(figsize=(12, 6))
plt.plot(y_test.reset_index(drop=True), label='Actual Values', color='blue')
plt.plot(y_pred_knn, label='Predicted Values', color='purple', linestyle='dashed')
plt.xlabel('Time Steps')
plt.ylabel('Consumption (kWh)')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Set a threshold and check for high consumption
threshold = st.slider("Set Consumption Threshold", min_value=3500, max_value=5800, value=4000)
exceeding_consumption = data[data['Consumption'] > threshold]

if not exceeding_consumption.empty:
    st.warning(f"Warning! Consumption exceeded {threshold} kWh at the following times:")
    st.write(exceeding_consumption[['DateTime', 'Consumption']])