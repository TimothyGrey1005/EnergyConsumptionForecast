import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Updated_Energy_Consumption_Price.csv')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.sort_values('DateTime', inplace=True)

# Streamlit Title
st.title('Real-Time Energy Monitoring Dashboard')

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
plt.plot(data['DateTime'], data['Consumption'], label='Consumption', color='blue')
plt.xlabel('DateTime')
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