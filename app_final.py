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
import plotly.graph_objects as go

# Data preprocessing
@st.cache_data
def load_data():
    data = pd.read_csv('Updated_Energy_Consumption_Price.csv')
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data.sort_values('DateTime', inplace=True)
    return data

data = load_data()

# Normalize features
features_to_scale = ['Production', 'Water', 'Wind', 'Hydroelectric', 'Oil and Gas', 'Coal', 'Solar']
scaler = MinMaxScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

def evaluate_all_models(X, y):
    models = {
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "LSTM": Sequential([
            LSTM(50, activation='relu', input_shape=(1, X.shape[1])),
            Dense(1)
        ])
    }
    
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    for name, model in models.items():
        if name != "LSTM":
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
            model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
            y_pred = model.predict(X_test_lstm).flatten()
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MSE": mse, "R2": r2, "y_test": y_test, "y_pred": y_pred}
    
    return results

# Calculate overall model performance
overall_results = evaluate_all_models(data[features_to_scale], data['Consumption'])

# Streamlit App Title
st.title('Real-Time Energy Monitoring Dashboard')

# Sidebar
st.sidebar.header("Settings")
selected_building = st.sidebar.selectbox("Choose a building", data['Building'].unique())
model_option = st.sidebar.selectbox(
    "Select a model for predictions:",
    ("XGBoost", "Decision Tree", "Random Forest", "KNN", "LSTM")
)

st.sidebar.write("---")
st.sidebar.header("Overall Model Comparison")
overall_model_option = st.sidebar.selectbox(
    "Select a model for overall comparison:",
    ("XGBoost", "Decision Tree", "Random Forest", "KNN", "LSTM", "All Models")
)

# Filter data based on selected building
filtered_data = data[data['Building'] == selected_building]

# Train and test split
X = filtered_data[features_to_scale]
y = filtered_data['Consumption']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train and predict based on chosen model
if model_option == 'XGBoost':
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
elif model_option == 'Decision Tree':
    model = DecisionTreeRegressor(random_state=42)
elif model_option == 'Random Forest':
    model = RandomForestRegressor(n_estimators=100, random_state=42)
elif model_option == 'KNN':
    model = KNeighborsRegressor(n_neighbors=5)
else:  # LSTM
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(1, X_train.shape[1])),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

if model_option != 'LSTM':
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
else:
    X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
    model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
    y_pred = model.predict(X_test_lstm).flatten()

# Calculation of evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Define the create_interactive_bar_chart function
def create_interactive_bar_chart(data):
    fig = go.Figure(data=[
        go.Bar(
            x=data['DateTime'],
            y=data['Consumption'],
            hovertext=data['Consumption'].round(2).astype(str) + ' kWh',
            hoverinfo='text'
        )
    ])
    fig.update_layout(
        title='Energy Consumption by DateTime',
        xaxis_title='DateTime',
        yaxis_title='Consumption (kWh)',
        height=500
    )
    return fig

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("Energy Consumption Overview")
    st.metric("Total Energy Consumption", f"{filtered_data['Consumption'].sum():.2f} kWh")
    st.metric("Average Energy Consumption", f"{filtered_data['Consumption'].mean():.2f} kWh")
    peak_consumption = filtered_data['Consumption'].max()
    peak_time = filtered_data.loc[filtered_data['Consumption'].idxmax(), 'DateTime']
    st.metric("Peak Energy Consumption", f"{peak_consumption:.2f} kWh at {peak_time}")

with col2:
    st.header("Model Evaluation")
    st.write(f"**Model:** {model_option}")
    st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    st.metric("R² Score", f"{r2:.2f}")

st.write("---")

col3, col4 = st.columns(2)

with col3:
    st.header('Energy Consumption Over Time')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(filtered_data['DateTime'], filtered_data['Consumption'], label='Actual Consumption', color='blue')
    ax.set_xlabel('DateTime')
    ax.set_ylabel('Consumption (kWh)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

with col4:
    st.header(f'{model_option} Predictions vs Actual Values')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_test.reset_index(drop=True), label='Actual Values', color='blue')
    ax.plot(y_pred, label=f'{model_option} Predicted Values', color='red', linestyle='dashed')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Consumption (kWh)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

st.write("---")

col5, col6 = st.columns(2)

with col5:
    st.header('Interactive Energy Consumption Bar Chart')
    bar_chart = create_interactive_bar_chart(filtered_data)
    st.plotly_chart(bar_chart, use_container_width=True)

with col6:
    st.header('Consumption Threshold Analysis')
    threshold = st.slider(
        "Set Consumption Threshold", 
        min_value=int(np.min(filtered_data['Consumption'])), 
        max_value=int(np.max(filtered_data['Consumption'])), 
        value=int(filtered_data['Consumption'].mean())
    )

    exceeding_consumption = filtered_data[filtered_data['Consumption'] > threshold]

    if not exceeding_consumption.empty:
        st.warning(f"Warning! Consumption exceeded {threshold} kWh at the following times:")
        st.dataframe(exceeding_consumption[['DateTime', 'Consumption']])
    else:
        st.success(f"No consumption values exceed {threshold} kWh.")

st.write("---")
st.header("Overall Model Comparison")

if overall_model_option == "All Models":
    # Create a bar chart comparing MSE and R2 for all models
    mse_values = [results["MSE"] for results in overall_results.values()]
    r2_values = [results["R2"] for results in overall_results.values()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.bar(overall_results.keys(), mse_values)
    ax1.set_title("Mean Squared Error Comparison")
    ax1.set_ylabel("MSE")
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.bar(overall_results.keys(), r2_values)
    ax2.set_title("R² Score Comparison")
    ax2.set_ylabel("R²")
    ax2.tick_params(axis='x', rotation=45)
    
    st.pyplot(fig)
else:
    # Display detailed results for the selected model
    results = overall_results[overall_model_option]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Squared Error (MSE)", f"{results['MSE']:.4f}")
    with col2:
        st.metric("R² Score", f"{results['R2']:.4f}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(results['y_test'], results['y_pred'], alpha=0.5)
    ax.plot([min(results['y_test']), max(results['y_test'])], [min(results['y_test']), max(results['y_test'])], 'r--', lw=2)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"{overall_model_option} - Actual vs Predicted")
    st.pyplot(fig)