import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# --- Configuration and File Paths ---
MODEL_PATH = 'linear_regression_model.joblib'
SCALER_PATH = 'scaler.joblib'
LOC_UPDATE_PATH = 'loc_update.pickle'

# --- Load Data and Models ---
@st.cache_data
def load_data_and_models():
    """Loads the dataset and trained models."""
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df['Target'] = data.target

    # Load pre-processed location data
    if os.path.exists(LOC_UPDATE_PATH):
        with open(LOC_UPDATE_PATH, 'rb') as f:
            loc_update = pickle.load(f)

        # Create DataFrame from loaded data
        location_data = []
        for coordinates, address in loc_update.items():
            if isinstance(address, dict):
                county = address.get('county')
                road = address.get('road')
                location_data.append({'Latitude': coordinates[0], 'Longitude': coordinates[1], 'county': county, 'Road': road})
            else:
                location_data.append({'Latitude': coordinates[0], 'Longitude': coordinates[1], 'county': None, 'Road': None})
        
        df_locations_features = pd.DataFrame(location_data)

        # Merge new features
        df = pd.merge(df, df_locations_features, on=['Latitude', 'Longitude'], how='left')
        df['county'].fillna('Unknown', inplace=True)
        df['Road'].fillna('Unknown', inplace=True)

    else:
        st.warning("Location data not found. Skipping geographical feature engineering.")
        df['county'] = 'Unknown'
        df['Road'] = 'Unknown'
        
    df = pd.get_dummies(df, columns=['county', 'Road'], dummy_na=False)

    # Load model and scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return df, model, scaler

# --- Streamlit App ---
st.title("🏡 California House Price Predictor")
st.markdown("This app uses a Linear Regression model to predict house prices in California.")

df, model, scaler = load_data_and_models()

# Split data for evaluation (optional, but good for context)
X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Sidebar for user input ---
st.sidebar.header("User Input Features")

def user_input_features():
    MedInc = st.sidebar.slider('Median Income (MedInc)', float(df['MedInc'].min()), float(df['MedInc'].max()), float(df['MedInc'].mean()))
    HouseAge = st.sidebar.slider('House Age (HouseAge)', float(df['HouseAge'].min()), float(df['HouseAge'].max()), float(df['HouseAge'].mean()))
    AveRooms = st.sidebar.slider('Average Rooms (AveRooms)', float(df['AveRooms'].min()), float(df['AveRooms'].max()), float(df['AveRooms'].mean()))
    AveBedrms = st.sidebar.slider('Average Bedrooms (AveBedrms)', float(df['AveBedrms'].min()), float(df['AveBedrms'].max()), float(df['AveBedrms'].mean()))
    Population = st.sidebar.slider('Population', float(df['Population'].min()), float(df['Population'].max()), float(df['Population'].mean()))
    AveOccup = st.sidebar.slider('Average Occupancy (AveOccup)', float(df['AveOccup'].min()), float(df['AveOccup'].max()), float(df['AveOccup'].mean()))
    Latitude = st.sidebar.slider('Latitude', float(df['Latitude'].min()), float(df['Latitude'].max()), float(df['Latitude'].mean()))
    Longitude = st.sidebar.slider('Longitude', float(df['Longitude'].min()), float(df['Longitude'].max()), float(df['Longitude'].mean()))
    
    data = {'MedInc': MedInc,
            'HouseAge': HouseAge,
            'AveRooms': AveRooms,
            'AveBedrms': AveBedrms,
            'Population': Population,
            'AveOccup': AveOccup,
            'Latitude': Latitude,
            'Longitude': Longitude}
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Prediction Logic ---
if st.sidebar.button('Predict'):
    # Prepare the input data for the model
    numerical_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    
    # Create a DataFrame with all columns from the training set, filled with zeros
    input_for_prediction = pd.DataFrame(0, index=[0], columns=X_train.columns)
    
    # Scale the numerical features from user input
    scaled_input_numerical = scaler.transform(input_df[numerical_cols])
    
    # Fill in the scaled numerical values
    for i, col in enumerate(numerical_cols):
        input_for_prediction[col] = scaled_input_numerical[0][i]

    # Make the prediction
    prediction = model.predict(input_for_prediction)
    
    # Display the result
    st.subheader("Prediction Result")
    st.success(f'Predicted House Price: ${prediction[0] * 100000:,.2f}')
    st.info("The price is in US dollars. The original target variable was scaled, so we multiply by 100,000 for the final result.")

# --- Model Evaluation Section (Optional) ---
st.header("Model Performance Metrics")
if st.button("Show Model Metrics"):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
    st.write(f"**R-squared (R2) Score:** {r2:.4f}")
    
    st.markdown("""
        **R-squared (R2) Score:**
        An R-squared score of **0.5818** means that the model explains approximately **58.2%** of the variance in the house prices. 
        This is a decent result, but it also indicates there's room for improvement by perhaps trying more complex models or additional feature engineering.
    """)
