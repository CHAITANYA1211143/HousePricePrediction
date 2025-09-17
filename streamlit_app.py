import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import base64
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Indian House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Function to encode a local image to base64 ---
@st.cache_data
def get_base64_of_bin_file(bin_file):
    """ Reads a binary file and returns its base64 encoded string. """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- Function to set the background ---
def set_background(png_file):
    """ Sets the background of the Streamlit app from a local file. """
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Custom CSS for Styling ---
def local_css():
    st.markdown("""
    <style>
    .st-emotion-cache-1avcm0n {
        background-color: rgba(255, 255, 255, 0.15); /* Sidebar background */
        backdrop-filter: blur(10px);
        border-radius: 10px;
    }
    .st-emotion-cache-16txtl3 {
        padding: 1rem 1rem 1rem;
    }
    .stApp > header {
        background-color: transparent;
    }
    .main-title {
        font-size: 3rem !important;
        font-weight: bold;
        color: #FFFFFF;
        text-shadow: 2px 2px 4px #000000;
    }
    .prediction-box {
        background-color: rgba(46, 204, 113, 0.8);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        border: 2px solid #FFFFFF;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .prediction-value {
        font-size: 2.5rem !important;
        font-weight: bold;
        color: white;
    }
    .prediction-label {
        font-size: 1.2rem !important;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Data and Train Model (Cached) ---
@st.cache_data
def load_and_train():
    """ Loads data, processes it, and trains the model. """
    df = pd.read_csv('House Price India.csv')
    df_processed = df.drop(['id', 'Date'], axis=1)
    df_processed = df_processed.fillna(df_processed.mean())
    
    features = [
        'number of bedrooms', 'number of bathrooms', 'living area', 'lot area',
        'number of floors', 'waterfront present', 'number of views',
        'condition of the house', 'grade of the house',
        'Area of the house(excluding basement)', 'Area of the basement',
        'Built Year', 'Renovation Year', 'Lattitude', 'Longitude',
        'living_area_renov', 'lot_area_renov', 'Number of schools nearby',
        'Distance from the airport'
    ]
    X = df_processed[features]
    y = df_processed['Price']
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X, df

# --- Main App Logic ---

# Set background and styles
if os.path.exists('background.jpg'):
    set_background('background.jpg')
else:
    st.warning("Background image `background.jpg` not found. Please add it to the folder.")

local_css()
model, X, df = load_and_train()

# App Title
st.markdown('<p class="main-title">Indian House Price Predictor</p>', unsafe_allow_html=True)
st.write('Select the features of a house, including its location, to get an estimated market price.')

# Sidebar Inputs
st.sidebar.header('🏠 Input Features')
st.sidebar.markdown("Adjust the values below to get a price prediction.")

# Location Selector
postal_codes = ["- Select a Location -"] + sorted(df['Postal Code'].unique().tolist())
selected_location = st.sidebar.selectbox("Location (by Postal Code)", postal_codes)

# Sliders for other features
num_bedrooms = st.sidebar.slider('Number of Bedrooms', int(X['number of bedrooms'].min()), int(X['number of bedrooms'].max()), int(X['number of bedrooms'].mean()))
num_bathrooms = st.sidebar.slider('Number of Bathrooms', float(X['number of bathrooms'].min()), 10.0, float(X['number of bathrooms'].mean()))
living_area = st.sidebar.slider('Living Area (sq ft)', int(X['living area'].min()), int(X['living area'].max()), int(X['living area'].mean()))
lot_area = st.sidebar.slider('Lot Area (sq ft)', int(X['lot area'].min()), int(X['lot area'].max()), int(X['lot area'].mean()))
num_floors = st.sidebar.slider('Number of Floors', float(X['number of floors'].min()), 5.0, float(X['number of floors'].mean()))
built_year = st.sidebar.slider('Year Built', int(X['Built Year'].min()), int(X['Built Year'].max()), int(X['Built Year'].mean()))
num_schools = st.sidebar.slider('Number of Schools Nearby', int(X['Number of schools nearby'].min()), int(X['Number of schools nearby'].max()), int(X['Number of schools nearby'].mean()))
distance_airport = st.sidebar.slider('Distance from Airport (km)', int(X['Distance from the airport'].min()), int(X['Distance from the airport'].max()), int(X['Distance from the airport'].mean()))

# Determine Latitude and Longitude
if selected_location != "- Select a Location -":
    location_data = df[df['Postal Code'] == selected_location]
    lat = location_data['Lattitude'].mean()
    lon = location_data['Longitude'].mean()
    st.sidebar.info(f"Using average coordinates for Postal Code {selected_location}.")
else:
    lat = df['Lattitude'].mean()
    lon = df['Longitude'].mean()
    st.sidebar.warning("No location selected. Using overall average coordinates.")

# Create Input DataFrame
data = {
    'number of bedrooms': num_bedrooms,
    'number of bathrooms': num_bathrooms,
    'living area': living_area,
    'lot area': lot_area,
    'number of floors': num_floors,
    'waterfront present': int(X['waterfront present'].mean()),
    'number of views': int(X['number of views'].mean()),
    'condition of the house': int(X['condition of the house'].mean()),
    'grade of the house': int(X['grade of the house'].mean()),
    'Area of the house(excluding basement)': living_area, 
    'Area of the basement': int(X['Area of the basement'].mean()),
    'Built Year': built_year,
    'Renovation Year': int(X['Renovation Year'].mean()),
    'Lattitude': lat,
    'Longitude': lon,
    'living_area_renov': living_area,
    'lot_area_renov': lot_area, 
    'Number of schools nearby': num_schools,
    'Distance from the airport': distance_airport
}
input_df = pd.DataFrame(data, index=[0])

# Prediction and Display
prediction = model.predict(input_df)

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Your Selections")
    if selected_location != "- Select a Location -":
        st.write(f"**Location (Postal Code):** {selected_location}")
    st.write(f"**Bedrooms:** {num_bedrooms}")
    st.write(f"**Bathrooms:** {num_bathrooms}")
    st.write(f"**Living Area:** {living_area} sq ft")
    st.write(f"**Year Built:** {built_year}")
    st.write(f"**Schools Nearby:** {num_schools}")

with col2:
    st.write("") 
    st.write("") 
    st.markdown(
        f"""
        <div class="prediction-box">
            <p class="prediction-label">Predicted House Price</p>
            <p class="prediction-value">₹{prediction[0]:,.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Expander for More Info
with st.expander("ℹ️ About this App"):
    st.write("""
        This application uses a **Linear Regression model** to predict house prices based on the features you select. 
        The model was trained on a dataset of house sales in India. Please note that this is an estimation, and actual market prices can vary based on many other factors.
    """)
    st.write("**Dataset:** House Price India.csv")
    st.write("**Model:** Scikit-learn LinearRegression")
