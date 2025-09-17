import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import base64
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# --- Page Configuration ---
st.set_page_config(
    page_title="Indian House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- NEW: Function to set a CSS Gradient Background ---
def set_gradient_background():
    """
    Sets a stylish gradient background for the Streamlit app.
    This removes the dependency on an external image file.
    """
    page_bg_img = '''
    <style>
    .stApp {
        background-image: linear-gradient(to top right, #002f4b, #dc4225);
        background-attachment: fixed;
        background-size: cover;
    }
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

# --- Data Loading and Processing ---
@st.cache_data
def get_location_mapping(df):
    """
    Performs reverse geocoding to map postal codes to location names.
    This is slow and runs only once thanks to caching.
    """
    geolocator = Nominatim(user_agent="streamlit_house_price_app")
    postal_code_coords = df.groupby('Postal Code')[['Lattitude', 'Longitude']].mean()
    mapping = {}
    
    progress_bar = st.progress(0, text="Fetching location names (first-time setup)...")
    total_codes = len(postal_code_coords)

    for i, (code, row) in enumerate(postal_code_coords.iterrows()):
        try:
            location = geolocator.reverse(f"{row['Lattitude']}, {row['Longitude']}", exactly_one=True, timeout=10)
            if location and location.raw.get('address'):
                address = location.raw['address']
                name = address.get('city', address.get('suburb', address.get('county', str(code))))
                mapping[code] = name
            else:
                mapping[code] = str(code) # Fallback to postal code
        except (GeocoderTimedOut, Exception):
            mapping[code] = str(code) # Fallback on error
        
        progress_bar.progress((i + 1) / total_codes, text=f"Fetching location names... ({i+1}/{total_codes})")

    progress_bar.empty()
    return mapping

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
set_gradient_background() # Using the new gradient background
local_css()
model, X, df = load_and_train()
location_map = get_location_mapping(df)

# Create display list and a map from the display string back to the postal code
display_locations = ["- Select a Location -"]
display_to_postal = {}
for code, name in sorted(location_map.items(), key=lambda item: item[1]):
    display_str = f"{name} ({code})"
    display_locations.append(display_str)
    display_to_postal[display_str] = code

# App Title
st.markdown('<p class="main-title">Indian House Price Predictor</p>', unsafe_allow_html=True)
st.write('Select the features of a house, including its location, to get an estimated market price.')

# Sidebar Inputs
st.sidebar.header('🏠 Input Features')
st.sidebar.markdown("Adjust the values below to get a price prediction.")

selected_display_location = st.sidebar.selectbox("Location", display_locations)
selected_postal_code = display_to_postal.get(selected_display_location)

num_bedrooms = st.sidebar.slider('Number of Bedrooms', int(X['number of bedrooms'].min()), int(X['number of bedrooms'].max()), int(X['number of bedrooms'].mean()))
num_bathrooms = st.sidebar.slider('Number of Bathrooms', float(X['number of bathrooms'].min()), 10.0, float(X['number of bathrooms'].mean()))
living_area = st.sidebar.slider('Living Area (sq ft)', int(X['living area'].min()), int(X['living area'].max()), int(X['living area'].mean()))
built_year = st.sidebar.slider('Year Built', int(X['Built Year'].min()), int(X['Built Year'].max()), int(X['Built Year'].mean()))
num_schools = st.sidebar.slider('Number of Schools Nearby', int(X['Number of schools nearby'].min()), int(X['Number of schools nearby'].max()), int(X['Number of schools nearby'].mean()))
distance_airport = st.sidebar.slider('Distance from Airport (km)', int(X['Distance from the airport'].min()), int(X['Distance from the airport'].max()), int(X['Distance from the airport'].mean()))

# Determine Latitude and Longitude
if selected_postal_code:
    location_data = df[df['Postal Code'] == selected_postal_code]
    lat = location_data['Lattitude'].mean()
    lon = location_data['Longitude'].mean()
else:
    lat = df['Lattitude'].mean()
    lon = df['Longitude'].mean()
    st.sidebar.warning("No location selected. Using overall average coordinates.")

# Create Input DataFrame
data = {
    'number of bedrooms': num_bedrooms,
    'number of bathrooms': num_bathrooms,
    'living area': living_area,
    'lot area': int(X['lot area'].mean()),
    'number of floors': float(X['number of floors'].mean()),
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
    'lot_area_renov': int(X['lot area'].mean()), 
    'Number of schools nearby': num_schools,
    'Distance from the airport': distance_airport
}
input_df = pd.DataFrame(data, index=[0])

# Prediction and Display
prediction = model.predict(input_df)

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Your Selections")
    if selected_postal_code:
        st.write(f"**Location:** {selected_display_location}")
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
