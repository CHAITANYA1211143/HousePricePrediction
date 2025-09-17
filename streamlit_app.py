import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import base64
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Indian House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Function to set a CSS Gradient Background ---
def set_gradient_background():
    """ Sets a stylish gradient background for the Streamlit app. """
    page_bg_img = '''
    <style>
    .stApp {
        background-image: linear-gradient(to top right, #2c3e50, #4ca1af);
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
    /* --- General Text Color --- */
    .stApp, .st-emotion-cache-16txtl3, h1, h2, h3, p, .st-emotion-cache-1629p8f span {
        color: white !important; /* Force all text to be white */
        text-shadow: 1px 1px 2px #000000; /* Add shadow for readability */
    }
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
        background-color: rgba(76, 161, 175, 0.8);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        border: 2px solid #FFFFFF;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .prediction-value, .prediction-label {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Data Loading and Processing ---
@st.cache_data
def get_location_mapping(df):
    geolocator = Nominatim(user_agent="streamlit_house_price_app")
    postal_code_coords = df.groupby('Postal Code')[['Lattitude', 'Longitude']].mean()
    mapping = {}
    total_codes = len(postal_code_coords)
    if total_codes > 0:
        progress_bar = st.progress(0, text="Fetching location names (first-time setup)...")
        for i, (code, row) in enumerate(postal_code_coords.iterrows()):
            try:
                location = geolocator.reverse(f"{row['Lattitude']}, {row['Longitude']}", exactly_one=True, timeout=10)
                if location and location.raw.get('address'):
                    address = location.raw['address']
                    name = address.get('city', address.get('suburb', address.get('county', str(code))))
                    mapping[code] = name
                else: mapping[code] = str(code)
            except (GeocoderTimedOut, Exception):
                mapping[code] = str(code)
            progress_bar.progress((i + 1) / total_codes, text=f"Fetching... ({i+1}/{total_codes})")
        progress_bar.empty()
    return mapping

@st.cache_data
def load_and_train():
    df = pd.read_csv('House Price India.csv')
    df_processed = df.drop(['id', 'Date'], axis=1)
    df_processed = df_processed.fillna(df_processed.mean())
    features = ['number of bedrooms', 'number of bathrooms', 'living area', 'lot area', 'number of floors', 'waterfront present', 'number of views', 'condition of the house', 'grade of the house', 'Area of the house(excluding basement)', 'Area of the basement', 'Built Year', 'Renovation Year', 'Lattitude', 'Longitude', 'living_area_renov', 'lot_area_renov', 'Number of schools nearby', 'Distance from the airport']
    X = df_processed[features]
    y = df_processed['Price']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X, df

# --- Main App Logic ---
set_gradient_background() 
local_css()
model, X, df = load_and_train()
location_map = get_location_mapping(df)

display_locations = ["- Select a Location -"]
display_to_postal = {}
for code, name in sorted(location_map.items(), key=lambda item: item[1]):
    display_str = f"{name} ({code})"
    display_locations.append(display_str)
    display_to_postal[display_str] = code

st.markdown('<p class="main-title">Indian House Price Predictor</p>', unsafe_allow_html=True)
st.sidebar.header('🏠 Input Features')
selected_display_location = st.sidebar.selectbox("Location", display_locations)
selected_postal_code = display_to_postal.get(selected_display_location)
num_bedrooms = st.sidebar.slider('Number of Bedrooms', int(X['number of bedrooms'].min()), int(X['number of bedrooms'].max()), int(X['number of bedrooms'].mean()))
num_bathrooms = st.sidebar.slider('Number of Bathrooms', float(X['number of bathrooms'].min()), 10.0, float(X['number of bathrooms'].mean()))
living_area = st.sidebar.slider('Living Area (sq ft)', int(X['living area'].min()), int(X['living area'].max()), int(X['living area'].mean()))
built_year = st.sidebar.slider('Year Built', int(X['Built Year'].min()), int(X['Built Year'].max()), int(X['Built Year'].mean()))

if selected_postal_code:
    location_data = df[df['Postal Code'] == selected_postal_code]
    lat, lon = location_data['Lattitude'].mean(), location_data['Longitude'].mean()
else:
    lat, lon = df['Lattitude'].mean(), df['Longitude'].mean()

data = {'number of bedrooms': num_bedrooms, 'number of bathrooms': num_bathrooms, 'living area': living_area, 'lot area': int(X['lot area'].mean()), 'number of floors': float(X['number of floors'].mean()), 'waterfront present': int(X['waterfront present'].mean()), 'number of views': int(X['number of views'].mean()), 'condition of the house': int(X['condition of the house'].mean()), 'grade of the house': int(X['grade of the house'].mean()), 'Area of the house(excluding basement)': living_area, 'Area of the basement': int(X['Area of the basement'].mean()), 'Built Year': built_year, 'Renovation Year': int(X['Renovation Year'].mean()), 'Lattitude': lat, 'Longitude': lon, 'living_area_renov': living_area, 'lot_area_renov': int(X['lot area'].mean()), 'Number of schools nearby': int(X['Number of schools nearby'].mean()), 'Distance from the airport': int(X['Distance from the airport'].mean())}
input_df = pd.DataFrame(data, index=[0])
prediction = model.predict(input_df)

col1, col2 = st.columns([2, 3])
with col1:
    st.subheader("Your Selections")
    if selected_postal_code: st.write(f"**Location:** {selected_display_location}")
    st.write(f"**Bedrooms:** {num_bedrooms}")
    st.write(f"**Bathrooms:** {num_bathrooms}")
    st.write(f"**Living Area:** {living_area} sq ft")
    st.write(f"**Year Built:** {built_year}")

with col2:
    st.write("") 
    st.write("") 
    st.markdown(f"""<div class="prediction-box"><p class="prediction-label">Predicted House Price</p><p class="prediction-value">₹{prediction[0]:,.2f}</p></div>""", unsafe_allow_html=True)

# --- Data Visualizations Section ---
with st.expander("Explore Data Visualizations 📊"):
    st.markdown("### Price Insights")
    
    # Plot Style
    plt.style.use('dark_background')
    
    # Plot 1: Distribution of House Prices
    fig1, ax1 = plt.subplots()
    ax1.hist(df['Price'] / 100000, bins=50, color='#4ca1af') # Price in Lakhs
    ax1.set_title("Distribution of House Prices", color='white')
    ax1.set_xlabel("Price (in Lakhs ₹)", color='white')
    ax1.set_ylabel("Number of Houses", color='white')
    ax1.tick_params(colors='white')
    fig1.patch.set_alpha(0.0) # Transparent background
    st.pyplot(fig1)

    # Plot 2: Living Area vs. Price
    fig2, ax2 = plt.subplots()
    ax2.scatter(df['living area'], df['Price'] / 100000, alpha=0.3, color='#2c3e50', edgecolors='white')
    ax2.set_title("Living Area vs. Price", color='white')
    ax2.set_xlabel("Living Area (sq ft)", color='white')
    ax2.set_ylabel("Price (in Lakhs ₹)", color='white')
    ax2.tick_params(colors='white')
    fig2.patch.set_alpha(0.0)
    st.pyplot(fig2)

    # Plot 3: Average Price by Top Locations
    st.markdown("### Location Insights")
    avg_price_by_loc = df.groupby('Postal Code')['Price'].mean().sort_values(ascending=False).head(10)
    # Get location names for the top postal codes
    top_loc_names = [location_map.get(code, code) for code in avg_price_by_loc.index]
    
    fig3, ax3 = plt.subplots()
    ax3.bar(top_loc_names, avg_price_by_loc.values / 100000, color='#4ca1af')
    ax3.set_title("Top 10 Most Expensive Locations (Average Price)", color='white')
    ax3.set_ylabel("Average Price (in Lakhs ₹)", color='white')
    ax3.tick_params(axis='x', rotation=45, colors='white')
    ax3.tick_params(axis='y', colors='white')
    fig3.patch.set_alpha(0.0)
    st.pyplot(fig3)
