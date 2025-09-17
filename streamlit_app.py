import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import base64
import requests # Added to fetch the image from a URL

# --- Function to set background image from a URL ---
@st.cache_data
def get_base64_of_url_image(url):
    """ Fetches an image from a URL and returns its base64 encoded string. """
    response = requests.get(url)
    if response.status_code == 200:
        return base64.b64encode(response.content).decode()
    else:
        return None

def set_background_from_url(url):
    """ Sets the background of the Streamlit app from a URL. """
    bin_str = get_base64_of_url_image(url)
    if bin_str:
        page_bg_img = f'''
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
    else:
        st.warning("Failed to load background image from URL.")

# --- Set the background ---
# Using a URL for an image of an Indian house
image_url = 'https://images.unsplash.com/photo-1600585154340-be6164a83639?auto=format&fit=crop&w=1770'
set_background_from_url(image_url)


# --- The rest of your app code remains the same ---

# Load the dataset
@st.cache_data
def load_data():
    """ Loads the Indian House Price dataset. """
    df = pd.read_csv('House Price India.csv')
    # Drop columns that are not useful for this model
    df = df.drop(['id', 'Date', 'Postal Code'], axis=1)
    # For simplicity, we'll fill missing values with the mean
    df = df.fillna(df.mean())
    return df

df = load_data()

# Define the features (X) and target (y)
features = [
    'number of bedrooms', 'number of bathrooms', 'living area', 'lot area',
    'number of floors', 'waterfront present', 'number of views',
    'condition of the house', 'grade of the house',
    'Area of the house(excluding basement)', 'Area of the basement',
    'Built Year', 'Renovation Year', 'Lattitude', 'Longitude',
    'living_area_renov', 'lot_area_renov', 'Number of schools nearby',
    'Distance from the airport'
]
X = df[features]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Create the Streamlit app
st.title('🏠 Indian House Price Prediction App')
st.write('This app predicts the price of a house in India based on its features.')

# Create the sidebar for user input
st.sidebar.header('Input Features')

def user_input_features():
    """ Creates sidebar sliders for user input. """
    num_bedrooms = st.sidebar.slider('Number of Bedrooms', int(X['number of bedrooms'].min()), int(X['number of bedrooms'].max()), int(X['number of bedrooms'].mean()))
    num_bathrooms = st.sidebar.slider('Number of Bathrooms', float(X['number of bathrooms'].min()), float(X['number of bathrooms'].max()), float(X['number of bathrooms'].mean()))
    living_area = st.sidebar.slider('Living Area (sq ft)', int(X['living area'].min()), int(X['living area'].max()), int(X['living area'].mean()))
    lot_area = st.sidebar.slider('Lot Area (sq ft)', int(X['lot area'].min()), int(X['lot area'].max()), int(X['lot area'].mean()))
    num_floors = st.sidebar.slider('Number of Floors', float(X['number of floors'].min()), float(X['number of floors'].max()), float(X['number of floors'].mean()))
    built_year = st.sidebar.slider('Built Year', int(X['Built Year'].min()), int(X['Built Year'].max()), int(X['Built Year'].mean()))
    num_schools = st.sidebar.slider('Number of Schools Nearby', int(X['Number of schools nearby'].min()), int(X['Number of schools nearby'].max()), int(X['Number of schools nearby'].mean()))
    distance_airport = st.sidebar.slider('Distance from Airport', int(X['Distance from the airport'].min()), int(X['Distance from the airport'].max()), int(X['Distance from the airport'].mean()))


    # For the model, we need all the features, so we'll use mean values for the ones not in the sidebar
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
        'Area of the house(excluding basement)': int(X['Area of the house(excluding basement)'].mean()),
        'Area of the basement': int(X['Area of the basement'].mean()),
        'Built Year': built_year,
        'Renovation Year': int(X['Renovation Year'].mean()),
        'Lattitude': float(X['Lattitude'].mean()),
        'Longitude': float(X['Longitude'].mean()),
        'living_area_renov': int(X['living_area_renov'].mean()),
        'lot_area_renov': int(X['lot_area_renov'].mean()),
        'Number of schools nearby': num_schools,
        'Distance from the airport': distance_airport
    }
    features_df = pd.DataFrame(data, index=[0])
    return features_df

input_df = user_input_features()

# Display the user input
st.subheader('User Input')
st.write(input_df)

# Predict the price
prediction = model.predict(input_df)

# Display the prediction
st.subheader('Prediction')
st.write(f'The predicted price of the house is: **₹{prediction[0]:,.2f}**')
