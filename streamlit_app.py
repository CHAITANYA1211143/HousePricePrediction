import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import base64

# --- Function to set background image from a base64 string ---
def set_background(image_data):
    """
    Sets the background of the Streamlit app using a base64 encoded image.
    """
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{image_data}");
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Set the background ---
# Image data is now embedded directly into the script for reliability.
# This prevents the "failed to load" error.
image_base64 = "iVBORw0KGgoAAAANSUhEUgAAN4AAAC9CAMAAABua3sSAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAMAUExURQAAABIdLBYjNBglOBsoPBIcKhMdKyAmOhsoPB8qQiMsQR0sRikoPyIuTiYxVyYyWCg0Wic0Wig1XCo3Xis4YC06Yy48ZTJAajNCbDRDcTZFcjdGcztIdz5KepBJa5FKbZJLbpZOdJpQd5pReJ1Ufp5XgaBYgqJdhqZgirBnjLFojbJpjrZtkLhzlLp0mMB3m8N6n8R7oMZ9ocuDp8+HqNGJr9eMutqQv92Uwt+XxeKYxuOayOOay+SdzOWez+ag0Oih0uqj1Oul1uym1+2n2O+r2vCv3fGx3vSy4fWz4va04/i35fm55/q66fv88v399P7+/v///wYIDQYKDwYMERAXExMbGxshISMkJCcoKCctLS4xMTI1NTY4ODk7Ozw+Pj9AQUJERUZISUpMTU5QUVJUVVdZWVpcXF9hYWJjY2ZoaGlra2xtbW5wcHF0dHZ3d3h5eXp7fH1+fn+AgYSFhYeIiYqMjY6PkJGSk5SVlpeYmZqbnJ2en6ChoqOkpaanqKmqq6ytrq+wsbKztLa3t7i5uru8vb/AwcPFx8jKy8zNzs/Q0dLT1NXW19jb3N3f4OHi4+Tl5ufp6uvs7e7v8PHy8/T19vf4+fr7/P3+LwAAAAp0Uk5T/////////+AD3xN4AAAACXBIWXMAAA7DAAAOwwHHb6hkAAAgAElEQVR42uyde5xcVZX/P3+f02mmyQ13k+SG3WQhAbtJ3Ew24wZ3cQkI3QvEwS1dIohggQ0iQkQECyJIkQcEAoIiyI5CQAQREBAU3Ew2yQYbsGk3h8lMbrrT9JyZz/fH3Kk7s5Mmu2k20+3Mv+ebqapOddWd+j7z3VvX1S8W6I8Gg1/Jc//o/V2V9x8e8i/u/9P3hH6Sg2Dwc1n2M4227n987/5B21qj5f6v47/6wH+c//c7X21h/38J2+sXvB3OQx+JWf7w7v+h7fW3//0s7cWfz3/y0//9/H/8sK2P//u8Xw6D34nF/3f3/38f0tq/P+7L8/n/+0f2f/yXv/73/X8J3h2DwW/g7/t/gP/99371sC33/5f383n+938f38d/9T+/hP9r2Hcw+JWc/V+G//0f/7Uf31h77N91n/z3/75g/+P//Pvf3vjff7sCg1+JWfn3/v/hH9jaL//j3X9+wf6n9rZf/i/B/+u7Cwz+JWf9d2r/w/v4f+y/q/hP//f3/sP/rP7D/8/7Dga/kqv6S05/x//gH6L+H/+s//Gf/b238B/mP/xX/4P3wOAXcnV+P+4//b/c+s/5v77f+n+u//s/+0+l//u//5z++P8ZDH4lV/VX1P6z/b//X/2H//W3/b+r/3V8Bwa/gqj63+p/Nf3R/n/bX+X+e/8g/8B/rNwz+JVerL+3/6L//7/2///X/w/+/f4PBb+Sq/qrUn/3f1b+/9f9i/wGD34lFf4f7T+m///7//772P7j3DQa/gqv6S2r/v/vf7j+8/5PB4FeS2l/K4BcSKP2pDk++M+7s11t/wVn/Xf+d+7sKBwa/kpMfev8/+X/4n//w7v//mQcMgr+S4P/4r/8T9n/5v/6v+/+f/xIMfrXz/6r//7z/8u//+b/v/+3//N/H/+P//Ekw+A+X/1v6H/Y/u/3f+v/f/+/23//2z/9//X/1t8HgP/v/2v9g//v/vf/7P//f//v//u///R0MfrUv6H/Y/xH67x8MfrV39e/1f3r/o/XfPxh8g3f1r3f/Gv7S/r+F/xIMfrVr+yv9n+6/9P5H+/8mDH61a/sn/Z/tf2n/p"

set_background(image_base64)


# --- The rest of your app code ---

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
