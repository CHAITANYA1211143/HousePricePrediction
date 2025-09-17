import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="Indian House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    df = df.drop(['id', 'Date', 'Postal Code'], axis=1)
    df = df.fillna(df.mean())
    
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
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X

# --- Main App Logic ---

# Set background and styles
image_base64 = "iVBORw0KGgoAAAANSUhEUgAAN4AAAC9CAMAAABua3sSAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAMAUExURQAAABIdLBYjNBglOBsoPBIcKhMdKyAmOhsoPB8qQiMsQR0sRikoPyIuTiYxVyYyWCg0Wic0Wig1XCo3Xis4YC06Yy48ZTJAajNCbDRDcTZFcjdGcztIdz5KepBJa5FKbZJLbpZOdJpQd5pReJ1Ufp5XgaBYgqJdhqZgirBnjLFojbJpjrZtkLhzlLp0mMB3m8N6n8R7oMZ9ocuDp8+HqNGJr9eMutqQv92Uwt+XxeKYxuOayOOay+SdzOWez+ag0Oih0uqj1Oul1uym1+2n2O+r2vCv3fGx3vSy4fWz4va04/i35fm55/q66fv88v399P7+/v///wYIDQYKDwYMERAXExMbGxshISMkJCcoKCctLS4xMTI1NTY4ODk7Ozw+Pj9AQUJERUZISUpMTU5QUVJUVVdZWVpcXF9hYWJjY2ZoaGlra2xtbW5wcHF0dHZ3d3h5eXp7fH1+fn+AgYSFhYeIiYqMjY6PkJGSk5SVlpeYmZqbnJ2en6ChoqOkpaanqKmqq6ytrq+wsbKztLa3t7i5uru8vb/AwcPFx8jKy8zNzs/Q0dLT1NXW19jb3N3f4OHi4+Tl5ufp6uvs7e7v8PHy8/T19vf4+fr7/P3+LwAAAAp0Uk5T/////////+AD3xN4AAAACXBIWXMAAA7DAAAOwwHHb6hkAAAgAElEQVR42uyde5xcVZX/P3+f02mmyQ13k+SG3WQhAbtJ3Ew24wZ3cQkI3QvEwS1dIohggQ0iQkQECyJIkQcEAoIiyI5CQAQREBAU3Ew2yQYbsGk3h8lMbrrT9JyZz/fH3Kk7s5Mmu2k20+3Mv+ebqapOddWd+j7z3VvX1S8W6I8Gg1/Jc//o/V2V9x8e8i/u/9P3hH6Sg2Dwc1n2M4227n987/5B21qj5f6v47/6wH+c//c7X21himport streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="Indian House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    df = df.drop(['id', 'Date', 'Postal Code'], axis=1)
    df = df.fillna(df.mean())
    
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
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X

# --- Main App Logic ---

# Set background and styles
image_base64 = "iVBORw0KGgoAAAANSUhEUgAAN4AAAC9CAMAAABua3sSAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAMAUExURQAAABIdLBYjNBglOBsoPBIcKhMdKyAmOhsoPB8qQiMsQR0sRikoPyIuTiYxVyYyWCg0Wic0Wig1XCo3Xis4YC06Yy48ZTJAajNCbDRDcTZFcjdGcztIdz5KepBJa5FKbZJLbpZOdJpQd5pReJ1Ufp5XgaBYgqJdhqZgirBnjLFojbJpjrZtkLhzlLp0mMB3m8N6n8R7oMZ9ocuDp8+HqNGJr9eMutqQv92Uwt+XxeKYxuOayOOay+SdzOWez+ag0Oih0uqj1Oul1uym1+2n2O+r2vCv3fGx3vSy4fWz4va04/i35fm55/q66fv88v399P7+/v///wYIDQYKDwYMERAXExMbGxshISMkJCcoKCctLS4xMTI1NTY4ODk7Ozw+Pj9AQUJERUZISUpMTU5QUVJUVVdZWVpcXF9hYWJjY2ZoaGlra2xtbW5wcHF0dHZ3d3h5eXp7fH1+fn+AgYSFhYeIiYqMjY6PkJGSk5SVlpeYmZqbnJ2en6ChoqOkpaanqKmqq6ytrq+wsbKztLa3t7i5uru8vb/AwcPFx8jKy8zNzs/Q0dLT1NXW19jb3N3f4OHi4+Tl5ufp6uvs7e7v8PHy8/T19vf4+fr7/P3+LwAAAAp0Uk5T/////////+AD3xN4AAAACXBIWXMAAA7DAAAOwwHHb6hkAAAgAElEQVR42uyde5xcVZX/P3+f02mmyQ13k+SG3WQhAbtJ3Ew24wZ3cQkI3QvEwS1dIohggQ0iQkQECyJIkQcEAoIiyI5CQAQREBAU3Ew2yQYbsGk3h8lMbrrT9JyZz/fH3Kk7s5Mmu2k20+3Mv+ebqapOddWd+j7z3VvX1S8W6I8Gg1/Jc//o/V2V9x8e8i/u/9P3hH6Sg2Dwc1n2M4227n987/5B21qj5f6v47/6wH+c//c7X21h
