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
# CORRECTED THIS LINE: The long string is now enclosed in triple quotes (""")
image_base64 = """iVBORw0KGgoAAAANSUhEUgAAN4AAAC9CAMAAABua3sSAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAMAUExURQAAABIdLBYjNBglOBsoPBIcKhMdKyAmOhsoPB8qQiMsQR0sRikoPyIuTiYxVyYyWCg0Wic0Wig1XCo3Xis4YC06Yy48ZTJAajNCbDRDcTZFcjdGcztIdz5KepBJa5FKbZJLbpZOdJpQd5pReJ1Ufp5XgaBYgqJdhqZgirBnjLFojbJpjrZtkLhzlLp0mMB3m8N6n8R7oMZ9ocuDp8+HqNGJr9eMutqQv92Uwt+XxeKYxuOayOOay+SdzOWez+ag0Oih0uqj1Oul1uym1+2n2O+r2vCv3fGx3vSy4fWz4va04/i35fm55/q66fv88v399P7+/v///wYIDQYKDwYMERAXExMbGxshISMkJCcoKCctLS4xMTI1NTY4ODk7Ozw+Pj9AQUJERUZISUpMTU5QUVJUVVdZWVpcXF9hYWJjY2ZoaGlra2xtbW5wcHF0dHZ3d3h5eXp7fH1+fn+AgYSFhYeIiYqMjY6PkJGSk5SVlpeYmZqbnJ2en6ChoqOkpaanqKmqq6ytrq+wsbKztLa3t7i5uru8vb/AwcPFx8jKy8zNzs/Q0dLT1NXW19jb3N3f4OHi4+Tl5ufp6uvs7e7v8PHy8/T19vf4+fr7/P3+LwAAAAp0Uk5T/////////+AD3xN4AAAACXBIWXMAAA7DAAAOwwHHb6hkAAAgAElEQVR42uyde5xcVZX/P3+f02mmyQ13k+SG3WQhAbtJ3Ew24wZ3cQkI3QvEwS1dIohggQ0iQkQECyJIkQcEAoIiyI5CQAQREBAU3Ew2yQYbsGk3h8lMbrrT9JyZz/fH3Kk7s5Mmu2k20+3Mv+ebqapOddWd+j7z3VvX1S8W6I8Gg1/Jc//o/V2V9x8e8i/u/9P3hH6Sg2Dwc1n2M4227n987/5B21qj5f6v47/6wH+c//c7X21h/38J2+sXvB3OQx+JWf7w7v+h7fW3//0s7cWfz3/y0//9/H/8sK2P//u8Xw6D34nF/3f3/38f0tq/P+7L8/n/+0f2f/yXv/73/X8J3h2DwW/g7/t/gP/99371sC33/5f383n+938f38d/9T+/hP9r2Hcw+JWc/V+G//0f/7Uf31h77N91n/z3/75g/+P//Pvf3vjff7sCg1+JWfn3/v/hH9jaL//j3X9+wf6n9rZf/i/B/+u7Cwz+JWf9d2r/w/v4f+y/q/hP//f3/sP/rP7D/8/7Dga/kqv6S05/x//gH6L+H/+s//Gf/b238B/mP/xX/4P3wOAXcnV+P+4//b/c+s/5v77f+n+u//s/+0+l//u//5z++P8ZDH4lV/VX1P6z/b//X/2H//W3/b+r/3V8Bwa/gqj63+p/Nf3R/n/bX+X+e/8g/8B/rNwz+JVerL+3/6L//7/2///X/w/+/f4PBb+Sq/qrUn/3f1b+/9f9i/wGD34lFf4f7T+m///7//772P7j3DQa/gqv6S2r/v/vf7j+8/5PB4FeS2l/K4BcSKP2pDk++M+7s11t/wVn/Xf+d+7sKBwa/kpMfev8/+X/4n//w7v//mQcMgr+S4P/4r/8T9n/5v/6v+/+f/xIMfrXz/6r//7z/8u//+b/v/+3//N/H/+P//Ekw+A+X/1v6H/Y/u/3f+v/f/+/23//2z/9//X/1t8HgP/v/2v9g//v/vf/7P//f//v//u///R0MfrUv6H/Y/xH67x8MfrV39e/1f3r/o/XfPxh8g3f1r3f/Gv7S/r+F/xIMfrVr+yv9n+6/9P5H+/8mDH61a/sn/Z/tf2n/p"""
set_background(image_base64)
local_css()
model, X = load_and_train()

# --- App Title ---
st.markdown('<p class="main-title">Indian House Price Predictor</p>', unsafe_allow_html=True)
st.write('Use the sliders in the sidebar to set the features of a house and see its estimated price.')

# --- Sidebar Inputs ---
st.sidebar.header('🏠 Input Features')
st.sidebar.markdown("Adjust the values below to get a price prediction.")

num_bedrooms = st.sidebar.slider('Number of Bedrooms', int(X['number of bedrooms'].min()), int(X['number of bedrooms'].max()), int(X['number of bedrooms'].mean()))
num_bathrooms = st.sidebar.slider('Number of Bathrooms', float(X['number of bathrooms'].min()), 10.0, float(X['number of bathrooms'].mean()))
living_area = st.sidebar.slider('Living Area (sq ft)', int(X['living area'].min()), int(X['living area'].max()), int(X['living area'].mean()))
lot_area = st.sidebar.slider('Lot Area (sq ft)', int(X['lot area'].min()), int(X['lot area'].max()), int(X['lot area'].mean()))
num_floors = st.sidebar.slider('Number of Floors', float(X['number of floors'].min()), 5.0, float(X['number of floors'].mean()))
built_year = st.sidebar.slider('Year Built', int(X['Built Year'].min()), int(X['Built Year'].max()), int(X['Built Year'].mean()))
num_schools = st.sidebar.slider('Number of Schools Nearby', int(X['Number of schools nearby'].min()), int(X['Number of schools nearby'].max()), int(X['Number of schools nearby'].mean()))
distance_airport = st.sidebar.slider('Distance from Airport (km)', int(X['Distance from the airport'].min()), int(X['Distance from the airport'].max()), int(X['Distance from the airport'].mean()))

# --- Create Input DataFrame ---
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
    'Area of the house(excluding basement)': living_area, # Approximation
    'Area of the basement': int(X['Area of the basement'].mean()),
    'Built Year': built_year,
    'Renovation Year': int(X['Renovation Year'].mean()),
    'Lattitude': float(X['Lattitude'].mean()),
    'Longitude': float(X['Longitude'].mean()),
    'living_area_renov': living_area, # Approximation
    'lot_area_renov': lot_area, # Approximation
    'Number of schools nearby': num_schools,
    'Distance from the airport': distance_airport
}
input_df = pd.DataFrame(data, index=[0])

# --- Prediction and Display ---
prediction = model.predict(input_df)

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Your Selections")
    st.write(f"**Bedrooms:** {num_bedrooms}")
    st.write(f"**Bathrooms:** {num_bathrooms}")
    st.write(f"**Living Area:** {living_area} sq ft")
    st.write(f"**Lot Area:** {lot_area} sq ft")
    st.write(f"**Year Built:** {built_year}")
    st.write(f"**Schools Nearby:** {num_schools}")

with col2:
    st.write("") # Spacer
    st.write("") # Spacer
    st.markdown(
        f"""
        <div class="prediction-box">
            <p class="prediction-label">Predicted House Price</p>
            <p class="prediction-value">₹{prediction[0]:,.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Expander for More Info ---
with st.expander("ℹ️ About this App"):
    st.write("""
        This application uses a **Linear Regression model** to predict house prices based on the features you select. 
        
        The model was trained on a dataset of house sales in India. Please note that this is an estimation, and actual market prices can vary based on many other factors.
    """)
    st.write("**Dataset:** House Price India.csv")
    st.write("**Model:** Scikit-learn LinearRegression")
