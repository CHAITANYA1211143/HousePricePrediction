import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the dataset
@st.cache_data
def load_data():
    # Column names for the California Housing dataset
    col_names = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income',
        'median_house_value'
    ]
    df = pd.read_csv('cal_housing.data', header=None, names=col_names)
    # The dataset has some missing values, we will fill them with the mean
    df = df.fillna(df.mean())
    return df

df = load_data()

# Split the data into features (X) and target (y)
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Create the Streamlit app
st.title('🏠 California House Price Prediction App')
st.write('This app predicts the median price of a house in California based on its features.')

# Create the sidebar for user input
st.sidebar.header('Input Features')

def user_input_features():
    longitude = st.sidebar.slider('Longitude', float(df.longitude.min()), float(df.longitude.max()), float(df.longitude.mean()))
    latitude = st.sidebar.slider('Latitude', float(df.latitude.min()), float(df.latitude.max()), float(df.latitude.mean()))
    housing_median_age = st.sidebar.slider('Housing Median Age', int(df.housing_median_age.min()), int(df.housing_median_age.max()), int(df.housing_median_age.mean()))
    total_rooms = st.sidebar.slider('Total Rooms', int(df.total_rooms.min()), int(df.total_rooms.max()), int(df.total_rooms.mean()))
    total_bedrooms = st.sidebar.slider('Total Bedrooms', int(df.total_bedrooms.min()), int(df.total_bedrooms.max()), int(df.total_bedrooms.mean()))
    population = st.sidebar.slider('Population', int(df.population.min()), int(df.population.max()), int(df.population.mean()))
    households = st.sidebar.slider('Households', int(df.households.min()), int(df.households.max()), int(df.households.mean()))
    median_income = st.sidebar.slider('Median Income', float(df.median_income.min()), float(df.median_income.max()), float(df.median_income.mean()))

    data = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the user input
st.subheader('User Input')
st.write(input_df)

# Predict the price
prediction = model.predict(input_df)

# Display the prediction
st.subheader('Prediction')
st.write(f'The predicted median price of the house is: **${prediction[0]:,.2f}**')
