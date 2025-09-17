import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load the dataset
@st.cache_data
def load_data():
 df = pd.read_csv('Housing.csv')
 return df

df = load_data()

# Preprocess the data
def preprocess_data(df):
 # Handle categorical variables
 categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
 le = LabelEncoder()
 for col in categorical_cols:
  df[col] = le.fit_transform(df[col])
  return df

df = preprocess_data(df.copy())

# Split the data into features (X) and target (y)
X = df.drop('price', axis=1)
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Create the Streamlit app
st.title('🏠 House Price Prediction App')
st.write('This app predicts the price of a house based on its features.')

# Create the sidebar for user input
st.sidebar.header('Input Features')

def user_input_features():
 area = st.sidebar.slider('Area (in sq. ft.)', int(df.area.min()), int(df.area.max()), int(df.area.mean()))
 bedrooms = st.sidebar.slider('Bedrooms', int(df.bedrooms.min()), int(df.bedrooms.max()), int(df.bedrooms.mean()))
 bathrooms = st.sidebar.slider('Bathrooms', int(df.bathrooms.min()), int(df.bathrooms.max()), int(df.bathrooms.mean()))
 stories = st.sidebar.slider('Stories', int(df.stories.min()), int(df.stories.max()), int(df.stories.mean()))
 mainroad = st.sidebar.selectbox('Mainroad', ('yes', 'no'))
 guestroom = st.sidebar.selectbox('Guestroom', ('yes', 'no'))
 basement = st.sidebar.selectbox('Basement', ('yes', 'no'))
 hotwaterheating = st.sidebar.selectbox('Hot Water Heating', ('yes', 'no'))
 airconditioning = st.sidebar.selectbox('Air Conditioning', ('yes', 'no'))
 parking = st.sidebar.slider('Parking', int(df.parking.min()), int(df.parking.max()), int(df.parking.mean()))
 prefarea = st.sidebar.selectbox('Preferred Area', ('yes', 'no'))
 furnishingstatus = st.sidebar.selectbox('Furnishing Status', ('furnished', 'semi-furnished', 'unfurnished'))

 # Convert categorical inputs to numerical
 mainroad = 1 if mainroad == 'yes' else 0
 guestroom = 1 if guestroom == 'yes' else 0
 basement = 1 if basement == 'yes' else 0
 hotwaterheating = 1 if hotwaterheating == 'yes' else 0
 airconditioning = 1 if airconditioning == 'yes' else 0
 prefarea = 1 if prefarea == 'yes' else 0
 if furnishingstatus == 'furnished':
 furnishingstatus = 0
 elif furnishingstatus == 'semi-furnished':
 furnishingstatus = 1
 else:
 furnishingstatus = 2


 data = {'area': area,
 'bedrooms': bedrooms,
 'bathrooms': bathrooms,
 'stories': stories,
 'mainroad': mainroad,
 'guestroom': guestroom,
 'basement': basement,
 'hotwaterheating': hotwaterheating,
 'airconditioning': airconditioning,
 'parking': parking,
 'prefarea': prefarea,
 'furnishingstatus': furnishingstatus}
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
st.write(f'The predicted price of the house is: **${prediction[0]:,.2f}**')
