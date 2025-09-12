import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
try:
    model = joblib.load('delivery_time_model.pkl')
except FileNotFoundError:
    st.error("Model file 'delivery_time_model.pkl' not found. Please make sure to train and save your model first.")
    st.stop()

# Set up the Streamlit app title and description
st.title('Food Delivery Time Prediction')
st.markdown('Enter the details of the delivery to predict the time it will take.')

# Define the exact columns from the training data, in the correct order
training_columns = [
    'Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition',
    'multiple_deliveries', 'distance_km', 'order_hour', 'order_day_of_week',
    'order_day_of_month', 'Weatherconditions_conditions Fog',
    'Weatherconditions_conditions NaN', 'Weatherconditions_conditions Sandstorms',
    'Weatherconditions_conditions Stormy', 'Weatherconditions_conditions Sunny',
    'Weatherconditions_conditions Windy', 'Road_traffic_density_Jam ',
    'Road_traffic_density_Low ', 'Road_traffic_density_Medium ',
    'Road_traffic_density_NaN ', 'Type_of_order_Drinks ', 'Type_of_order_Meal ',
    'Type_of_order_Snack ', 'Type_of_vehicle_electric_scooter ',
    'Type_of_vehicle_motorcycle ', 'Type_of_vehicle_scooter ', 'Festival_No ',
    'Festival_Yes ', 'City_NaN ', 'City_Semi-Urban ', 'City_Urban '
]

# Create input widgets for user data
st.header("Delivery Person & Vehicle")
col1, col2 = st.columns(2)
with col1:
    delivery_person_age = st.slider('Delivery Person Age', min_value=18, max_value=50, value=25)
    delivery_person_ratings = st.slider('Delivery Person Ratings', min_value=1.0, max_value=5.0, value=4.5)
with col2:
    vehicle_condition = st.number_input('Vehicle Condition (0-3)', min_value=0, max_value=3, value=1)
    multiple_deliveries = st.radio('Multiple Deliveries?', [0, 1])

st.header("Order & Environment")
col3, col4 = st.columns(2)
with col3:
    distance_km = st.number_input('Distance from Restaurant (km)', min_value=0.1, max_value=50.0, value=5.0)
    order_hour = st.slider('Order Hour (0-23)', min_value=0, max_value=23, value=12)
    order_day_of_week = st.slider('Day of the Week (0=Mon, 6=Sun)', min_value=0, max_value=6, value=2)
    order_day_of_month = st.slider('Day of the Month (1-31)', min_value=1, max_value=31, value=15)
with col4:
    weather_conditions = st.selectbox('Weather Conditions', ['Sunny', 'Stormy', 'Cloudy', 'Windy', 'Fog', 'Sandstorms', 'NaN'])
    traffic_density = st.selectbox('Road Traffic Density', ['Low', 'Medium', 'High', 'Jam', 'NaN'])
    order_type = st.selectbox('Type of Order', ['Snack', 'Meal', 'Drinks', 'Buffet'])
    vehicle_type = st.selectbox('Type of Vehicle', ['scooter', 'motorcycle', 'electric_scooter', 'bicycle'])
    festival = st.selectbox('Festival', ['No', 'Yes'])
    city = st.selectbox('City', ['Urban', 'Metropolitian', 'Semi-Urban', 'NaN'])

# Create a button to trigger the prediction
if st.button('Predict Delivery Time'):
    # Prepare the input data in the same format as the training data
    input_data = pd.DataFrame(columns=training_columns)
    input_data.loc[0] = 0

    # Populate the input data with user values
    input_data['Delivery_person_Age'] = delivery_person_age
    input_data['Delivery_person_Ratings'] = delivery_person_ratings
    input_data['Vehicle_condition'] = vehicle_condition
    input_data['distance_km'] = distance_km
    input_data['multiple_deliveries'] = multiple_deliveries
    input_data['order_hour'] = order_hour
    input_data['order_day_of_week'] = order_day_of_week
    input_data['order_day_of_month'] = order_day_of_month

    # Set the one-hot encoded values based on user input
    weather_key = f'Weatherconditions_conditions {weather_conditions}'
    if weather_conditions == 'NaN':
        input_data['Weatherconditions_conditions NaN'] = 1
    elif weather_key in input_data.columns:
        input_data[weather_key] = 1

    traffic_key = f'Road_traffic_density_{traffic_density} '
    if traffic_density == 'NaN':
        input_data['Road_traffic_density_NaN '] = 1
    elif traffic_key in input_data.columns:
        input_data[traffic_key] = 1

    order_key = f'Type_of_order_{order_type} '
    if order_key in input_data.columns:
        input_data[order_key] = 1

    vehicle_key = f'Type_of_vehicle_{vehicle_type} '
    if vehicle_key in input_data.columns:
        input_data[vehicle_key] = 1

    festival_key = f'Festival_{festival} '
    if festival_key in input_data.columns:
        input_data[festival_key] = 1

    city_key = f'City_{city} '
    if city == 'NaN':
        input_data['City_NaN '] = 1
    elif city_key in input_data.columns:
        input_data[city_key] = 1
    
    # Make the prediction
    prediction = model.predict(input_data)
    
    # Display the result
    st.success(f'Predicted delivery time is: {prediction[0]:.2f} minutes')