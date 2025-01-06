import streamlit as st
import numpy as np
import pandas as pd
import pickle


regressor = pickle.load(open('new_forest_regressor.pkl', 'rb'))


def show_predict_page():
    st.title('Taxi Fare Prediction')

    st.image('Taxi image.png', width=200)

    ## Create user imput

    Distance_in_km = st.number_input("Distance in Km", min_value=1, max_value=None, value=None)
    Traffic_condition = st.selectbox("Traffic Condition", ['Low', 'High', 'Medium'])
    Weather = st.selectbox("Weather", ['Clear', 'Rain', 'Snow'])

    # mapping
    Traffic_condition_mapping = {'Low':1, 'High':0, 'Medium':2}
    Weather_mapping = {'Clear':0, 'Rain':1, 'Snow':2}

    # convert to numeric
    Traffic_condition_numeric = Traffic_condition_mapping[Traffic_condition]
    Weather_numeric = Weather_mapping[Weather]


    ## create prediction button
    ok = st.button("Submit")
    if ok:  
        # create np array
        x = np.array([[Distance_in_km, Traffic_condition_numeric, Weather_numeric]])

        # Predict using the loaded model
    
        prediction = regressor.predict(x)
        st.subheader(f"Taxi fare : ${np.round(prediction, 2)}")

