import streamlit as st
import os
import subprocess

# Install scikit-learn
subprocess.run(['pip', 'install', 'scikit-learn'])

import numpy as np 
from sklearn.preprocessing import StandardScaler
import joblib
st.set_page_config(layout="wide")

scaler= joblib.load("Scaler.pkl")
scaler.n_features_in_ = 4
st.title("Resturant Rating Prediction App")


st.caption("This app helps you to predict a restaurant rating review class based on the features you provide")

averagecost = st.number_input("Average Cost for two", min_value=50, max_value=50000, value=1000, step=200)

tableBooking = st.selectbox("Table Booking ?", ["Yes", "No"])

OnlineDelivery = st.selectbox("Online Delivery ?", ["Yes", "No"])

pricerange = st.selectbox("Price Range (1 Cheapest ,4 Most Expensive)", ["1", "2", "3", "4"])

predictionbutton=st.button("Predict the Rating Class")
st.divider()

model=joblib.load("mlmodel.pkl")

bookingstatus=1 if tableBooking=="Yes" else 0
deliverystatus=1 if OnlineDelivery=="Yes" else 0    



values = [[averagecost, bookingstatus, deliverystatus, pricerange]]
my_X_values = np.array(values)


X = scaler.fit_transform(my_X_values)

# Make the prediction
if predictionbutton: 
    st.snow()
    prediction = model.predict(X)
    st.write(prediction)
    if prediction <2.5:
        st.write("The restaurant rating is Bad")
    elif prediction <3.5:
        st.write("The resturant rating is Average")
    elif prediction <4.0:
        st.write("The resturant rating is Good")
    else:
        st.write("The resturant rating is Excellent")   