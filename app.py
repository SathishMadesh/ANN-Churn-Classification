import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StanderdScaler, LabelEncoder, OneHotEncoder
import pickle

# Load the trained model
model = tf.keras.models.load_model('churn_model.h5')

## Load the scaler and encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)


## Streamlit App
st.title("Customer Churn Prediction")

# Input fields
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance", min_value=0.0)
num_of_products = st.slider("Number of Products", 1, 4)
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary", min_value=0.0)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
tenure = st.slider("Tenure", 0, 10)

## Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Tenure': [tenure]
})

#OneHot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Concatenate One Hot Encoded Geography with input data

input_data = pd.concat([input_data, reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]

# Display result
if st.button("Predict Churn"):
    if prediction_prob > 0.5:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is unlikely to churn.")