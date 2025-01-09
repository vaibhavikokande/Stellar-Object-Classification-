import streamlit as st
import pandas as pd
from joblib import load

# Load the trained Random Forest model
model = load('stellar_model.joblib')

# CSS for background image
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1444703686981-a3abbc4d4fe3");
    background-size: cover;
    background-attachment: fixed;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Create a Streamlit app
st.title("Stellar Object Prediction")

# Input fields for feature values on the main screen
st.header("Stellar Object Information")
z = st.number_input("Infrared filter in the photometric system")
cam_col = st.selectbox("Camera column to identify the scanline within the run", ('1', '2', '3', '4', '5', '6'))
i = st.number_input("Near Infrared filter in the photometric system")
delta = st.number_input("Declination angle (at J2000 epoch)")
MJD = st.number_input("Modified Julian Date, used to indicate when a given piece of SDSS data was taken")
plate = st.number_input("Plate ID, identifies each plate in SDSS")
spec_obj_ID = st.number_input("Unique ID used for optical spectroscopic objects")
alpha = st.number_input("Right Ascension angle (at J2000 epoch)")
field_ID = st.number_input("Field number to identify each field")

# Map input values to numeric using the label mapping
label_mapping = {
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6
}
cam_col = label_mapping[cam_col]

# Make a prediction using the model
prediction = model.predict([[z, cam_col, i, delta, MJD, plate, spec_obj_ID, alpha, field_ID]])

# Display the prediction result on the main screen
st.header("Prediction Result")

# Define the outputs for the three classes
output_labels = {
    0: "GALAXY.",
    1: "STAR",
    2: "QSO"
}

# Display the appropriate message based on the prediction
predicted_class = prediction[0]  # Assuming prediction is a list or array with the predicted class
if predicted_class in output_labels:
    st.info(output_labels[predicted_class])
else:
    st.warning("Unexpected prediction result.")
