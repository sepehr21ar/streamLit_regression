import streamlit as st
import pandas as pd
import numpy as np
from keras.saving import load_model
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
import joblib
import locale

# Set up the Streamlit app
st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏠 House Price Prediction")

# Load the dataset
@st.cache_data
def load_dataset():
    """
    Load the house price dataset.
    """
    cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
    df = pd.read_csv("https://raw.githubusercontent.com/emanhamed/Houses-dataset/master/Houses%20Dataset/HousesInfo.txt", sep=" ", header=None, names=cols)
    # Convert zip codes to strings and strip whitespace
    df["zipcode"] = df["zipcode"].astype(str).str.strip()
    return df

# Load the dataset
df = load_dataset()

# Load the trained model and preprocessing tools
@st.cache_resource
def load_trained_model():
    """
    Load the trained model and preprocessing tools.
    """

    model = load_model("house_price_model.keras")
    cs = joblib.load("minmax_scaler.pkl")
    zipBinarizer = joblib.load("zip_binarizer.pkl")
    maxPrice = joblib.load("max_price.pkl")
    zipBinarizer.classes_ = [str(zipcode).strip() for zipcode in zipBinarizer.classes_]
    return model, cs, zipBinarizer, maxPrice

# Load the model and preprocessing tools
model, cs, zipBinarizer, maxPrice = load_trained_model()

# Input fields for user data
st.sidebar.header("Enter House Details")
bedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.sidebar.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
area = st.sidebar.number_input("Area (sqft)", min_value=500, max_value=10000, value=2000)

# Get valid zip codes from the LabelBinarizer
valid_zip_codes = zipBinarizer.classes_

# Use a dropdown for zip code input
zipcode = st.sidebar.selectbox("Zip Code", valid_zip_codes)

# Predict button
if st.sidebar.button("Predict Price"):
    try:
        # Convert zip code to string and strip whitespace
        zipcode = str(zipcode).strip()

        # Check if the zip code is valid
        if zipcode not in valid_zip_codes:
            st.error(f"Invalid zip code: {zipcode}. Please enter a valid zip code.")
        else:
            # Scale continuous features
            continuous_features = np.array([[bedrooms, bathrooms, area]])
            continuous_scaled = cs.transform(continuous_features)

            # One-hot encode zip code
            zipcode_encoded = zipBinarizer.transform([zipcode])

            # Combine features
            input_data = np.hstack([zipcode_encoded, continuous_scaled])

            # Predict the price
            predicted_price = model.predict(input_data) * maxPrice

            # Display the result
            st.success(f"### Predicted House Price: **${predicted_price[0][0]:,.2f}**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Display some statistics
st.subheader("Model Statistics")
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
st.write("**Average House Price:**", locale.currency(df["price"].mean(), grouping=True))
st.write("**Standard Deviation of House Price:**", locale.currency(df["price"].std(), grouping=True))
