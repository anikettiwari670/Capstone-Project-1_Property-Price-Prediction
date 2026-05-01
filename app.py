import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Loading the inputs. 
model = joblib.load("Property_Price_Prediction.pkl")
columns = joblib.load("Model_columns.pkl")
scaler = joblib.load("Scaler.pkl")

# Setting up the page UI.
st.set_page_config(page_title = "Property Price Predictor", layout = "centered")

st.markdown("""
    <style>
    .stApp {
        background-color: #2E0854; /* Deep Purple */
    }
    h1, p, label {
        color: white !important; /* Makes title and labels white for contrast */
    }
    </style>
    """, unsafe_allow_html = True)

st.title("Property Price Prediction App")

st.write("Enter Property Details:")

# Defining the inputs. 
posted_by = st.selectbox("Posted By", ["Owner", "Dealer", "Builder"])
under_construction = {"No": 0, "Yes": 1}[st.selectbox("Under Construction", ["No", "Yes"])]
rera = {"No": 0, "Yes": 1}[st.selectbox("RERA Approved", ["No", "Yes"])]
bhk_no = st.number_input("Number of BHK", min_value = 1, max_value = 20)
bhk_or_rk = st.selectbox("Type", ["BHK", "RK"])
square_ft = st.number_input("Square Feet", min_value = 100, max_value = 10000)
ready_to_move = {"No": 0, "Yes": 1}[st.selectbox("Ready To Move", ["No", "Yes"])]
resale = {"No": 0, "Yes": 1}[st.selectbox("Resale", ["No", "Yes"])]
longitude = st.number_input("Longitude", min_value = -90.0, max_value = 90.0, format = "%.6f")
latitude = st.number_input("Latitude", min_value = -180.0, max_value = 180.0, format = "%.6f")

# Encoding for Posted_By column. 
POSTED_BY_Dealer = 1 if posted_by == "Dealer" else 0
POSTED_BY_Owner = 1 if posted_by == "Owner" else 0
# For Builder, both 0. 

# Encoding for BHK_OR_RK column. 
BHK_OR_RK_RK = 1 if bhk_or_rk == "RK" else 0
# For BHK, it is 0.

# Creating inputs. 
input_df = pd.DataFrame({
    "POSTED_BY_Dealer": [POSTED_BY_Dealer],
    "POSTED_BY_Owner": [POSTED_BY_Owner],
    "UNDER_CONSTRUCTION": [under_construction],
    "RERA": [rera],
    "BHK_NO.": [bhk_no],
    "BHK_OR_RK_RK": [BHK_OR_RK_RK],
    "SQUARE_FT": [square_ft],
    "READY_TO_MOVE": [ready_to_move],
    "RESALE": [resale],
    "LONGITUDE": [longitude],
    "LATITUDE": [latitude]
})

# Ensuring correct column order.
input_df = input_df.reindex(columns = columns, fill_value = 0)

if st.button("Predict Price"):
    
    # 🚨 Validation check.
    if longitude == 0 and latitude == 0:
        st.warning("Please enter valid location coordinates.")
    else:
        # Feature scaling.
        input_scaled = scaler.transform(input_df)

        # Making prediction on the scaled input features.
        prediction = model.predict(input_scaled)

    # Displaying the results. 
        st.balloons()
        st.markdown(f"""
        ### Result:
        The Estimated Market Price for this Property is: 💰 ₹ {prediction[0]:,.2f} Lakhs.""")
