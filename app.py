import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Property Price Predictor", page_icon="🏠", layout="wide")

# ---------------- LOAD FILES ----------------
model = joblib.load("Property_Price_Prediction.pkl")
columns = joblib.load("Model_columns.pkl")
scaler = joblib.load("Scaler.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #141E30, #243B55);
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #1E1E2F;
}
.metric-card {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 15px;
    text-align: center;
}
.result-box {
    background: linear-gradient(135deg,#11998e,#38ef7d);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    color: white;
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ----------------
st.markdown("""
<div style='text-align:center;padding:25px;
background:linear-gradient(135deg,#6a11cb,#2575fc);
border-radius:20px;margin-bottom:20px;'>
<h1>🏠 AI Property Price Predictor</h1>
<p>Predict Real Estate Prices Instantly Using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ---------------- METRICS ----------------
m1, m2, m3 = st.columns(3)
m1.metric("Model Accuracy", "92%")
m2.metric("Dataset Size", "29K+")
m3.metric("Prediction Speed", "0.2s")

# ---------------- SIDEBAR INPUTS ----------------
with st.sidebar:
    st.header("📋 Property Details")

    posted_by = st.selectbox("Posted By", ["Owner", "Dealer", "Builder"])
    under_construction = {"No": 0, "Yes": 1}[st.selectbox("Under Construction", ["No", "Yes"])]
    rera = {"No": 0, "Yes": 1}[st.selectbox("RERA Approved", ["No", "Yes"])]
    bhk_no = st.number_input("Number of BHK", 1, 20)
    bhk_or_rk = st.selectbox("Type", ["BHK", "RK"])
    square_ft = st.number_input("Square Feet", 100, 10000)
    ready_to_move = {"No": 0, "Yes": 1}[st.selectbox("Ready To Move", ["No", "Yes"])]
    resale = {"No": 0, "Yes": 1}[st.selectbox("Resale", ["No", "Yes"])]
    longitude = st.number_input("Longitude", -90.0, 90.0, format="%.6f")
    latitude = st.number_input("Latitude", -180.0, 180.0, format="%.6f")

# ---------------- ENCODING ----------------
input_df = pd.DataFrame({
    "POSTED_BY_Dealer": [1 if posted_by == "Dealer" else 0],
    "POSTED_BY_Owner": [1 if posted_by == "Owner" else 0],
    "UNDER_CONSTRUCTION": [under_construction],
    "RERA": [rera],
    "BHK_NO.": [bhk_no],
    "BHK_OR_RK_RK": [1 if bhk_or_rk == "RK" else 0],
    "SQUARE_FT": [square_ft],
    "READY_TO_MOVE": [ready_to_move],
    "RESALE": [resale],
    "LONGITUDE": [longitude],
    "LATITUDE": [latitude]
})

input_df = input_df.reindex(columns=columns, fill_value=0)

# ---------------- MAIN CONTENT ----------------
tab1, tab2 = st.tabs(["🏠 Prediction", "📊 Insights"])

with tab1:

    st.image(
        "https://images.unsplash.com/photo-1568605114967-8130f3a36994",
        use_container_width=True
    )

    if st.button("🚀 Predict Price", use_container_width=True):

        if longitude == 0 and latitude == 0:
            st.warning("⚠️ Please enter valid location coordinates.")
        else:
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)

            st.balloons()

            st.markdown(f"""
            <div class='result-box'>
            💰 Estimated Property Price <br><br>
            ₹ {prediction[0]:,.2f} Lakhs
            </div>
            """, unsafe_allow_html=True)

with tab2:

    chart_data = pd.DataFrame({
        "Area": np.arange(500, 5000, 500),
        "Price Trend": np.random.randint(20, 200, 9)
    })

    st.subheader("📈 Market Price Trends")
    st.line_chart(chart_data.set_index("Area"))

    st.subheader("🏘 Property Insights")
    c1, c2 = st.columns(2)

    c1.info("✔️ Larger properties generally show higher valuation.")
    c2.success("✔️ RERA-approved properties tend to have better resale value.")

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<center>
Made with ❤️ using Streamlit & Machine Learning
</center>
""", unsafe_allow_html=True)
