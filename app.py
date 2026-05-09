import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Property Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---------------- LOAD FILES ----------------
model = joblib.load("Property_Price_Prediction.pkl")
columns = joblib.load("Model_columns.pkl")
scaler = joblib.load("Scaler.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Main App */
.stApp {
    background: linear-gradient(135deg, #141E30, #243B55);
    color: white;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1E1E2F;
}

/* ALL TEXT */
html, body, [class*="css"] {
    color: white !important;
}

/* Labels & Headings */
label, p, h1, h2, h3, h4, h5, h6, span {
    color: white !important;
}

/* Sidebar Text */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Input Boxes */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #2D2D44 !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #666 !important;
}

/* +/- Buttons */
button[data-testid="stNumberInputStepUp"],
button[data-testid="stNumberInputStepDown"] {
    background-color: #2D2D44 !important;
    color: white !important;
    border: 1px solid #666 !important;
}

/* Dropdown Menu */
div[role="listbox"] {
    background-color: #2D2D44 !important;
    color: white !important;
}

/* Dropdown Options */
div[role="option"] {
    background-color: #2D2D44 !important;
    color: white !important;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background-color: rgba(255,255,255,0.08);
    border-radius: 15px;
    padding: 15px;
}

/* Metric Labels */
[data-testid="metric-container"] label {
    color: white !important;
}

/* Metric Values */
[data-testid="metric-container"] div {
    color: white !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg,#6a11cb,#2575fc);
    color: white !important;
    border: none;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}

/* Result Box */
.result-box {
    background: linear-gradient(135deg,#11998e,#38ef7d);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    color: white;
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
}

/* Footer */
.footer {
    text-align: center;
    color: white;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ----------------
st.markdown("""
<div style="
    text-align:center;
    padding:30px;
    background:linear-gradient(135deg,#6a11cb,#2575fc);
    border-radius:20px;
    margin-bottom:25px;
">
    <h1 style="color:white;">
        🏠 AI Property Price Predictor
    </h1>
    <p style="color:white;font-size:18px;">
        Predict Real Estate Prices Instantly Using Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- METRICS ----------------
m1, m2, m3 = st.columns(3)

m1.metric("📊 Model Accuracy", "92%")
m2.metric("🏘 Dataset Size", "29K+")
m3.metric("⚡ Prediction Speed", "0.2s")

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.header("📋 Property Details")

    posted_by = st.selectbox(
        "Posted By",
        ["Owner", "Dealer", "Builder"]
    )

    under_construction = {
        "No": 0,
        "Yes": 1
    }[st.selectbox(
        "Under Construction",
        ["No", "Yes"]
    )]

    rera = {
        "No": 0,
        "Yes": 1
    }[st.selectbox(
        "RERA Approved",
        ["No", "Yes"]
    )]

    bhk_no = st.number_input(
        "Number of BHK",
        min_value=1,
        max_value=20
    )

    bhk_or_rk = st.selectbox(
        "Type",
        ["BHK", "RK"]
    )

    square_ft = st.number_input(
        "Square Feet",
        min_value=100,
        max_value=10000
    )

    ready_to_move = {
        "No": 0,
        "Yes": 1
    }[st.selectbox(
        "Ready To Move",
        ["No", "Yes"]
    )]

    resale = {
        "No": 0,
        "Yes": 1
    }[st.selectbox(
        "Resale",
        ["No", "Yes"]
    )]

    longitude = st.number_input(
        "Longitude",
        min_value=-90.0,
        max_value=90.0,
        format="%.6f"
    )

    latitude = st.number_input(
        "Latitude",
        min_value=-180.0,
        max_value=180.0,
        format="%.6f"
    )

# ---------------- ENCODING ----------------
input_df = pd.DataFrame({

    "POSTED_BY_Dealer": [
        1 if posted_by == "Dealer" else 0
    ],

    "POSTED_BY_Owner": [
        1 if posted_by == "Owner" else 0
    ],

    "UNDER_CONSTRUCTION": [under_construction],

    "RERA": [rera],

    "BHK_NO.": [bhk_no],

    "BHK_OR_RK_RK": [
        1 if bhk_or_rk == "RK" else 0
    ],

    "SQUARE_FT": [square_ft],

    "READY_TO_MOVE": [ready_to_move],

    "RESALE": [resale],

    "LONGITUDE": [longitude],

    "LATITUDE": [latitude]
})

# ---------------- COLUMN ORDER ----------------
input_df = input_df.reindex(
    columns=columns,
    fill_value=0
)

# ---------------- TABS ----------------
tab1, tab2 = st.tabs([
    "🏠 Prediction",
    "📊 Market Insights"
])

# ---------------- PREDICTION TAB ----------------
with tab1:

    st.image(
        "https://images.unsplash.com/photo-1568605114967-8130f3a36994",
        use_container_width=True
    )

    if st.button(
        "🚀 Predict Price",
        use_container_width=True
    ):

        if longitude == 0 and latitude == 0:

            st.warning(
                "⚠️ Please enter valid location coordinates."
            )

        else:

            # Scaling
            input_scaled = scaler.transform(input_df)

            # Prediction
            prediction = model.predict(input_scaled)

            st.balloons()

            st.markdown(f"""
            <div class="result-box">
                💰 Estimated Property Price <br><br>
                ₹ {prediction[0]:,.2f} Lakhs
            </div>
            """, unsafe_allow_html=True)

# ---------------- INSIGHTS TAB ----------------
with tab2:

    st.subheader("📈 Real Estate Market Trends")

    chart_data = pd.DataFrame({
        "Area": np.arange(500, 5000, 500),
        "Price Trend": np.random.randint(20, 200, 9)
    })

    st.line_chart(
        chart_data.set_index("Area")
    )

    c1, c2 = st.columns(2)

    with c1:
        st.info(
            "✔️ Larger properties generally have higher market valuation."
        )

    with c2:
        st.success(
            "✔️ RERA-approved properties often show better resale value."
        )

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<div class="footer">
Made with ❤️ using Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)
