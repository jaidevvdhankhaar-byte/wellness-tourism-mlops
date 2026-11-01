
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Wellness Tourism Predictor",
    page_icon="✈️"
)

st.title("Visit with Us")
st.subheader("Wellness Tourism Package Prediction")

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "model_building/artifacts/best_model.pkl"
DATA_PATH = BASE_DIR / "data/tourism.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
data = load_data()

def get_options(column):
    return sorted(data[column].dropna().unique().tolist())

with st.form("customer_form"):

    st.write("Enter customer details")

    customer = {
        "Age": st.number_input("Age", 18, 100, 35),
        "TypeofContact": st.selectbox("Contact Type", get_options("TypeofContact")),
        "CityTier": st.selectbox("City Tier", [1, 2, 3]),
        "Occupation": st.selectbox("Occupation", get_options("Occupation")),
        "Gender": st.selectbox("Gender", get_options("Gender")),
        "NumberOfPersonVisiting": st.number_input("Visitors", 1, 20, 2),
        "PreferredPropertyStar": st.selectbox("Preferred Hotel Stars", [3, 4, 5]),
        "MaritalStatus": st.selectbox("Marital Status", get_options("MaritalStatus")),
        "NumberOfTrips": st.number_input("Annual Trips", 0, 50, 3),
        "Passport": st.selectbox("Passport", [0, 1]),
        "OwnCar": st.selectbox("Own Car", [0, 1]),
        "NumberOfChildrenVisiting": st.number_input("Children Below 5", 0, 10, 0),
        "Designation": st.selectbox("Designation", get_options("Designation")),
        "MonthlyIncome": st.number_input("Monthly Income", 0, 1000000, 25000)
    }

    submitted = st.form_submit_button("Predict Purchase")

if submitted:

    customer_df = pd.DataFrame([customer])

    probability = model.predict_proba(customer_df)[0][1]

    st.metric("Purchase Probability", f"{probability:.2%}")

    if probability >= 0.5:
        st.success("High predicted purchase likelihood!")
    else:
        st.info("Lower predicted purchase likelihood.")

    st.caption("Prediction is an estimate, not a purchase guarantee.")
