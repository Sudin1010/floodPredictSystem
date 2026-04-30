import os
import joblib
import pandas as pd
import streamlit as st
import numpy as np

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="Flood Prediction System",
    page_icon="🌊",
    layout="wide"
)

# ---------------- LOAD MODEL FILES ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models", "logistic_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "models", "features.pkl"))

# ---------------- HEADER ----------------
st.title("🌊 Flood Prediction System")
st.write(
    "A machine learning-based system that predicts flood risk using environmental, "
    "land, drainage, and infrastructure-related factors."
)

st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.title("Project Information")
st.sidebar.write("""
**Model Used:** Logistic Regression  
**Tech Stack:** Python, Scikit-learn, Streamlit  
**Output:** Low / High Flood Risk
""")

st.sidebar.info(
    "Engineered features such as RainFactor, LandRisk, WaterStress, and Blockage "
    "are calculated automatically by the system."
)

# ---------------- INPUT SECTION ----------------
st.subheader("Enter Flood Risk Factors")

input_data = {}

with st.expander("🌧️ Rainfall & Climate Factors", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        input_data["MonsoonIntensity"] = st.slider("Monsoon Intensity", 0.0, 10.0, 5.0, 0.1)

    with col2:
        input_data["ClimateChange"] = st.slider("Climate Change", 0.0, 10.0, 5.0, 0.1)


with st.expander("🏞️ Land & Human Activity Factors", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        input_data["Deforestation"] = st.slider("Deforestation", 0.0, 10.0, 5.0, 0.1)

    with col2:
        input_data["Urbanization"] = st.slider("Urbanization", 0.0, 10.0, 5.0, 0.1)

    with col3:
        input_data["Encroachments"] = st.slider("Encroachments", 0.0, 10.0, 5.0, 0.1)


with st.expander("🌊 River, Drainage & Water System Factors"):
    col1, col2, col3 = st.columns(3)

    with col1:
        input_data["RiverManagement"] = st.slider("River Management", 0.0, 10.0, 5.0, 0.1)

    with col2:
        input_data["DrainageSystems"] = st.slider("Drainage Systems", 0.0, 10.0, 5.0, 0.1)

    with col3:
        input_data["DamsQuality"] = st.slider("Dams Quality", 0.0, 10.0, 5.0, 0.1)


with st.expander("⛰️ Soil, Terrain & Blockage Factors"):
    col1, col2, col3 = st.columns(3)

    with col1:
        input_data["Siltation"] = st.slider("Siltation", 0.0, 10.0, 5.0, 0.1)

    with col2:
        input_data["Landslides"] = st.slider("Landslides", 0.0, 10.0, 5.0, 0.1)

    with col3:
        input_data["TopographyDrainage"] = st.slider("Topography Drainage", 0.0, 10.0, 5.0, 0.1)


with st.expander("🏗️ Preparedness & Infrastructure Factors"):
    col1, col2, col3 = st.columns(3)

    with col1:
        input_data["IneffectiveDisasterPreparedness"] = st.slider(
            "Ineffective Disaster Preparedness", 0.0, 10.0, 5.0, 0.1
        )

    with col2:
        input_data["DeterioratingInfrastructure"] = st.slider(
            "Deteriorating Infrastructure", 0.0, 10.0, 5.0, 0.1
        )

    with col3:
        input_data["InadequatePlanning"] = st.slider(
            "Inadequate Planning", 0.0, 10.0, 5.0, 0.1
        )


with st.expander("👥 Population & Environmental Factors"):
    col1, col2, col3 = st.columns(3)

    with col1:
        input_data["PopulationScore"] = st.slider("Population Score", 0.0, 10.0, 5.0, 0.1)

    with col2:
        input_data["WetlandLoss"] = st.slider("Wetland Loss", 0.0, 10.0, 5.0, 0.1)

    with col3:
        input_data["Watersheds"] = st.slider("Watersheds", 0.0, 10.0, 5.0, 0.1)


with st.expander("🌾 Agricultural Factor"):
    input_data["AgriculturalPractices"] = st.slider(
        "Agricultural Practices", 0.0, 10.0, 5.0, 0.1
    )
    # Apply same log transformation used during training
base_features = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement",
    "Deforestation", "Urbanization", "ClimateChange", "DamsQuality",
    "Siltation", "AgriculturalPractices", "Encroachments",
    "IneffectiveDisasterPreparedness", "DrainageSystems",
    "Landslides", "Watersheds", "DeterioratingInfrastructure",
    "PopulationScore", "WetlandLoss", "InadequatePlanning"
]

for feature in base_features:
    input_data[feature] = np.log1p(input_data[feature])

# ---------------- AUTOMATIC FEATURE ENGINEERING ----------------
input_data["RainFactor"] = (
    input_data["MonsoonIntensity"] * input_data["ClimateChange"]
) / 10

input_data["LandRisk"] = (
    input_data["Deforestation"]
    + input_data["Urbanization"]
    + input_data["Encroachments"]
) / 3

input_data["WaterStress"] = (
    input_data["RiverManagement"]
    + input_data["DrainageSystems"]
    + input_data["DamsQuality"]
) / 3

input_data["Blockage"] = (
    input_data["Siltation"] + input_data["Landslides"]
) / 2

# ---------------- PREDICTION ----------------
st.divider()

if st.button("Predict Flood Risk", type="primary"):

    input_df = pd.DataFrame([input_data])

    # Keep same feature order used during training
    input_df = input_df[features]

    # Scale input
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Flood Risk")
    else:
        st.success("✅ Low Flood Risk")

    st.metric("Flood Risk Probability", f"{probability * 100:.2f}%")

    if probability < 0.4:
        st.info("Risk Level: Low")
    elif probability < 0.7:
        st.warning("Risk Level: Medium")
    else:
        st.error("Risk Level: High")

    with st.expander("View Input Summary"):
        st.dataframe(input_df)

    with st.expander("View Automatically Generated Features"):
        engineered_df = pd.DataFrame([{
            "RainFactor": input_data["RainFactor"],
            "LandRisk": input_data["LandRisk"],
            "WaterStress": input_data["WaterStress"],
            "Blockage": input_data["Blockage"]
        }])
        st.dataframe(engineered_df)