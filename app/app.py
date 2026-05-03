import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st



# ANN Helper Functions


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def forward_propagation(X_data, parameters):
    Z1 = X_data @ parameters["W1"] + parameters["b1"]
    A1 = relu(Z1)

    Z2 = A1 @ parameters["W2"] + parameters["b2"]
    A2 = relu(Z2)

    Z3 = A2 @ parameters["W3"] + parameters["b3"]
    A3 = sigmoid(Z3)

    return A3



# Preprocessing Pipeline


def preprocess_user_input(raw_df, model_package):
    df = raw_df.copy()

    # 1. Log transformation same as 02_Transformation.ipynb
    for col in df.columns:
        df[col] = np.log1p(df[col])

    # 2. Drop same columns as 03_Feature_Engineering.ipynb
    drop_cols = ["CoastalVulnerability", "PoliticalFactors"]

    for col in drop_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # 3. Feature engineering same as notebook
    df["RainFactor"] = (
        df["MonsoonIntensity"] * df["ClimateChange"]
    )

    df["LandRisk"] = (
        df["Deforestation"] +
        df["Urbanization"] +
        df["Encroachments"]
    ) / 3

    df["WaterStress"] = (
        df["RiverManagement"] +
        df["DrainageSystems"] +
        df["DamsQuality"]
    ) / 3

    df["Blockage"] = (
        df["Siltation"] +
        df["Landslides"]
    ) / 2

    # 4. Arrange columns exactly like training
    features = model_package["features"]
    df = df[features]

    return df


def predict_ann(raw_input_df, model_package):
    parameters = model_package["parameters"]
    scaler = model_package["scaler"]
    threshold = model_package["prediction_threshold"]

    processed_df = preprocess_user_input(raw_input_df, model_package)

    input_scaled = scaler.transform(processed_df)

    probability = forward_propagation(input_scaled, parameters)[0][0]
    probability = float(np.clip(probability, 0, 1))

    prediction = 1 if probability >= threshold else 0

    return prediction, probability, processed_df



# Load Model


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "ann_scratch_model.pkl")

model_package = joblib.load(MODEL_PATH)


# Streamlit UI


st.set_page_config(
    page_title="Flood Prediction System",
    page_icon="🌊",
    layout="centered"
)

st.title("🌊 Flood Prediction System")

st.subheader("Enter Raw Environmental and Hydrological Values")

raw_features = [
    "MonsoonIntensity",
    "TopographyDrainage",
    "RiverManagement",
    "Deforestation",
    "Urbanization",
    "ClimateChange",
    "DamsQuality",
    "Siltation",
    "AgriculturalPractices",
    "Encroachments",
    "IneffectiveDisasterPreparedness",
    "DrainageSystems",
    "Landslides",
    "Watersheds",
    "DeterioratingInfrastructure",
    "PopulationScore",
    "WetlandLoss",
    "InadequatePlanning",
]

input_data = {}

for feature in raw_features:
    input_data[feature] = st.number_input(
        label=feature,
        min_value=0.0,
        max_value=20.0,
        value=4.9,
        step=0.1
    )

raw_input_df = pd.DataFrame([input_data])

st.write("### Raw Input Preview")
st.dataframe(raw_input_df)



# Prediction


if st.button("Predict Flood Risk"):
    prediction, probability, processed_df = predict_ann(raw_input_df, model_package)

    risk_percentage = probability * 100

    st.write("### Prediction Result")
    st.write(f"Flood Probability: **{risk_percentage:.2f}%**")

    st.write("### Risk Level")

    if probability < 0.40:
        st.success("✅ Low Flood Risk")
    elif probability < 0.60:
        st.warning("⚠️ Medium Flood Risk")
    else:
        st.error("🚨 High Flood Risk")

    st.write("### Final Decision")

    if prediction == 1:
        st.error("🚨 Flood Risk Detected")
    else:
        st.success("✅ Flood Risk Not Detected")

    with st.expander("Processed Input Used by Model"):
        st.dataframe(processed_df)



# Model Information


with st.expander("Model Information"):
    st.write("Model:", model_package["model_name"])
    st.write("Hidden Layers:", model_package["hidden_layers"])
    st.write("Activation Hidden:", model_package["activation_hidden"])
    st.write("Activation Output:", model_package["activation_output"])
    st.write("Optimizer:", model_package["optimizer"])
    st.write("Prediction Threshold:", model_package["prediction_threshold"])
    st.write("Feature Count:", model_package["feature_count"])