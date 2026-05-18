import os
from typing import Dict

import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Flood Prediction System")

# Static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="app/templates")

# Load trained ANN model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "ann_scratch_model.pkl"
)

model_data = joblib.load(os.path.abspath(MODEL_PATH))

FEATURES = model_data["features"]
FEATURE_MEANS = model_data["feature_means"]
FEATURE_STDS = model_data["feature_stds"]
PARAMETERS = model_data["parameters"]
PREDICTION_THRESHOLD = model_data.get("prediction_threshold", 0.5)

# Raw features shown in browser form
BASE_FEATURES = [
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


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def forward_propagation(X_data: np.ndarray) -> np.ndarray:
    Z1 = X_data @ PARAMETERS["W1"] + PARAMETERS["b1"]
    A1 = relu(Z1)

    Z2 = A1 @ PARAMETERS["W2"] + PARAMETERS["b2"]
    A2 = relu(Z2)

    Z3 = A2 @ PARAMETERS["W3"] + PARAMETERS["b3"]
    A3 = sigmoid(Z3)

    return A3


def apply_log_transformation(values: Dict[str, float]) -> Dict[str, float]:
    """
    Apply log1p transformation to raw input features.
    This must match the training pipeline.
    """
    for field in BASE_FEATURES:
        values[field] = float(np.log1p(values[field]))
    return values


def compute_derived_features(values: Dict[str, float]) -> Dict[str, float]:
    """
    Create engineered features used during model training.
    These 4 features convert 18 raw inputs into 22 final ANN inputs.
    """
    values["RainFactor"] = values["MonsoonIntensity"] * values["ClimateChange"]

    values["LandRisk"] = (
        values["Deforestation"]
        + values["Urbanization"]
        + values["Encroachments"]
    ) / 3

    values["WaterStress"] = (
        values["RiverManagement"]
        + values["DrainageSystems"]
        + values["DamsQuality"]
    ) / 3

    values["Blockage"] = (
        values["Siltation"]
        + values["Landslides"]
    ) / 2

    return values


def validate_features(values: Dict[str, float]) -> None:
    """
    Check whether backend features match trained model features.
    """
    missing = set(FEATURES) - set(values.keys())
    extra = set(values.keys()) - set(FEATURES)

    print("\n===== DEBUG CHECK =====")
    print("\nFORM + ENGINEERED FEATURES:")
    print(list(values.keys()))

    print("\nMODEL EXPECTED FEATURES:")
    print(FEATURES)

    print("\nMISSING FEATURES:")
    print(missing)

    print("\nEXTRA FEATURES:")
    print(extra)

    if missing:
        raise ValueError(f"Missing model features: {missing}")


def build_feature_vector(values: Dict[str, float]) -> np.ndarray:
    """
    Build final feature vector in the exact same order used during training.
    """
    return np.array([[values[name] for name in FEATURES]], dtype=float)


def scale_features(feature_vector: np.ndarray) -> np.ndarray:
    """
    Standard scaling using feature means and stds saved from training.
    """
    means = np.array([FEATURE_MEANS[name] for name in FEATURES], dtype=float)
    stds = np.array([FEATURE_STDS[name] for name in FEATURES], dtype=float)

    stds[stds == 0] = 1.0

    return (feature_vector - means) / stds


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": "Flood Prediction System",
            "url_for": request.url_for,
        },
    )


@app.get("/predict", response_class=HTMLResponse)
async def predict_form(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
            "title": "Flood Risk Prediction",
            "url_for": request.url_for,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_result(request: Request):
    form = await request.form()
    values = {}

    try:
        # 1. Collect 18 raw input features from browser form
        for field in BASE_FEATURES:
            raw_value = form.get(field)

            if raw_value is None or raw_value == "":
                raise ValueError(f"{field} is required.")

            values[field] = float(raw_value)

        # 2. Apply same log transformation used during training
        values = apply_log_transformation(values)

        # 3. Create 4 engineered features
        values = compute_derived_features(values)

        # 4. Check final feature alignment
        validate_features(values)

        # 5. Build 22-feature vector in correct order
        feature_vector = build_feature_vector(values)

        # 6. Scale features using saved training means/stds
        scaled_vector = scale_features(feature_vector)

        # 7. ANN prediction
        probability = float(forward_propagation(scaled_vector).ravel()[0])

        prediction_label = (
            "High Flood Risk"
            if probability >= PREDICTION_THRESHOLD
            else "Low Flood Risk"
        )

        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                "title": "Flood Risk Prediction",
                "url_for": request.url_for,
                "prediction": prediction_label,
                "probability": round(probability * 100, 2),
                "values": values,
                "error": None,
            },
        )

    except Exception as exc:
        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                "title": "Flood Risk Prediction",
                "url_for": request.url_for,
                "error": str(exc),
                "values": values,
            },
        )