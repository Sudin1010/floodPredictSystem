import os
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from app.auth import authenticate_user, get_current_user, get_user_by_username_or_email, hash_password
from app.database import Base, engine, get_db
from app.models import PredictionHistory, User

app = FastAPI(title="Flood Prediction System")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "change-this-development-secret")

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY,
    same_site="lax",
    https_only=False,
)

# Static files
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Templates
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def format_datetime(value) -> str:
    if value is None:
        return "Not available"

    day = value.strftime("%d").lstrip("0")
    month_year = value.strftime("%b %Y")
    time_value = value.strftime("%I:%M %p").lstrip("0")
    return f"{day} {month_year} • {time_value}"


templates.env.filters["format_datetime"] = format_datetime

# Load trained ANN model
MODEL_PATH = PROJECT_ROOT / "models" / "ann_scratch_model.pkl"

model_data = joblib.load(MODEL_PATH)

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


@app.on_event("startup")
def create_database_tables() -> None:
    Base.metadata.create_all(bind=engine)
    ensure_prediction_history_user_id()


def ensure_prediction_history_user_id() -> None:
    inspector = inspect(engine)
    if not inspector.has_table("prediction_history"):
        return

    columns = {column["name"] for column in inspector.get_columns("prediction_history")}
    if "user_id" not in columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE prediction_history "
                    "ADD COLUMN user_id INTEGER NULL REFERENCES users(id)"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS "
                    "ix_prediction_history_user_id ON prediction_history (user_id)"
                )
            )


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


def get_submitted_values(form) -> Dict[str, str]:
    """
    Keep raw browser form values available for redisplay after submit.
    """
    return {field: form.get(field, "") for field in BASE_FEATURES}


def validate_raw_inputs(form) -> Dict[str, float]:
    """
    Validate browser form values before log1p transformation.
    """
    values = {}

    for field in BASE_FEATURES:
        raw_value = form.get(field)

        if raw_value is None or raw_value == "":
            raise ValueError(f"{field} is required.")

        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a numeric value.") from exc

        if not np.isfinite(value):
            raise ValueError(f"{field} must be a finite number.")

        if value < 0:
            raise ValueError(f"{field} must be greater than or equal to 0.")

        values[field] = value

    return values


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

    if missing:
        raise ValueError(f"Missing model features: {missing}")

    if extra:
        raise ValueError(f"Unexpected model features: {extra}")


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


def save_prediction_history(
    db: Session,
    raw_values: Dict[str, float],
    probability: float,
    risk_level: str,
    user_id: int | None = None,
) -> None:
    history = PredictionHistory(
        **raw_values,
        probability=probability,
        risk_level=risk_level,
        user_id=user_id,
    )
    db.add(history)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, db: Session = Depends(get_db)):
    current_user = get_current_user(request, db)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": "Flood Prediction System",
            "url_for": request.url_for,
            "current_user": current_user,
        },
    )


@app.get("/predict", response_class=HTMLResponse)
async def predict_form(request: Request, db: Session = Depends(get_db)):
    current_user = get_current_user(request, db)
    if current_user is None:
        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
            "title": "Flood Risk Prediction",
            "url_for": request.url_for,
            "values": {},
            "current_user": current_user,
        },
    )


@app.get("/history", response_class=HTMLResponse)
async def prediction_history(request: Request, db: Session = Depends(get_db)):
    current_user = get_current_user(request, db)
    if current_user is None:
        return RedirectResponse(url="/login", status_code=303)

    history_rows = db.scalars(
        select(PredictionHistory)
        .where(PredictionHistory.user_id == current_user.id)
        .order_by(PredictionHistory.created_at.desc(), PredictionHistory.id.desc())
        .limit(20)
    ).all()
    total_predictions = db.scalar(
        select(func.count())
        .select_from(PredictionHistory)
        .where(PredictionHistory.user_id == current_user.id)
    )
    latest_prediction = history_rows[0] if history_rows else None

    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={
            "title": "Prediction History",
            "url_for": request.url_for,
            "history_rows": history_rows,
            "total_predictions": total_predictions or 0,
            "latest_prediction": latest_prediction,
            "current_user": current_user,
        },
    )


@app.get("/register", response_class=HTMLResponse)
async def register_form(request: Request, db: Session = Depends(get_db)):
    current_user = get_current_user(request, db)
    if current_user is not None:
        return RedirectResponse(url="/predict", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="register.html",
        context={
            "title": "Register",
            "url_for": request.url_for,
            "current_user": current_user,
            "error": None,
        },
    )


@app.post("/register", response_class=HTMLResponse)
async def register_user(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    username = form.get("username", "").strip()
    email = form.get("email", "").strip().lower()
    password = form.get("password", "")
    confirm_password = form.get("confirm_password", "")

    error = None
    if not username or not email or not password or not confirm_password:
        error = "All fields are required."
    elif len(password) < 8:
        error = "Password must be at least 8 characters."
    elif password != confirm_password:
        error = "Passwords do not match."
    elif get_user_by_username_or_email(db, username) is not None:
        error = "Username is already registered."
    elif get_user_by_username_or_email(db, email) is not None:
        error = "Email is already registered."

    if error:
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "title": "Register",
                "url_for": request.url_for,
                "current_user": None,
                "error": error,
                "username": username,
                "email": email,
            },
        )

    user = User(
        username=username,
        email=email,
        password_hash=hash_password(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return RedirectResponse(url="/login?registered=1", status_code=303)


@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request, db: Session = Depends(get_db)):
    current_user = get_current_user(request, db)
    if current_user is not None:
        return RedirectResponse(url="/predict", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "title": "Login",
            "url_for": request.url_for,
            "current_user": current_user,
            "error": None,
            "success": "Registration successful. Please login."
            if request.query_params.get("registered") == "1"
            else None,
        },
    )


@app.post("/login", response_class=HTMLResponse)
async def login_user(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    username_or_email = form.get("username_or_email", "").strip()
    password = form.get("password", "")
    user = authenticate_user(db, username_or_email, password)

    if user is None:
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "title": "Login",
                "url_for": request.url_for,
                "current_user": None,
                "error": "Invalid username/email or password.",
                "username_or_email": username_or_email,
            },
        )

    request.session["user_id"] = user.id
    return RedirectResponse(url="/predict", status_code=303)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@app.post("/predict", response_class=HTMLResponse)
async def predict_result(request: Request, db: Session = Depends(get_db)):
    current_user = get_current_user(request, db)
    if current_user is None:
        return RedirectResponse(url="/login", status_code=303)

    form = await request.form()
    form_values = get_submitted_values(form)

    try:
        # 1. Collect 18 raw input features from browser form
        raw_values = validate_raw_inputs(form)

        # 2. Apply same log transformation used during training
        prediction_values = apply_log_transformation(raw_values.copy())

        # 3. Create 4 engineered features
        prediction_values = compute_derived_features(prediction_values)

        # 4. Check final feature alignment
        validate_features(prediction_values)

        # 5. Build 22-feature vector in correct order
        feature_vector = build_feature_vector(prediction_values)

        # 6. Scale features using saved training means/stds
        scaled_vector = scale_features(feature_vector)

        # 7. ANN prediction
        probability = float(forward_propagation(scaled_vector).ravel()[0])
        probability_percent = round(probability * 100, 2)

        if probability_percent < 40:
            risk_level = "Low Risk"
            risk_class = "low"
            risk_explanation = "Current conditions indicate a low possibility of flood risk."
            recommendation = "Continue regular monitoring."
        elif probability_percent < 70:
            risk_level = "Medium Risk"
            risk_class = "medium"
            risk_explanation = "Moderate flood risk detected."
            recommendation = "Increased monitoring and preparedness are recommended."
        else:
            risk_level = "High Risk"
            risk_class = "high"
            risk_explanation = "High flood risk detected."
            recommendation = "Immediate preparedness and safety measures are recommended."

        save_prediction_history(
            db=db,
            raw_values=raw_values,
            probability=probability_percent,
            risk_level=risk_level,
            user_id=current_user.id,
        )

        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                "title": "Flood Risk Prediction",
                "url_for": request.url_for,
                "prediction": risk_level,
                "probability": probability_percent,
                "risk_level": risk_level,
                "risk_class": risk_class,
                "risk_explanation": risk_explanation,
                "recommendation": recommendation,
                "values": form_values,
                "error": None,
                "current_user": current_user,
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
                "values": form_values,
                "current_user": current_user,
            },
        )
