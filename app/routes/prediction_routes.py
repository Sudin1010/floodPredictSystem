from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database.connection import get_db
from app.ml import (
    BASE_FEATURES,
    apply_log_transformation,
    build_feature_vector,
    compute_derived_features,
    map_risk_level,
    predict_probability,
    scale_features,
    validate_features,
    validate_raw_inputs,
)
from app.services import save_prediction_history

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def get_submitted_values(form) -> dict[str, str]:
    """
    Keep raw browser form values available for redisplay after submit.
    """
    return {field: form.get(field, "") for field in BASE_FEATURES}


@router.get("/", response_class=HTMLResponse)
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


@router.get("/predict", response_class=HTMLResponse)
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


@router.post("/predict", response_class=HTMLResponse)
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
        probability = predict_probability(scaled_vector)
        probability_percent = round(probability * 100, 2)

        risk_level, risk_class, risk_explanation, recommendation = map_risk_level(
            probability_percent
        )

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
