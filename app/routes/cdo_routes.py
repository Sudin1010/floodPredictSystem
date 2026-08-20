from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.auth import require_cdo
from app.database.connection import get_db
from app.ml import BASE_FEATURES
from app.services import run_flood_prediction, save_prediction_history

router = APIRouter(prefix="/cdo")

BASE_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def get_submitted_values(form) -> dict[str, str]:
    return {
        field: form.get(field, "")
        for field in BASE_FEATURES
    }


def get_assigned_district_error(current_user) -> str | None:
    if not current_user.district:
        return "This CDO account requires an assigned district before district prediction can be created."
    return None


@router.get("/dashboard", response_class=HTMLResponse)
async def cdo_dashboard(request: Request, db: Session = Depends(get_db)):
    current_user = require_cdo(request, db)
    if isinstance(current_user, RedirectResponse):
        return current_user

    return templates.TemplateResponse(
        request=request,
        name="cdo_dashboard.html",
        context={
            "title": "CDO Dashboard",
            "url_for": request.url_for,
            "current_user": current_user,
        },
    )


@router.get("/predict", response_class=HTMLResponse)
async def cdo_predict_form(request: Request, db: Session = Depends(get_db)):
    current_user = require_cdo(request, db)
    if isinstance(current_user, RedirectResponse):
        return current_user

    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
            "title": "CDO Flood Risk Prediction",
            "page_heading": "CDO Flood Risk Prediction",
            "page_intro": "Predictions created here apply to your assigned district.",
            "form_action": "/cdo/predict",
            "submit_label": "Analyze Flood Risk",
            "reset_url": "/cdo/predict",
            "back_url": "/cdo/dashboard",
            "back_label": "Back to CDO Dashboard",
            "district_label": current_user.district,
            "district_error": get_assigned_district_error(current_user),
            "values": {},
            "error": None,
            "current_user": current_user,
            "is_cdo_prediction": True,
        },
    )


@router.post("/predict", response_class=HTMLResponse)
async def cdo_predict_result(request: Request, db: Session = Depends(get_db)):
    current_user = require_cdo(request, db)
    if isinstance(current_user, RedirectResponse):
        return current_user

    district_error = get_assigned_district_error(current_user)
    form = await request.form()
    form_values = get_submitted_values(form)

    if district_error:
        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                "title": "CDO Flood Risk Prediction",
                "page_heading": "CDO Flood Risk Prediction",
                "page_intro": "Predictions created here apply to your assigned district.",
                "form_action": "/cdo/predict",
                "submit_label": "Analyze Flood Risk",
                "reset_url": "/cdo/predict",
                "back_url": "/cdo/dashboard",
                "back_label": "Back to CDO Dashboard",
                "district_label": current_user.district,
                "district_error": district_error,
                "values": form_values,
                "error": None,
                "current_user": current_user,
                "is_cdo_prediction": True,
            },
            status_code=400,
        )

    try:
        prediction_result = run_flood_prediction(form)
        save_prediction_history(
            db=db,
            raw_values=prediction_result["raw_values"],
            probability=prediction_result["probability"],
            risk_level=prediction_result["risk_level"],
            user_id=current_user.id,
            district=current_user.district,
            prediction_source="cdo",
        )

        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                "title": "CDO Flood Risk Prediction",
                "page_heading": "CDO Flood Risk Prediction",
                "page_intro": "Predictions created here apply to your assigned district.",
                "form_action": "/cdo/predict",
                "submit_label": "Analyze Flood Risk",
                "reset_url": "/cdo/predict",
                "back_url": "/cdo/dashboard",
                "back_label": "Back to CDO Dashboard",
                "district_label": current_user.district,
                "result_district": current_user.district,
                "prediction": prediction_result["risk_level"],
                "probability": prediction_result["probability"],
                "risk_level": prediction_result["risk_level"],
                "risk_class": prediction_result["risk_class"],
                "risk_explanation": prediction_result["risk_explanation"],
                "recommendation": prediction_result["recommendation"],
                "predicted_class": prediction_result["predicted_class"],
                "classification_label": prediction_result["classification_label"],
                "values": form_values,
                "error": None,
                "current_user": current_user,
                "is_cdo_prediction": True,
            },
        )

    except Exception as exc:
        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                "title": "CDO Flood Risk Prediction",
                "page_heading": "CDO Flood Risk Prediction",
                "page_intro": "Predictions created here apply to your assigned district.",
                "form_action": "/cdo/predict",
                "submit_label": "Analyze Flood Risk",
                "reset_url": "/cdo/predict",
                "back_url": "/cdo/dashboard",
                "back_label": "Back to CDO Dashboard",
                "district_label": current_user.district,
                "error": str(exc),
                "values": form_values,
                "current_user": current_user,
                "is_cdo_prediction": True,
            },
        )
