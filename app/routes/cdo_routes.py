from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth import require_cdo
from app.database.connection import get_db
from app.database.models import Alert, AlertSubscription, PredictionHistory, User
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


def get_cdo_prediction_or_404(
    db: Session,
    current_user: User,
    prediction_id: int,
) -> PredictionHistory:
    prediction = db.scalar(
        select(PredictionHistory).where(
            PredictionHistory.id == prediction_id,
            PredictionHistory.user_id == current_user.id,
            PredictionHistory.prediction_source == "cdo",
            PredictionHistory.district == current_user.district,
        )
    )
    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="CDO prediction not found.",
        )
    return prediction


def get_high_risk_alert_error(prediction: PredictionHistory) -> str | None:
    if prediction.risk_level != "High Risk":
        return "Public alert drafts can only be created from High Risk CDO predictions."
    return None


def get_matching_email_subscriber_count(db: Session, district: str) -> int:
    return db.scalar(
        select(func.count())
        .select_from(AlertSubscription)
        .join(User, User.id == AlertSubscription.user_id)
        .where(
            AlertSubscription.district == district,
            AlertSubscription.is_active.is_(True),
            AlertSubscription.email_enabled.is_(True),
            User.role == "user",
        )
    ) or 0


def get_alert_by_prediction(db: Session, prediction_id: int) -> Alert | None:
    return db.scalar(
        select(Alert).where(Alert.prediction_id == prediction_id)
    )


def build_default_alert_title(prediction: PredictionHistory) -> str:
    return f"Flood Risk Alert - {prediction.district}"


def build_default_alert_message(prediction: PredictionHistory) -> str:
    return (
        f"A high flood risk has been identified for {prediction.district} District.\n\n"
        f"Estimated flood probability: {prediction.probability}%\n"
        f"Risk level: {prediction.risk_level}\n\n"
        "Residents are advised to remain alert and follow instructions from the responsible authorities.\n\n"
        "This is a prototype flood prediction advisory and should support, not replace, official emergency information."
    )


def build_alert_context(
    request: Request,
    current_user: User,
    prediction: PredictionHistory,
    alert: Alert | None,
    matching_subscriber_count: int,
    *,
    title: str | None = None,
    message: str | None = None,
    error: str | None = None,
    success: str | None = None,
) -> dict:
    return {
        "title": "CDO Alert Review",
        "url_for": request.url_for,
        "current_user": current_user,
        "prediction": prediction,
        "alert": alert,
        "matching_subscriber_count": matching_subscriber_count,
        "alert_title": title if title is not None else (alert.title if alert else build_default_alert_title(prediction)),
        "alert_message": message if message is not None else (alert.message if alert else build_default_alert_message(prediction)),
        "status": alert.status if alert else "draft",
        "error": error,
        "success": success,
    }


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
        history = save_prediction_history(
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
                "alert_prediction_id": history.id
                if prediction_result["risk_level"] == "High Risk"
                else None,
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


@router.get("/predictions/{prediction_id}/alert", response_class=HTMLResponse)
async def cdo_alert_review(
    prediction_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    current_user = require_cdo(request, db)
    if isinstance(current_user, RedirectResponse):
        return current_user

    prediction = get_cdo_prediction_or_404(db, current_user, prediction_id)
    alert = get_alert_by_prediction(db, prediction.id)
    matching_subscriber_count = get_matching_email_subscriber_count(
        db,
        prediction.district,
    )
    error = get_high_risk_alert_error(prediction)

    return templates.TemplateResponse(
        request=request,
        name="cdo_alert_review.html",
        context=build_alert_context(
            request,
            current_user,
            prediction,
            alert,
            matching_subscriber_count,
            error=error,
        ),
        status_code=400 if error else 200,
    )


@router.post("/predictions/{prediction_id}/alert", response_class=HTMLResponse)
async def save_cdo_alert_draft(
    prediction_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    current_user = require_cdo(request, db)
    if isinstance(current_user, RedirectResponse):
        return current_user

    prediction = get_cdo_prediction_or_404(db, current_user, prediction_id)
    alert = get_alert_by_prediction(db, prediction.id)
    matching_subscriber_count = get_matching_email_subscriber_count(
        db,
        prediction.district,
    )
    high_risk_error = get_high_risk_alert_error(prediction)

    form = await request.form()
    title = form.get("title", "").strip()
    message = form.get("message", "").strip()

    error = high_risk_error
    if error is None and not title:
        error = "Alert title is required."
    elif error is None and len(title) > 200:
        error = "Alert title must be 200 characters or fewer."
    elif error is None and not message:
        error = "Alert message is required."
    elif error is None and len(message) > 3000:
        error = "Alert message must be 3000 characters or fewer."

    if error:
        return templates.TemplateResponse(
            request=request,
            name="cdo_alert_review.html",
            context=build_alert_context(
                request,
                current_user,
                prediction,
                alert,
                matching_subscriber_count,
                title=title,
                message=message,
                error=error,
            ),
            status_code=400,
        )

    if alert is None:
        alert = Alert(
            prediction_id=prediction.id,
            district=prediction.district,
            title=title,
            message=message,
            probability=prediction.probability,
            risk_level=prediction.risk_level,
            created_by=current_user.id,
            status="draft",
        )
        db.add(alert)
    else:
        alert.title = title
        alert.message = message
        alert.district = prediction.district
        alert.probability = prediction.probability
        alert.risk_level = prediction.risk_level
        alert.created_by = current_user.id
        alert.status = "draft"

    db.commit()
    db.refresh(alert)

    return templates.TemplateResponse(
        request=request,
        name="cdo_alert_review.html",
        context=build_alert_context(
            request,
            current_user,
            prediction,
            alert,
            matching_subscriber_count,
            success="Alert draft saved successfully. No notifications have been sent yet.",
        ),
    )
