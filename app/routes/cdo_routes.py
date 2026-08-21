from pathlib import Path
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth import require_cdo
from app.constants import NEPAL_DISTRICTS
from app.database.connection import get_db
from app.database.models import Alert, AlertSubscription, PredictionHistory, User
from app.ml import BASE_FEATURES
from app.services import run_flood_prediction, save_prediction_history, send_alert_email

router = APIRouter(prefix="/cdo")

BASE_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def format_datetime(value) -> str:
    if value is None:
        return "Not available"

    day = value.strftime("%d").lstrip("0")
    month_year = value.strftime("%b %Y")
    time_value = value.strftime("%I:%M %p").lstrip("0")
    return f"{day} {month_year}, {time_value}"


def format_chart_datetime(value) -> str:
    if value is None:
        return "Not available"

    day = value.strftime("%d").lstrip("0")
    month = value.strftime("%b")
    time_value = value.strftime("%I:%M %p").lstrip("0")
    return f"{day} {month}, {time_value}"


templates.env.filters["format_datetime"] = format_datetime


def get_submitted_values(form) -> dict[str, str]:
    return {
        field: form.get(field, "")
        for field in BASE_FEATURES
    }


def get_cdo_prediction_district_error(district: str) -> str | None:
    if not district:
        return "District is required."
    if district not in NEPAL_DISTRICTS:
        return "Please select a valid district."
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


def get_matching_subscriber_emails(db: Session, district: str) -> list[str]:
    return db.scalars(
        select(User.email)
        .select_from(AlertSubscription)
        .join(User, User.id == AlertSubscription.user_id)
        .where(
            AlertSubscription.district == district,
            AlertSubscription.is_active.is_(True),
            AlertSubscription.email_enabled.is_(True),
            User.role == "user",
        )
        .order_by(User.id)
    ).all()


def get_alert_by_prediction(db: Session, prediction_id: int) -> Alert | None:
    return db.scalar(
        select(Alert).where(Alert.prediction_id == prediction_id)
    )


def get_authorized_alert_or_404(
    db: Session,
    current_user: User,
    alert_id: int,
    *,
    lock_for_update: bool = False,
) -> Alert:
    statement = (
        select(Alert)
        .join(PredictionHistory, PredictionHistory.id == Alert.prediction_id)
        .where(
            Alert.id == alert_id,
            Alert.created_by == current_user.id,
            Alert.district == PredictionHistory.district,
            PredictionHistory.prediction_source == "cdo",
            PredictionHistory.risk_level == "High Risk",
            PredictionHistory.user_id == current_user.id,
        )
    )
    if lock_for_update:
        statement = statement.with_for_update(of=Alert)

    alert = db.scalar(statement)
    if alert is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert draft not found.",
        )
    return alert


def build_default_alert_title(prediction: PredictionHistory) -> str:
    return f"Flood Risk Alert - {prediction.district}"


def build_default_alert_message(prediction: PredictionHistory) -> str:
    return (
        "A high flood risk has been identified.\n\n"
        "Residents are advised to remain alert and follow instructions from the responsible authorities."
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
        "sent_count": alert.sent_count if alert else 0,
        "failed_count": alert.failed_count if alert else 0,
        "sent_at": alert.sent_at if alert else None,
        "error": error,
        "success": success,
    }


def get_send_feedback(request: Request, alert: Alert | None) -> tuple[str | None, str | None]:
    send_status = request.query_params.get("send_status")
    if send_status == "already_sent":
        return None, "This alert has already been sent and cannot be sent again."
    if send_status == "not_draft":
        return None, "Only draft alerts can be sent."
    if send_status == "no_subscribers":
        return None, "No active email subscribers are available for this district."
    if send_status == "delivery_disabled":
        return None, "Email delivery is disabled. No real emails were sent and the alert remains a draft."
    if send_status == "sent" and alert is not None:
        return f"Alert sent successfully to {alert.sent_count} subscriber(s).", None
    if send_status == "partial_failed" and alert is not None:
        return None, f"Alert partially sent. Successful: {alert.sent_count}. Failed: {alert.failed_count}."
    if send_status == "failed" and alert is not None:
        return None, f"Alert delivery failed for all {alert.failed_count} subscriber(s)."
    return None, None


def redirect_to_alert_review(prediction_id: int, send_status: str) -> RedirectResponse:
    return RedirectResponse(
        url=f"/cdo/predictions/{prediction_id}/alert?send_status={send_status}",
        status_code=303,
    )


def get_recent_cdo_predictions(db: Session, user_id: int, limit: int = 5) -> list[PredictionHistory]:
    return db.scalars(
        select(PredictionHistory)
        .where(
            PredictionHistory.user_id == user_id,
            PredictionHistory.prediction_source == "cdo",
        )
        .order_by(PredictionHistory.created_at.desc(), PredictionHistory.id.desc())
        .limit(limit)
    ).all()


def get_latest_cdo_prediction(db: Session, user_id: int) -> PredictionHistory | None:
    return db.scalar(
        select(PredictionHistory)
        .where(
            PredictionHistory.user_id == user_id,
            PredictionHistory.prediction_source == "cdo",
        )
        .order_by(PredictionHistory.created_at.desc(), PredictionHistory.id.desc())
        .limit(1)
    )


def get_cdo_trend_predictions(
    db: Session,
    user_id: int,
    district: str,
    limit: int = 10,
) -> list[PredictionHistory]:
    newest_first = db.scalars(
        select(PredictionHistory)
        .where(
            PredictionHistory.user_id == user_id,
            PredictionHistory.prediction_source == "cdo",
            PredictionHistory.district == district,
        )
        .order_by(PredictionHistory.created_at.desc(), PredictionHistory.id.desc())
        .limit(limit)
    ).all()
    return list(reversed(newest_first))


def build_trend_chart_data(predictions: list[PredictionHistory]) -> dict:
    return {
        "labels": [format_chart_datetime(row.created_at) for row in predictions],
        "probabilities": [row.probability for row in predictions],
        "risk_levels": [row.risk_level for row in predictions],
        "full_dates": [format_datetime(row.created_at) for row in predictions],
    }


@router.get("/dashboard", response_class=HTMLResponse)
async def cdo_dashboard(request: Request, db: Session = Depends(get_db)):
    current_user = require_cdo(request, db)
    if isinstance(current_user, RedirectResponse):
        return current_user

    requested_district = request.query_params.get("district", "").strip()
    latest_cdo_prediction = get_latest_cdo_prediction(db, current_user.id)
    recent_predictions = get_recent_cdo_predictions(db, current_user.id, limit=5)
    has_cdo_predictions = latest_cdo_prediction is not None
    canonical_districts = list(NEPAL_DISTRICTS)
    selected_district = ""
    district_error = None

    if requested_district:
        if requested_district in canonical_districts:
            selected_district = requested_district
        else:
            district_error = "Please select a valid district."
    elif latest_cdo_prediction:
        latest_prediction_district = (latest_cdo_prediction.district or "").strip()
        if latest_prediction_district in canonical_districts:
            selected_district = latest_prediction_district

    trend_predictions = []
    if selected_district:
        trend_predictions = get_cdo_trend_predictions(
            db,
            current_user.id,
            selected_district,
        )

    return templates.TemplateResponse(
        request=request,
        name="cdo_dashboard.html",
        context={
            "title": "CDO Dashboard",
            "url_for": request.url_for,
            "current_user": current_user,
            "dashboard_districts": canonical_districts,
            "district_count": len(canonical_districts),
            "selected_district": selected_district or "",
            "district_error": district_error,
            "has_cdo_predictions": has_cdo_predictions,
            "trend_predictions": trend_predictions,
            "trend_chart_data": build_trend_chart_data(trend_predictions),
            "recent_predictions": recent_predictions,
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
            "page_intro": "Select the district to analyse.",
            "form_action": "/cdo/predict",
            "submit_label": "Analyze Flood Risk",
            "reset_url": "/cdo/predict",
            "back_url": "/cdo/dashboard",
            "back_label": "Back to CDO Dashboard",
            "districts": NEPAL_DISTRICTS,
            "selected_district": "",
            "district_error": None,
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

    form = await request.form()
    form_values = get_submitted_values(form)
    district = form.get("district", "").strip()
    district_error = get_cdo_prediction_district_error(district)

    if district_error:
        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                "title": "CDO Flood Risk Prediction",
                "page_heading": "CDO Flood Risk Prediction",
                "page_intro": "Select the district to analyse.",
                "form_action": "/cdo/predict",
                "submit_label": "Analyze Flood Risk",
                "reset_url": "/cdo/predict",
                "back_url": "/cdo/dashboard",
                "back_label": "Back to CDO Dashboard",
                "districts": NEPAL_DISTRICTS,
                "selected_district": district,
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
            district=district,
            prediction_source="cdo",
        )

        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                "title": "CDO Flood Risk Prediction",
                "page_heading": "CDO Flood Risk Prediction",
                "page_intro": "Select the district to analyse.",
                "form_action": "/cdo/predict",
                "submit_label": "Analyze Flood Risk",
                "reset_url": "/cdo/predict",
                "back_url": "/cdo/dashboard",
                "back_label": "Back to CDO Dashboard",
                "districts": NEPAL_DISTRICTS,
                "selected_district": district,
                "result_district": district,
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
                "page_intro": "Select the district to analyse.",
                "form_action": "/cdo/predict",
                "submit_label": "Analyze Flood Risk",
                "reset_url": "/cdo/predict",
                "back_url": "/cdo/dashboard",
                "back_label": "Back to CDO Dashboard",
                "districts": NEPAL_DISTRICTS,
                "selected_district": district,
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
    success, send_error = get_send_feedback(request, alert)
    if error is None:
        error = send_error

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
            success=success,
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
    elif error is None and alert is not None and alert.status != "draft":
        error = "Only draft alerts can be edited."

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


@router.post("/alerts/{alert_id}/send", response_class=HTMLResponse)
async def send_cdo_alert(
    alert_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    current_user = require_cdo(request, db)
    if isinstance(current_user, RedirectResponse):
        return current_user

    alert = get_authorized_alert_or_404(
        db,
        current_user,
        alert_id,
        lock_for_update=True,
    )
    prediction = alert.prediction

    if alert.status == "sent":
        return redirect_to_alert_review(prediction.id, "already_sent")

    if alert.status != "draft":
        return redirect_to_alert_review(prediction.id, "not_draft")

    recipient_emails = get_matching_subscriber_emails(db, alert.district)
    if not recipient_emails:
        return redirect_to_alert_review(prediction.id, "no_subscribers")

    sent_count = 0
    failed_count = 0
    delivery_disabled = False

    for recipient_email in recipient_emails:
        result = send_alert_email(
            recipient_email=recipient_email,
            alert_title=alert.title,
            alert_message=alert.message,
            district=alert.district,
            probability=alert.probability,
            risk_level=alert.risk_level,
        )
        if result.disabled:
            delivery_disabled = True
            break
        if result.success:
            sent_count += 1
        else:
            failed_count += 1

    if delivery_disabled:
        return redirect_to_alert_review(prediction.id, "delivery_disabled")

    alert.sent_count = sent_count
    alert.failed_count = failed_count
    alert.sent_at = datetime.now(timezone.utc)

    if sent_count == len(recipient_emails):
        alert.status = "sent"
    elif sent_count > 0:
        alert.status = "partial_failed"
    else:
        alert.status = "failed"

    db.commit()
    db.refresh(alert)

    return redirect_to_alert_review(prediction.id, alert.status)
