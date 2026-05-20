from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models import PredictionHistory

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def format_datetime(value) -> str:
    if value is None:
        return "Not available"

    day = value.strftime("%d").lstrip("0")
    month_year = value.strftime("%b %Y")
    time_value = value.strftime("%I:%M %p").lstrip("0")
    return f"{day} {month_year} • {time_value}"


templates.env.filters["format_datetime"] = format_datetime


@router.get("/history", response_class=HTMLResponse)
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
