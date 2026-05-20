from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database.connection import get_db
from app.services import get_user_prediction_summary

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

    history_rows, total_predictions, latest_prediction = get_user_prediction_summary(
        db,
        current_user.id,
    )

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
