from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.auth import require_cdo
from app.database.connection import get_db

router = APIRouter(prefix="/cdo")

BASE_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=BASE_DIR / "templates")


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
