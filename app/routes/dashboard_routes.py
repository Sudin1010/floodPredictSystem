from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database.connection import get_db

router = APIRouter()

@router.get("/history")
async def prediction_history(request: Request, db: Session = Depends(get_db)):
    current_user = get_current_user(request, db)
    if current_user is None:
        return RedirectResponse(url="/login", status_code=303)
    if current_user.role == "cdo":
        return RedirectResponse(url="/cdo/dashboard", status_code=303)

    return RedirectResponse(url="/subscription", status_code=303)
