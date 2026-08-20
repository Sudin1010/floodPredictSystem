from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.constants import NEPAL_DISTRICTS
from app.database.connection import get_db
from app.database.models import AlertSubscription, User

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def require_subscription_user(request: Request, db: Session) -> User | RedirectResponse:
    current_user = get_current_user(request, db)
    if current_user is None:
        return RedirectResponse(url="/login", status_code=303)

    if current_user.role != "user":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Normal user access required.",
        )

    return current_user


def get_user_subscription(db: Session, user_id: int) -> AlertSubscription | None:
    return db.scalar(
        select(AlertSubscription).where(AlertSubscription.user_id == user_id)
    )


def build_subscription_context(
    request: Request,
    current_user: User,
    subscription: AlertSubscription | None,
    *,
    error: str | None = None,
    success: str | None = None,
    selected_district: str | None = None,
    email_enabled: bool | None = None,
) -> dict:
    district_value = selected_district
    if district_value is None:
        district_value = subscription.district if subscription else current_user.district

    email_value = email_enabled
    if email_value is None:
        email_value = subscription.email_enabled if subscription else True

    return {
        "title": "Flood Alert Subscription",
        "url_for": request.url_for,
        "current_user": current_user,
        "districts": NEPAL_DISTRICTS,
        "subscription": subscription,
        "selected_district": district_value or "",
        "email_enabled": email_value,
        "error": error,
        "success": success,
    }


@router.get("/subscription", response_class=HTMLResponse)
async def subscription_form(request: Request, db: Session = Depends(get_db)):
    current_user = require_subscription_user(request, db)
    if isinstance(current_user, RedirectResponse):
        return current_user

    subscription = get_user_subscription(db, current_user.id)
    success = None
    if request.query_params.get("saved") == "1":
        success = "Subscription saved successfully."
    elif request.query_params.get("unsubscribed") == "1":
        success = "Subscription deactivated successfully."

    return templates.TemplateResponse(
        request=request,
        name="subscription.html",
        context=build_subscription_context(
            request,
            current_user,
            subscription,
            success=success,
        ),
    )


@router.post("/subscription", response_class=HTMLResponse)
async def save_subscription(request: Request, db: Session = Depends(get_db)):
    current_user = require_subscription_user(request, db)
    if isinstance(current_user, RedirectResponse):
        return current_user

    form = await request.form()
    district = form.get("district", "").strip()
    email_enabled = form.get("email_enabled") == "on"
    subscription = get_user_subscription(db, current_user.id)

    if not district:
        error = "District is required."
    elif district not in NEPAL_DISTRICTS:
        error = "Please select a valid district."
    else:
        error = None

    if error:
        return templates.TemplateResponse(
            request=request,
            name="subscription.html",
            context=build_subscription_context(
                request,
                current_user,
                subscription,
                error=error,
                selected_district=district,
                email_enabled=email_enabled,
            ),
            status_code=400,
        )

    if subscription is None:
        subscription = AlertSubscription(
            user_id=current_user.id,
            district=district,
            email_enabled=email_enabled,
            is_active=True,
        )
        db.add(subscription)
    else:
        subscription.district = district
        subscription.email_enabled = email_enabled
        subscription.is_active = True

    db.commit()
    return RedirectResponse(url="/subscription?saved=1", status_code=303)


@router.post("/subscription/unsubscribe")
async def unsubscribe(request: Request, db: Session = Depends(get_db)):
    current_user = require_subscription_user(request, db)
    if isinstance(current_user, RedirectResponse):
        return current_user

    subscription = get_user_subscription(db, current_user.id)
    if subscription is not None:
        subscription.is_active = False
        db.commit()

    return RedirectResponse(url="/subscription?unsubscribed=1", status_code=303)
