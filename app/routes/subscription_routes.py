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
    district_error: str | None = None,
    email_enabled: bool | None = None,
    show_district_form: bool = False,
) -> dict:
    email_value = email_enabled
    if email_value is None:
        email_value = subscription.email_enabled if subscription else True
    alert_district = subscription.district if subscription else current_user.district

    return {
        "title": "Flood Alert Subscription",
        "url_for": request.url_for,
        "current_user": current_user,
        "districts": NEPAL_DISTRICTS,
        "subscription": subscription,
        "account_district": current_user.district or "",
        "alert_district": alert_district or "",
        "email_enabled": email_value,
        "error": error,
        "success": success,
        "district_error": district_error,
        "show_district_form": show_district_form,
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
    elif request.query_params.get("district_updated") == "1":
        success = "Alert district updated successfully."
    elif request.query_params.get("no_subscription") == "1":
        success = "Subscribe first before changing your alert district."
    elif request.query_params.get("predictions_disabled") == "1":
        success = "Flood predictions are managed by CDO users. Subscribe here to receive district alerts."

    show_district_form = request.query_params.get("change_district") == "1"

    return templates.TemplateResponse(
        request=request,
        name="subscription.html",
        context=build_subscription_context(
            request,
            current_user,
            subscription,
            success=success,
            show_district_form=show_district_form,
        ),
    )


@router.post("/subscription", response_class=HTMLResponse)
async def save_subscription(request: Request, db: Session = Depends(get_db)):
    current_user = require_subscription_user(request, db)
    if isinstance(current_user, RedirectResponse):
        return current_user

    form = await request.form()
    email_enabled = form.get("email_enabled") == "on"
    subscription = get_user_subscription(db, current_user.id)
    account_district = (current_user.district or "").strip()

    if not account_district:
        error = "Please set your district before subscribing."
    elif account_district not in NEPAL_DISTRICTS:
        error = "Your account district is invalid."
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
                email_enabled=email_enabled,
            ),
            status_code=400,
        )

    if subscription is None:
        subscription = AlertSubscription(
            user_id=current_user.id,
            district=account_district,
            email_enabled=email_enabled,
            is_active=True,
        )
        db.add(subscription)
    else:
        subscription.email_enabled = email_enabled
        subscription.is_active = True

    db.commit()
    return RedirectResponse(url="/subscription?saved=1", status_code=303)


@router.post("/subscription/district", response_class=HTMLResponse)
async def update_subscription_district(request: Request, db: Session = Depends(get_db)):
    current_user = require_subscription_user(request, db)
    if isinstance(current_user, RedirectResponse):
        return current_user

    form = await request.form()
    district = form.get("district", "").strip()
    subscription = get_user_subscription(db, current_user.id)

    if subscription is None:
        return RedirectResponse(url="/subscription?no_subscription=1", status_code=303)

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
                district_error=error,
                show_district_form=True,
            ),
            status_code=400,
        )

    subscription.district = district

    db.commit()
    return RedirectResponse(url="/subscription?district_updated=1", status_code=303)


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
