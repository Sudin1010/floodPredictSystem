from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.auth import authenticate_user, get_current_user, get_user_by_username_or_email, hash_password
from app.database import get_db
from app.models import User

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@router.get("/register", response_class=HTMLResponse)
async def register_form(request: Request, db: Session = Depends(get_db)):
    current_user = get_current_user(request, db)
    if current_user is not None:
        return RedirectResponse(url="/predict", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="register.html",
        context={
            "title": "Register",
            "url_for": request.url_for,
            "current_user": current_user,
            "error": None,
        },
    )


@router.post("/register", response_class=HTMLResponse)
async def register_user(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    username = form.get("username", "").strip()
    email = form.get("email", "").strip().lower()
    password = form.get("password", "")
    confirm_password = form.get("confirm_password", "")

    error = None
    if not username or not email or not password or not confirm_password:
        error = "All fields are required."
    elif len(password) < 8:
        error = "Password must be at least 8 characters."
    elif password != confirm_password:
        error = "Passwords do not match."
    elif get_user_by_username_or_email(db, username) is not None:
        error = "Username is already registered."
    elif get_user_by_username_or_email(db, email) is not None:
        error = "Email is already registered."

    if error:
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "title": "Register",
                "url_for": request.url_for,
                "current_user": None,
                "error": error,
                "username": username,
                "email": email,
            },
        )

    user = User(
        username=username,
        email=email,
        password_hash=hash_password(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return RedirectResponse(url="/login?registered=1", status_code=303)


@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request, db: Session = Depends(get_db)):
    current_user = get_current_user(request, db)
    if current_user is not None:
        return RedirectResponse(url="/predict", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "title": "Login",
            "url_for": request.url_for,
            "current_user": current_user,
            "error": None,
            "success": "Registration successful. Please login."
            if request.query_params.get("registered") == "1"
            else None,
        },
    )


@router.post("/login", response_class=HTMLResponse)
async def login_user(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    username_or_email = form.get("username_or_email", "").strip()
    password = form.get("password", "")
    user = authenticate_user(db, username_or_email, password)

    if user is None:
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "title": "Login",
                "url_for": request.url_for,
                "current_user": None,
                "error": "Invalid username/email or password.",
                "username_or_email": username_or_email,
            },
        )

    request.session["user_id"] = user.id
    return RedirectResponse(url="/predict", status_code=303)


@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)
