from typing import Optional

from fastapi import HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.auth.hashing import verify_password
from app.database.models import User


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Fetch a user by primary key for session lookups."""
    return db.get(User, user_id)


def get_user_by_username_or_email(db: Session, value: str) -> Optional[User]:
    """Fetch a user by username or email, preserving login flexibility."""
    return db.scalar(
        select(User).where(or_(User.username == value, User.email == value))
    )


def authenticate_user(db: Session, username_or_email: str, password: str) -> Optional[User]:
    """Authenticate a user with username-or-email plus plaintext password."""
    user = get_user_by_username_or_email(db, username_or_email)
    if not user or not verify_password(password, user.password_hash):
        return None
    return user


def get_current_user(request: Request, db: Session) -> Optional[User]:
    """Return the currently logged-in user from request.session['user_id']."""
    user_id = request.session.get("user_id")
    if user_id is None:
        return None
    return get_user_by_id(db, int(user_id))


def require_cdo(request: Request, db: Session) -> User | RedirectResponse:
    """Require a logged-in CDO user for protected CDO pages."""
    current_user = get_current_user(request, db)
    if current_user is None:
        return RedirectResponse(url="/login", status_code=303)

    if current_user.role != "cdo":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CDO access required.",
        )

    return current_user
