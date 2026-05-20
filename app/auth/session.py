from typing import Optional

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.auth.hashing import verify_password
from app.database.models import User


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.get(User, user_id)


def get_user_by_username_or_email(db: Session, value: str) -> Optional[User]:
    return db.scalar(
        select(User).where(or_(User.username == value, User.email == value))
    )


def authenticate_user(db: Session, username_or_email: str, password: str) -> Optional[User]:
    user = get_user_by_username_or_email(db, username_or_email)
    if not user or not verify_password(password, user.password_hash):
        return None
    return user


def get_current_user(request, db: Session) -> Optional[User]:
    user_id = request.session.get("user_id")
    if user_id is None:
        return None
    return get_user_by_id(db, int(user_id))
