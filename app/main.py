import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy import inspect, text
from starlette.middleware.sessions import SessionMiddleware

from app.database.connection import Base, engine
from app.routes import auth_router, dashboard_router, prediction_router

app = FastAPI(title="Flood Prediction System")

BASE_DIR = Path(__file__).resolve().parent
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "change-this-development-secret")

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY,
    same_site="lax",
    https_only=False,
)

# Static files
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

app.include_router(prediction_router)
app.include_router(dashboard_router)
app.include_router(auth_router)


@app.on_event("startup")
def create_database_tables() -> None:
    """Create known ORM tables and preserve the existing history user_id migration."""
    Base.metadata.create_all(bind=engine)
    ensure_prediction_history_user_id()


def ensure_prediction_history_user_id() -> None:
    """Keep older prediction_history tables compatible with per-user history."""
    inspector = inspect(engine)
    if not inspector.has_table("prediction_history"):
        return

    columns = {column["name"] for column in inspector.get_columns("prediction_history")}
    if "user_id" not in columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE prediction_history "
                    "ADD COLUMN user_id INTEGER NULL REFERENCES users(id)"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS "
                    "ix_prediction_history_user_id ON prediction_history (user_id)"
                )
            )
