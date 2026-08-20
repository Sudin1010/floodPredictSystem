import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.routes import auth_router, cdo_router, dashboard_router, prediction_router

app = FastAPI(title="Flood Prediction System")

# app directory is the parent of this file
BASE_DIR = Path(__file__).resolve().parent
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "change-this-development-secret")

# middleware use secret key to signed session cookies. 
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
app.include_router(cdo_router)
