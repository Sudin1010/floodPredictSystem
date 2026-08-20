from app.routes.auth_routes import router as auth_router
from app.routes.cdo_routes import router as cdo_router
from app.routes.dashboard_routes import router as dashboard_router
from app.routes.prediction_routes import router as prediction_router
from app.routes.subscription_routes import router as subscription_router

__all__ = [
    "auth_router",
    "cdo_router",
    "dashboard_router",
    "prediction_router",
    "subscription_router",
]
