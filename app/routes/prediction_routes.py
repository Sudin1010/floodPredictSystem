from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database.connection import get_db

from app.ml import BASE_FEATURES

from app.services import run_flood_prediction, save_prediction_history


router = APIRouter()

# Locate the app directory so FastAPI can find the templates folder.
BASE_DIR = Path(__file__).resolve().parents[1]

templates = Jinja2Templates(directory=BASE_DIR / "templates")


def get_submitted_values(form) -> dict[str, str]:
    """Keep submitted values in the form after prediction."""

    return {
        field: form.get(field, "")
        for field in BASE_FEATURES
    }


@router.get( "/", response_class=HTMLResponse,)
async def home(
    request: Request,
    db: Session = Depends(get_db),
):
    """Display the home page."""

    # Check whether a user is currently logged in.
    current_user = get_current_user(request, db,)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": "Flood Prediction System",
            "url_for": request.url_for,
            "current_user": current_user,
        },
    )


@router.get( "/predict",response_class=HTMLResponse,)
async def predict_form(request: Request,db: Session = Depends(get_db),):
    """Display the flood prediction form."""

    current_user = get_current_user( request,  db,)

    # Only logged-in users can access the prediction page.
    if current_user is None:
        return RedirectResponse(  url="/login?next=predict", status_code=303, )

    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
            "title": "Flood Risk Prediction",
            "url_for": request.url_for,
            "values": {},
            "error": None,
            "current_user": current_user,
        },
    )


@router.post( "/predict",response_class=HTMLResponse,)
async def predict_result( request: Request, db: Session = Depends(get_db),):
    """Process the submitted form and generate an ANN prediction."""

    current_user = get_current_user( request, db,)

    # Prevent unauthenticated users from making predictions.
    if current_user is None:
        return RedirectResponse( url="/login?next=predict",status_code=303,)

    # Read all submitted form data.
    form = await request.form()

    # Keep the submitted values so they remain visible
    # in the form after prediction or validation failure.
    form_values = get_submitted_values( form )

    try:
        prediction_result = run_flood_prediction(form)

        # Save the original raw form inputs and result
        # in the prediction history database.
        save_prediction_history(
            db=db,
            raw_values=prediction_result["raw_values"],
            probability=prediction_result["probability"],
            risk_level=prediction_result["risk_level"],
            user_id=current_user.id,
            prediction_source="personal",
        )

        # Display the prediction result on predict.html.
        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                "title": "Flood Risk Prediction",
                "url_for": request.url_for,

                # Main prediction result
                "prediction": prediction_result["risk_level"],
                "probability": prediction_result["probability"],
                "risk_level": prediction_result["risk_level"],
                "risk_class": prediction_result["risk_class"],
                "risk_explanation": prediction_result["risk_explanation"],
                "recommendation": prediction_result["recommendation"],

                # Binary classification result
                "predicted_class": prediction_result["predicted_class"],
                "classification_label": prediction_result["classification_label"],

                # Keep form and user information
                "values": form_values,
                "error": None,
                "current_user": current_user,
            },
        )

    except Exception as exc:
        # Show validation or prediction errors on the same page.
        # Submitted form values are preserved.
        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                "title": "Flood Risk Prediction",
                "url_for": request.url_for,
                "error": str(exc),
                "values": form_values,
                "current_user": current_user,
            },
        )
    
    
