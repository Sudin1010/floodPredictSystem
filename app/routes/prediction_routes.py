from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database.connection import get_db

from app.ml import (
    BASE_FEATURES,
    apply_log_transformation,
    build_feature_vector,
    map_risk_level,
    predict_class,
    predict_probability,
    scale_features,
    validate_features,
    validate_raw_inputs,
)

from app.services import save_prediction_history


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
        # Step 1:
        # Collect and validate the 18 original form inputs.
        raw_values = validate_raw_inputs(form)

        # Step 2:
        # Apply the same log1p transformation that was
        # used before training the ANN.
        prediction_values = apply_log_transformation(raw_values)

        # Step 3:
        # Confirm that the runtime inputs exactly match
        # the 18 features stored in the model package.
        validate_features( prediction_values)

        # Step 4:
        # Arrange all values in the exact feature order
        # used during ANN training.
        feature_vector = build_feature_vector( prediction_values)

        # Step 5:
        # Standardize values using the training means
        # and standard deviations saved in the model.
        scaled_vector = scale_features(feature_vector)

        # Step 6:
        # Run ANN forward propagation and return a
        # probability between 0 and 1.
        probability = predict_probability(scaled_vector)

        # Step 7:
        # Convert the probability into class 0 or 1
        # using the threshold stored in the model package.
        # The current model threshold is approximately 0.40.
        predicted_class = predict_class(probability)

        # Give the binary class a readable website label.
        if predicted_class == 1:
            classification_label = (
                "Higher Flood Risk"
            )
        else:
            classification_label = (
                "Lower Flood Risk"
            )

        # Convert probability from decimal form to percentage.
        # Example: 0.65 becomes 65.00%.
        probability_percent = round(
            probability * 100,
            2,
        )

        # Convert the probability percentage into
        # Low, Medium or High risk for presentation.
        (
            risk_level,
            risk_class,
            risk_explanation,
            recommendation,
        ) = map_risk_level(
            probability_percent
        )

        # Save the original raw form inputs and result
        # in the prediction history database.
        save_prediction_history(
            db=db,
            raw_values=raw_values,
            probability=probability_percent,
            risk_level=risk_level,
            user_id=current_user.id,
        )

        # Display the prediction result on predict.html.
        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                "title": "Flood Risk Prediction",
                "url_for": request.url_for,

                # Main prediction result
                "prediction": risk_level,
                "probability": probability_percent,
                "risk_level": risk_level,
                "risk_class": risk_class,
                "risk_explanation": risk_explanation,
                "recommendation": recommendation,

                # Binary classification result
                "predicted_class": predicted_class,
                "classification_label": (
                    classification_label
                ),

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