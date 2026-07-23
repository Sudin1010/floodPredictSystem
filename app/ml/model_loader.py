from pathlib import Path
from typing import Any

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "ann_scratch_model.pkl"


def validate_model_package(model_package: dict[str, Any]) -> None:
    """Ensure the saved ANN package contains the data required for inference."""

    required_keys = {
        "features",
        "feature_means",
        "feature_stds",
        "parameters",

        # Important:
        # This is the threshold selected using validation data.
        "prediction_threshold",
    }

    missing_keys = required_keys - set(model_package.keys())

    if missing_keys:
        raise KeyError(
            f"Missing model package keys: {missing_keys}"
        )


def load_model_package(
    model_path: Path = MODEL_PATH
) -> dict[str, Any]:
    """Load the saved ANN package without retraining or modifying it."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"ANN model file was not found: {model_path}"
        )

    model_package = joblib.load(model_path)

    if not isinstance(model_package, dict):
        raise TypeError(
            "The saved ANN model package must be a dictionary."
        )

    validate_model_package(model_package)

    return model_package


# Load the saved ANN package once when the application starts
model_data = load_model_package()

# Feature information saved during model training
FEATURES = model_data["features"]
FEATURE_MEANS = model_data["feature_means"]
FEATURE_STDS = model_data["feature_stds"]

# Trained ANN weights and biases
PARAMETERS = model_data["parameters"]

# Important:
# This should now load the validation-selected threshold,
# such as 0.47, from the new model package.
PREDICTION_THRESHOLD = float(
    model_data["prediction_threshold"]
)