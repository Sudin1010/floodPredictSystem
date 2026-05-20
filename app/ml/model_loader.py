from pathlib import Path
from typing import Any

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "ann_scratch_model.pkl"


def validate_model_package(model_package: dict[str, Any]) -> None:
    required_keys = {
        "features",
        "feature_means",
        "feature_stds",
        "parameters",
    }
    missing_keys = required_keys - set(model_package.keys())

    if missing_keys:
        raise KeyError(f"Missing model package keys: {missing_keys}")


def load_model_package(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    model_package = joblib.load(model_path)
    validate_model_package(model_package)
    return model_package


model_data = load_model_package()

FEATURES = model_data["features"]
FEATURE_MEANS = model_data["feature_means"]
FEATURE_STDS = model_data["feature_stds"]
PARAMETERS = model_data["parameters"]
PREDICTION_THRESHOLD = model_data.get("prediction_threshold", 0.5)
