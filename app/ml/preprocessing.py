import numpy as np

from app.ml.model_loader import FEATURES, FEATURE_MEANS, FEATURE_STDS

# Raw features shown in browser form
BASE_FEATURES = [
    "MonsoonIntensity",
    "TopographyDrainage",
    "RiverManagement",
    "Deforestation",
    "Urbanization",
    "ClimateChange",
    "DamsQuality",
    "Siltation",
    "AgriculturalPractices",
    "Encroachments",
    "IneffectiveDisasterPreparedness",
    "DrainageSystems",
    "Landslides",
    "Watersheds",
    "DeterioratingInfrastructure",
    "PopulationScore",
    "WetlandLoss",
    "InadequatePlanning",
]


def validate_raw_inputs(form) -> dict[str, float]:
    """Validate the 18 browser inputs before any training-time transformations."""
    values: dict[str, float] = {}

    for field in BASE_FEATURES:
        raw_value = form.get(field)

        if raw_value is None or raw_value == "":
            raise ValueError(f"{field} is required.")

        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a numeric value.") from exc

        if not np.isfinite(value):
            raise ValueError(f"{field} must be a finite number.")

        if value < 0:
            raise ValueError(f"{field} must be greater than or equal to 0.")

        values[field] = value

    return values


def apply_log_transformation(values: dict[str, float]) -> dict[str, float]:
    """Apply the training pipeline's log1p transformation to raw features."""
    for field in BASE_FEATURES:
        values[field] = float(np.log1p(values[field]))
    return values


def compute_derived_features(values: dict[str, float]) -> dict[str, float]:
    """Create the 4 engineered features that expand 18 raw inputs to 22 ANN inputs."""
    values["RainFactor"] = values["MonsoonIntensity"] * values["ClimateChange"]

    values["LandRisk"] = (
        values["Deforestation"]
        + values["Urbanization"]
        + values["Encroachments"]
    ) / 3

    values["WaterStress"] = (
        values["RiverManagement"]
        + values["DrainageSystems"]
        + values["DamsQuality"]
    ) / 3

    values["Blockage"] = (
        values["Siltation"]
        + values["Landslides"]
    ) / 2

    return values


def validate_features(values: dict[str, float]) -> None:
    """Check that runtime features exactly match the saved model feature set."""
    missing = set(FEATURES) - set(values.keys())
    extra = set(values.keys()) - set(FEATURES)

    if missing:
        raise ValueError(f"Missing model features: {missing}")

    if extra:
        raise ValueError(f"Unexpected model features: {extra}")


def build_feature_vector(values: dict[str, float]) -> np.ndarray:
    """Build the model input vector in the saved training feature order."""
    return np.array([[values[name] for name in FEATURES]], dtype=float)


def scale_features(feature_vector: np.ndarray) -> np.ndarray:
    """Standardize features using means and standard deviations saved with the model."""
    means = np.array([FEATURE_MEANS[name] for name in FEATURES], dtype=float)
    stds = np.array([FEATURE_STDS[name] for name in FEATURES], dtype=float)

    stds[stds == 0] = 1.0

    return (feature_vector - means) / stds
