import numpy as np

from app.ml.model_loader import (
    FEATURES,
    FEATURE_MEANS,
    FEATURE_STDS,
)


# The 18 raw features shown in the browser form
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
    """Validate the 18 browser inputs before preprocessing."""

    values: dict[str, float] = {}

    for field in BASE_FEATURES:
        raw_value = form.get(field)

        if raw_value is None or raw_value == "":
            raise ValueError(f"{field} is required.")

        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field} must be a numeric value."
            ) from exc

        if not np.isfinite(value):
            raise ValueError(
                f"{field} must be a finite number."
            )

        if value < 0:
            raise ValueError(
                f"{field} must be greater than or equal to 0."
            )

        values[field] = value

    return values


def apply_log_transformation(values: dict[str, float]) -> dict[str, float]:
    """
    Apply the same log1p transformation used
    when training the ANN.
    """

    transformed_values = values.copy()

    for field in BASE_FEATURES:
        transformed_values[field] = float(
            np.log1p(transformed_values[field])
        )

    return transformed_values


def validate_features(
    values: dict[str, float]
) -> None:
    """
    Confirm that runtime inputs exactly match
    the 18 features expected by the saved model.
    """

    expected_features = set(FEATURES)
    received_features = set(values.keys())

    missing = expected_features - received_features
    extra = received_features - expected_features

    if missing:
        raise ValueError(
            f"Missing model features: {sorted(missing)}"
        )

    if extra:
        raise ValueError(
            f"Unexpected model features: {sorted(extra)}"
        )


def build_feature_vector(
    values: dict[str, float]
) -> np.ndarray:
    """
    Arrange the 18 features in the exact order
    used during model training.
    """

    return np.array(
        [[values[name] for name in FEATURES]],
        dtype=float,
    )


def scale_features(
    feature_vector: np.ndarray
) -> np.ndarray:
    """
    Standardize inputs using the training means
    and standard deviations saved in the model.
    """

    means = np.array(
        [FEATURE_MEANS[name] for name in FEATURES],
        dtype=float,
    )

    stds = np.array(
        [FEATURE_STDS[name] for name in FEATURES],
        dtype=float,
    )

    # Prevent division by zero
    stds = np.where(stds == 0, 1.0, stds)

    return (feature_vector - means) / stds