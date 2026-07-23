import numpy as np

from app.ml.model_loader import (
    PARAMETERS,
    PREDICTION_THRESHOLD,
)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Apply a numerically stable sigmoid activation."""

    # Prevent overflow when calculating exp(-z)
    z = np.clip(z, -500, 500)

    return 1 / (1 + np.exp(-z))


def relu(z: np.ndarray) -> np.ndarray:
    """Apply the hidden-layer ReLU activation."""

    return np.maximum(0, z)


def forward_propagation(
    X_data: np.ndarray
) -> np.ndarray:
    """
    Run the saved three-layer ANN forward pass
    without changing the trained weights.
    """

    # First hidden layer
    Z1 = (
        X_data @ PARAMETERS["W1"]
        + PARAMETERS["b1"]
    )
    A1 = relu(Z1)

    # Second hidden layer
    Z2 = (
        A1 @ PARAMETERS["W2"]
        + PARAMETERS["b2"]
    )
    A2 = relu(Z2)

    # Output layer
    Z3 = (
        A2 @ PARAMETERS["W3"]
        + PARAMETERS["b3"]
    )
    A3 = sigmoid(Z3)

    return A3


def predict_probability(
    scaled_vector: np.ndarray
) -> float:
    """
    Return the ANN output probability for an
    already-scaled feature vector.
    """

    probability = forward_propagation(
        scaled_vector
    ).ravel()[0]

    return float(probability)


def predict_class(
    probability: float
) -> int:
    """
    Convert ANN probability into binary class
    using the threshold selected on validation data.

    Class 0 = Lower Flood Risk
    Class 1 = Higher Flood Risk
    """

    # Important:
    # This uses 0.47 or whichever threshold is
    # stored in the latest ANN model package.
    return int(
        probability >= PREDICTION_THRESHOLD
    )


def map_risk_level(
    probability_percent: float
) -> tuple[str, str, str, str]:
    """
    Map the ANN probability percentage to the
    existing Low/Medium/High website presentation.
    """

    if probability_percent < 40:
        risk_level = "Low Risk"
        risk_class = "low"

        risk_explanation = (
            "Current conditions indicate a low "
            "possibility of flood risk."
        )

        recommendation = (
            "Continue regular monitoring."
        )

    elif probability_percent < 70:
        risk_level = "Medium Risk"
        risk_class = "medium"

        risk_explanation = (
            "Moderate flood risk detected."
        )

        recommendation = (
            "Increased monitoring and preparedness "
            "are recommended."
        )

    else:
        risk_level = "High Risk"
        risk_class = "high"

        risk_explanation = (
            "High flood risk detected."
        )

        recommendation = (
            "Immediate preparedness and safety "
            "measures are recommended."
        )

    return (
        risk_level,
        risk_class,
        risk_explanation,
        recommendation,
    )