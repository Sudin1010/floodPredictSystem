from app.ml import (
    BASE_FEATURES,
    apply_log_transformation,
    build_feature_vector,
    predict_class,
    predict_probability,
    scale_features,
    validate_features,
)


def test_value(value: float) -> None:
    raw_values = {
        feature: value
        for feature in BASE_FEATURES
    }

    transformed_values = apply_log_transformation(
        raw_values
    )

    validate_features(transformed_values)

    feature_vector = build_feature_vector(
        transformed_values
    )

    scaled_vector = scale_features(
        feature_vector
    )

    probability = predict_probability(
        scaled_vector
    )

    predicted_class = predict_class(
        probability
    )

    print(
        f"All inputs = {value:>4} | "
        f"Probability = {probability:.8f} | "
        f"Percentage = {probability * 100:.4f}% | "
        f"Class = {predicted_class}"
    )


for value in [0, 2, 5, 8, 12, 16]:
    test_value(value)