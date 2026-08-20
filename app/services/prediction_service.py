from app.ml import (
    apply_log_transformation,
    build_feature_vector,
    map_risk_level,
    predict_class,
    predict_probability,
    scale_features,
    validate_features,
    validate_raw_inputs,
)


def run_flood_prediction(form) -> dict:
    raw_values = validate_raw_inputs(form)
    prediction_values = apply_log_transformation(raw_values)
    validate_features(prediction_values)
    feature_vector = build_feature_vector(prediction_values)
    scaled_vector = scale_features(feature_vector)
    probability = predict_probability(scaled_vector)
    predicted_class = predict_class(probability)
    probability_percent = round(probability * 100, 2)
    risk_level, risk_class, risk_explanation, recommendation = map_risk_level(
        probability_percent
    )
    classification_label = (
        "Higher Flood Risk" if predicted_class == 1 else "Lower Flood Risk"
    )

    return {
        "raw_values": raw_values,
        "probability": probability_percent,
        "risk_level": risk_level,
        "risk_class": risk_class,
        "risk_explanation": risk_explanation,
        "recommendation": recommendation,
        "predicted_class": predicted_class,
        "classification_label": classification_label,
    }
