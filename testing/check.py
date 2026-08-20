from app.ml.model_loader import (
    FEATURES,
    FEATURE_MEANS,
    FEATURE_STDS,
    PARAMETERS,
    PREDICTION_THRESHOLD,
)


print("\n--- MODEL INFORMATION ---")

print("Number of saved features:", len(FEATURES))
print("Features:", FEATURES)

print("\n--- SCALING INFORMATION ---")

print("Number of feature means:", len(FEATURE_MEANS))
print("Number of feature standard deviations:", len(FEATURE_STDS))

print("\n--- ANN PARAMETER SHAPES ---")

print("W1 shape:", PARAMETERS["W1"].shape)
print("b1 shape:", PARAMETERS["b1"].shape)

print("W2 shape:", PARAMETERS["W2"].shape)
print("b2 shape:", PARAMETERS["b2"].shape)

print("W3 shape:", PARAMETERS["W3"].shape)
print("b3 shape:", PARAMETERS["b3"].shape)

print("\nPrediction threshold:", PREDICTION_THRESHOLD)

print("\n--- CONSISTENCY CHECK ---")

number_of_features = len(FEATURES)
number_of_ann_inputs = PARAMETERS["W1"].shape[0]

if number_of_features == number_of_ann_inputs:
    print(
        "Correct: saved features and ANN input size match:",
        number_of_features,
    )
else:
    print(
        "ERROR: feature count does not match W1 input size."
    )
    print("Saved features:", number_of_features)
    print("W1 input size:", number_of_ann_inputs)