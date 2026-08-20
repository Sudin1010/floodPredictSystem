import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "models" / "ann_scratch_model.pkl"

model_package = joblib.load(MODEL_PATH)

print("\nMODEL PACKAGE KEYS:")
print(model_package.keys())

print("\nFEATURE COUNT:")
print(model_package.get("feature_count"))

print("\nFEATURE NAMES:")
features = model_package.get("features")

if features:
    for i, feature in enumerate(features, start=1):
        print(f"{i}. {feature}")
else:
    print("No features found.")

print("\nFEATURE MEANS:")
print(model_package.get("feature_means"))

print("\nFEATURE STDS:")
print(model_package.get("feature_stds"))
