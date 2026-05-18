import joblib

MODEL_PATH = "models/ann_scratch_model.pkl"

model_package = joblib.load(MODEL_PATH)

print("\nMODEL PACKAGE KEYS:")
print(model_package.keys())

print("\nFEATURE COUNT:")
print(model_package.get("feature_count"))

print("\nFEATURE NAMES:")
feature_names = model_package.get("feature_names")

if feature_names:
    for i, feature in enumerate(feature_names, start=1):
        print(f"{i}. {feature}")
else:
    print("No feature_names found.")

print("\nFEATURE MEANS:")
print(model_package.get("feature_means"))

print("\nFEATURE STDS:")
print(model_package.get("feature_stds"))

