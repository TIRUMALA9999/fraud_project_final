"""Explain the selected FraudGuard model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import shap

from sklearn.inspection import permutation_importance


# Load the selected model.
saved = joblib.load("models/best_model.joblib")

pipeline = saved["model"]
features = saved["features"]
threshold = saved["threshold"]

print("Selected model:", saved["model_name"])

# Load the untouched test portion.
df = pd.read_csv("data/transactions.csv")

df = df.sort_values(
    "event_timestamp_utc",
    kind="stable",
)

test_start = int(len(df) * 0.85)

test = df.iloc[test_start:]

X_test = test[features]
y_test = test["is_fraud"]

Path("reports").mkdir(exist_ok=True)


# --------------------------------------
# 1. GLOBAL FEATURE IMPORTANCE
# --------------------------------------

print("\nCalculating feature importance...")

importance = permutation_importance(
    pipeline,
    X_test,
    y_test,
    scoring="average_precision",
    n_repeats=3,
    random_state=42,
    n_jobs=1,
    max_samples=5000,
)

importance_df = pd.DataFrame({
    "feature": features,
    "importance": importance.importances_mean,
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False,
)

print("\nFeature importance:")
print(importance_df.to_string(index=False))

plt.figure(figsize=(9, 5))

plt.barh(
    importance_df["feature"],
    importance_df["importance"],
    color="steelblue",
)

plt.gca().invert_yaxis()

plt.title("Global Feature Importance")
plt.xlabel("Decrease in Average Precision")

plt.tight_layout()

plt.savefig(
    "reports/feature_importance.png",
    dpi=150,
)

plt.close()


# --------------------------------------
# 2. SHAP EXPLANATIONS
# --------------------------------------

print("\nCalculating SHAP values...")

# Select 50 test transactions to keep it quick.
X_sample = X_test.sample(
    n=50,
    random_state=42,
)

# Use the same preprocessing used during training.
preprocessor = pipeline.named_steps["preprocessing"]

tree_model = pipeline.named_steps["model"]

processed = preprocessor.transform(X_sample)

if hasattr(processed, "toarray"):
    processed = processed.toarray()

processed_features = preprocessor.get_feature_names_out()

X_processed = pd.DataFrame(
    processed,
    columns=processed_features,
)

# Our winning algorithm is Random Forest.
explainer = shap.TreeExplainer(tree_model)

shap_values = explainer(X_processed)

# For Random Forest binary classification,
# select SHAP values for the fraud class (class 1).
if shap_values.values.ndim == 3:
    shap_values = shap_values[:, :, 1]


# --------------------------------------
# 3. SHAP SUMMARY PLOT
# --------------------------------------

shap.plots.beeswarm(
    shap_values,
    max_display=10,
    show=False,
)

plt.tight_layout()

plt.savefig(
    "reports/shap_summary.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------
# 4. EXPLAIN ONE HIGH-RISK PAYMENT
# --------------------------------------

risk_scores = pipeline.predict_proba(
    X_sample
)[:, 1]

highest_risk_index = int(
    np.argmax(risk_scores)
)

risk_score = float(
    risk_scores[highest_risk_index]
)

decision = (
    "REJECTED"
    if risk_score >= threshold
    else "APPROVED"
)

print("\nHigh-risk transaction example:")
print(X_sample.iloc[highest_risk_index])

print(f"\nFraud risk score: {risk_score:.4f}")
print(f"Decision threshold: {threshold:.4f}")
print("Decision:", decision)

shap.plots.waterfall(
    shap_values[highest_risk_index],
    max_display=10,
    show=False,
)

plt.tight_layout()

plt.savefig(
    "reports/shap_waterfall.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print("\nExplainability completed!")
print("reports/feature_importance.png")
print("reports/shap_summary.png")
print("reports/shap_waterfall.png")