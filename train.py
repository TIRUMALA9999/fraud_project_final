"""Train, tune, compare and select the best fraud detection model."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# 1. CONFIGURATION
# -----------------------------

DATA_PATH = "data/transactions.csv"
MODEL_PATH = "models/best_model.joblib"

FEATURES = [
    "transaction_type",
    "amount_usd",
    "sender_balance_before_usd",
    "prior_txn_count_1h",
    "is_new_receiver",
    "account_age_days",
    "is_new_device",
    "is_vpn_or_proxy",
    "country_mismatch",
    "geo_velocity_kmh",
]

NUMERIC_FEATURES = [
    feature for feature in FEATURES
    if feature != "transaction_type"
]

MAX_FALSE_POSITIVE_RATE = 0.01
TUNING_TRIALS = 8


# -----------------------------
# 2. PREPROCESSING
# -----------------------------

def create_pipeline(model, scale_numeric=False):

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["transaction_type"],
            ),
            (
                "numeric",
                StandardScaler() if scale_numeric else "passthrough",
                NUMERIC_FEATURES,
            ),
        ],
        sparse_threshold=0,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", model),
    ])


# -----------------------------
# 3. DECISION THRESHOLD
# -----------------------------

def choose_threshold(y_true, scores):
    """
    Maximize validation F1 while keeping the
    false-positive rate at or below 1%.
    """

    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
        drop_intermediate=False,
    )

    eligible = np.flatnonzero(
        (fpr <= MAX_FALSE_POSITIVE_RATE)
        & np.isfinite(thresholds)
    )

    if len(eligible) == 0:
        return float(
            np.nextafter(np.max(scores), np.inf)
        )

    positives = labels.sum()
    negatives = len(labels) - positives

    tp = tpr[eligible] * positives
    fp = fpr[eligible] * negatives
    fn = positives - tp

    f1_values = (
        2 * tp / np.maximum(2 * tp + fp + fn, 1)
    )

    best_position = max(
        range(len(eligible)),
        key=lambda i: (
            f1_values[i],
            tpr[eligible[i]],
            -fpr[eligible[i]],
        ),
    )

    return float(thresholds[eligible[best_position]])


# -----------------------------
# 4. MODEL EVALUATION
# -----------------------------

def evaluate(y_true, scores, threshold):

    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "ROC_AUC": float(roc_auc_score(y_true, scores)),
        "PR_AUC": float(average_precision_score(y_true, scores)),
        "Precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "F1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "FPR": float(fp / (fp + tn)),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


# -----------------------------
# 5. DEFINE THREE ALGORITHMS
# -----------------------------

def get_models():

    return {
        "Logistic Regression": {
            "model": LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                class_weight="balanced",
                random_state=42,
            ),
            "scale": True,
            "params": {
                "model__C": [0.01, 0.1, 1, 10],
                "model__penalty": ["l1", "l2"],
                "model__class_weight": [None, "balanced"],
            },
        },

        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=120,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=1,
            ),
            "scale": False,
            "params": {
                "model__n_estimators": [100, 150, 200],
                "model__max_depth": [8, 12, 16],
                "model__min_samples_leaf": [1, 5, 10],
                "model__class_weight": [
                    None, "balanced_subsample"
                ],
            },
        },

        "LightGBM": {
            "model": LGBMClassifier(
                n_estimators=180,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=80,
                random_state=42,
                n_jobs=1,
                verbosity=-1,
            ),
            "scale": False,
            "params": {
                "model__n_estimators": [100, 180, 300],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__num_leaves": [7, 15, 31],
                "model__min_child_samples": [20, 50, 80],
            },
        },
    }


# -----------------------------
# 6. TRAIN ALL MODELS
# -----------------------------

def main():

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    df = df.sort_values(
        "event_timestamp_utc",
        kind="stable",
    )

    if df[FEATURES + ["is_fraud"]].isna().any().any():
        raise ValueError("Missing values in training data.")

    # Chronological 70% / 15% / 15% split.
    train_end = int(len(df) * 0.70)
    valid_end = int(len(df) * 0.85)

    train = df.iloc[:train_end]
    valid = df.iloc[train_end:valid_end]
    test = df.iloc[valid_end:]

    for name, split in [
        ("Train", train),
        ("Validation", valid),
        ("Test", test),
    ]:
        if split["is_fraud"].nunique() != 2:
            raise ValueError(
                f"{name} split must contain both classes."
            )

    X_train = train[FEATURES]
    y_train = train["is_fraud"]

    X_valid = valid[FEATURES]
    y_valid = valid["is_fraud"]

    X_test = test[FEATURES]
    y_test = test["is_fraud"]

    print(f"Dataset: {len(df):,} rows")
    print(f"Training: {len(train):,}")
    print(f"Validation: {len(valid):,}")
    print(f"Testing: {len(test):,}")

    results = []

    for model_name, config in get_models().items():

        print(f"\n========== {model_name} ==========")

        pipeline = create_pipeline(
            config["model"],
            config["scale"],
        )

        # Experiment 1: Baseline
        print("\nTraining baseline...")

        pipeline.fit(X_train, y_train)

        baseline_scores = pipeline.predict_proba(
            X_valid
        )[:, 1]

        baseline_threshold = choose_threshold(
            y_valid, baseline_scores
        )

        baseline_metrics = evaluate(
            y_valid,
            baseline_scores,
            baseline_threshold,
        )

        results.append({
            "Name": f"{model_name} - Baseline",
            "Pipeline": pipeline,
            "Threshold": baseline_threshold,
            **baseline_metrics,
        })

        print(
            f"Baseline F1: {baseline_metrics['F1']:.4f}"
        )

        # Experiment 2: Hyperparameter tuning
        print("\nPerforming hyperparameter tuning...")

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=config["params"],
            n_iter=TUNING_TRIALS,
            scoring="average_precision",
            cv=TimeSeriesSplit(n_splits=3),
            n_jobs=2,
            random_state=42,
            refit=True,
            error_score="raise",
        )

        search.fit(X_train, y_train)

        print("Best hyperparameters:")
        print(search.best_params_)

        print(
            f"Best CV PR-AUC: {search.best_score_:.4f}"
        )

        tuned_model = search.best_estimator_

        tuned_scores = tuned_model.predict_proba(
            X_valid
        )[:, 1]

        tuned_threshold = choose_threshold(
            y_valid, tuned_scores
        )

        tuned_metrics = evaluate(
            y_valid,
            tuned_scores,
            tuned_threshold,
        )

        results.append({
            "Name": f"{model_name} - Tuned",
            "Pipeline": tuned_model,
            "Threshold": tuned_threshold,
            **tuned_metrics,
        })

        print(
            f"Tuned F1: {tuned_metrics['F1']:.4f}"
        )

    # -----------------------------
    # 7. COMPARE ALL SIX EXPERIMENTS
    # -----------------------------

    comparison = pd.DataFrame([
        {
            key: value
            for key, value in result.items()
            if key not in ["Pipeline", "Threshold"]
        }
        for result in results
    ])

    comparison = comparison.sort_values(
        "F1",
        ascending=False,
    )

    print("\n========== MODEL COMPARISON ==========")

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # -----------------------------
    # 8. SELECT BEST MODEL
    # -----------------------------

    winner = max(
        results,
        key=lambda result: (
            result["F1"],
            result["Recall"],
            result["PR_AUC"],
        ),
    )

    print("\n========== WINNING MODEL ==========")

    print("Model:", winner["Name"])
    print(f"Validation F1: {winner['F1']:.4f}")
    print(f"Threshold: {winner['Threshold']:.5f}")

    # -----------------------------
    # 9. FINAL TEST EVALUATION
    # -----------------------------

    # Test data is used only after model selection.
    test_scores = winner["Pipeline"].predict_proba(
        X_test
    )[:, 1]

    test_metrics = evaluate(
        y_test,
        test_scores,
        winner["Threshold"],
    )

    print("\n========== FINAL TEST RESULTS ==========")

    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    # -----------------------------
    # 10. SAVE WINNING MODEL
    # -----------------------------

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": winner["Pipeline"],
            "threshold": winner["Threshold"],
            "features": FEATURES,
            "model_name": winner["Name"],
        },
        MODEL_PATH,
    )

    comparison.to_csv(
        "reports/model_comparison.csv",
        index=False,
    )

    with open(
        "reports/test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "selected_model": winner["Name"],
                "threshold": winner["Threshold"],
                "test_metrics": test_metrics,
            },
            file,
            indent=2,
        )

    print("\nTraining completed successfully!")
    print("Best model saved:", MODEL_PATH)


if __name__ == "__main__":
    main()