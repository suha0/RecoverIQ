
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from preprocessing import build_preprocessor


# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "recovery_customers.csv"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)


# Load data
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[
    "customer_id",
    "repaid_next_30_days"
])

y = df["repaid_next_30_days"]


# Split: 60% train, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("Training records:", len(X_train))
print("Validation records:", len(X_validation))
print("Testing records:", len(X_test))


# Models with class weighting
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


best_model = None
best_name = None
best_threshold = 0.5
best_validation_f1 = -1


# Train and select threshold using validation data
for name, algorithm in models.items():

    pipeline = Pipeline([
        ("preprocessing", build_preprocessor()),
        ("model", algorithm)
    ])

    print(f"\nTraining {name}...")

    pipeline.fit(X_train, y_train)

    validation_probabilities = pipeline.predict_proba(
        X_validation
    )[:, 1]

    model_best_f1 = -1
    model_best_threshold = 0.5

    for threshold in [0.20, 0.25, 0.30, 0.35, 0.40,
                      0.45, 0.50, 0.55, 0.60]:

        validation_predictions = (
            validation_probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_validation,
            validation_predictions,
            zero_division=0
        )

        if score > model_best_f1:
            model_best_f1 = score
            model_best_threshold = threshold

    print("Best validation threshold:", model_best_threshold)
    print("Validation F1:", round(model_best_f1, 4))

    if model_best_f1 > best_validation_f1:
        best_validation_f1 = model_best_f1
        best_model = pipeline
        best_name = name
        best_threshold = model_best_threshold


# Final evaluation on untouched test data
test_probabilities = best_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)


print("\n===== FINAL MODEL EVALUATION =====")
print("Selected model:", best_name)
print("Selected threshold:", best_threshold)

print("Accuracy:", round(
    accuracy_score(y_test, test_predictions), 4
))

print("Precision:", round(
    precision_score(y_test, test_predictions, zero_division=0), 4
))

print("Recall:", round(
    recall_score(y_test, test_predictions, zero_division=0), 4
))

print("F1 Score:", round(
    f1_score(y_test, test_predictions, zero_division=0), 4
))

print("ROC AUC:", round(
    roc_auc_score(y_test, test_probabilities), 4
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    test_predictions,
    zero_division=0
))


# Save model and selected threshold
model_artifact = {
    "pipeline": best_model,
    "threshold": best_threshold,
    "model_name": best_name
}

model_path = MODEL_DIR / "recovery_model.joblib"

joblib.dump(model_artifact, model_path)

print("\nSaved improved model:", model_path)
