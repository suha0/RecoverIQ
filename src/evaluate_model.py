
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "recovery_customers.csv"
MODEL_PATH = ROOT / "models" / "recovery_model.joblib"
REPORT_PATH = ROOT / "reports"

REPORT_PATH.mkdir(exist_ok=True)


# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------

data = pd.read_csv(DATA_PATH)
artifact = joblib.load(MODEL_PATH)

model = artifact["pipeline"]
threshold = artifact["threshold"]

print("Model:", artifact["model_name"])
print("Threshold:", threshold)


# --------------------------------------------------
# RECREATE EXACT TRAIN / VALIDATION / TEST SPLIT
# --------------------------------------------------

X = data.drop(
    columns=["customer_id", "repaid_next_30_days"]
)

y = data["repaid_next_30_days"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

y_probability = model.predict_proba(X_test)[:, 1]

y_prediction = (
    y_probability >= threshold
).astype(int)


# --------------------------------------------------
# METRICS
# --------------------------------------------------

metrics = {
    "Accuracy": accuracy_score(y_test, y_prediction),
    "Precision": precision_score(
        y_test, y_prediction, zero_division=0
    ),
    "Recall": recall_score(
        y_test, y_prediction, zero_division=0
    ),
    "F1 Score": f1_score(
        y_test, y_prediction, zero_division=0
    ),
    "ROC AUC": roc_auc_score(y_test, y_probability),
    "Average Precision": average_precision_score(
        y_test, y_probability
    )
}

print("\n========== MODEL EVALUATION ==========")

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

print("\n========== CLASSIFICATION REPORT ==========")

report = classification_report(
    y_test,
    y_prediction,
    target_names=["Not Repaid", "Repaid"],
    zero_division=0
)

print(report)

# Save metrics and report
pd.DataFrame(
    metrics.items(),
    columns=["Metric", "Score"]
).to_csv(REPORT_PATH / "model_metrics.csv", index=False)

with open(
    REPORT_PATH / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_prediction,
    display_labels=["Not Repaid", "Repaid"],
    cmap="Blues",
    ax=ax
)

ax.set_title("RecoverIQ - Confusion Matrix")

plt.tight_layout()
plt.savefig(
    REPORT_PATH / "confusion_matrix.png",
    dpi=300
)
plt.close()


# --------------------------------------------------
# ROC CURVE
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

RocCurveDisplay.from_predictions(
    y_test,
    y_probability,
    ax=ax
)

ax.set_title("RecoverIQ - ROC Curve")

plt.tight_layout()
plt.savefig(
    REPORT_PATH / "roc_curve.png",
    dpi=300
)
plt.close()


# --------------------------------------------------
# PRECISION-RECALL CURVE
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

PrecisionRecallDisplay.from_predictions(
    y_test,
    y_probability,
    ax=ax
)

ax.set_title("RecoverIQ - Precision-Recall Curve")

plt.tight_layout()
plt.savefig(
    REPORT_PATH / "precision_recall_curve.png",
    dpi=300
)
plt.close()


# --------------------------------------------------
# LOGISTIC REGRESSION COEFFICIENTS
# --------------------------------------------------

classifier = None
preprocessor = None

if hasattr(model, "named_steps"):

    for step in model.named_steps.values():

        if hasattr(step, "coef_"):
            classifier = step

        if hasattr(step, "get_feature_names_out"):
            preprocessor = step

if classifier is not None and preprocessor is not None:

    feature_names = preprocessor.get_feature_names_out()

    coefficients = classifier.coef_[0]

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
        "Absolute Importance": np.abs(coefficients)
    })

    feature_importance = feature_importance.sort_values(
        "Absolute Importance",
        ascending=False
    )

    feature_importance.to_csv(
        REPORT_PATH / "feature_coefficients.csv",
        index=False
    )

    top_features = feature_importance.head(15).sort_values(
        "Absolute Importance"
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        top_features["Feature"],
        top_features["Coefficient"]
    )

    ax.set_title("RecoverIQ - Top Logistic Regression Coefficients")
    ax.set_xlabel("Coefficient")

    plt.tight_layout()
    plt.savefig(
        REPORT_PATH / "feature_coefficients.png",
        dpi=300
    )
    plt.close()

    print("\nFeature coefficient analysis saved.")

else:
    print(
        "\nCoefficient analysis skipped: "
        "could not identify the classifier and preprocessor."
    )


# --------------------------------------------------
# COMPLETION
# --------------------------------------------------

print("\n========== EVALUATION COMPLETE ==========")

print("Test records:", len(X_test))
print("Reports saved in:", REPORT_PATH)
