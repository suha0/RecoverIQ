
import pandas as pd
import joblib

from pathlib import Path


# Project paths
ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "recovery_model.joblib"


# Load trained model
artifact = joblib.load(MODEL_PATH)

model = artifact["pipeline"]
threshold = artifact["threshold"]

print("\n====================================")
print("       RECOVERIQ PREDICTION ENGINE")
print("====================================")

print("Model:", artifact["model_name"])
print("Decision threshold:", threshold)


def predict_customer(customer):

    # Convert customer details into DataFrame
    customer_df = pd.DataFrame([customer])

    # Generate repayment score
    repayment_score = model.predict_proba(
        customer_df
    )[0][1]

    # Classify using selected threshold
    prediction = int(repayment_score >= threshold)

    print("\n===== CUSTOMER PREDICTION =====")

    print(
        "Repayment score:",
        round(repayment_score * 100, 2),
        "%"
    )

    if prediction == 1:
        print("Prediction: LIKELY TO REPAY")
    else:
        print("Prediction: LESS LIKELY TO REPAY")

    return repayment_score, prediction


# Collect customer information
try:

    customer = {
        "age": int(input("\nCustomer age: ")),

        "outstanding_amount": float(
            input("Outstanding amount: ")
        ),

        "days_overdue": int(
            input("Days overdue: ")
        ),

        "missed_payments": int(
            input("Number of missed payments: ")
        ),

        "prior_repayment_rate": float(
            input("Previous repayment rate (0 to 1): ")
        ),

        "contacts_last_30_days": int(
            input("Contacts in last 30 days: ")
        ),

        "last_contact_days_ago": int(
            input("Days since last contact: ")
        ),

        "income_band": input(
            "Income band (Low/Medium/High): "
        ).strip().title(),

        "preferred_channel": input(
            "Preferred channel (Call/WhatsApp/Email): "
        ).strip().title(),

        "payment_plan_active": int(
            input("Active payment plan? (0=No, 1=Yes): ")
        )
    }

    # Basic input validation
    if not 21 <= customer["age"] <= 65:
        raise ValueError("Age must be between 21 and 65.")

    if customer["outstanding_amount"] <= 0:
        raise ValueError("Outstanding amount must be positive.")

    if not 1 <= customer["days_overdue"] <= 180:
        raise ValueError("Days overdue must be between 1 and 180.")

    if not 0 <= customer["missed_payments"] <= 8:
        raise ValueError("Missed payments must be between 0 and 8.")

    if not 0 <= customer["prior_repayment_rate"] <= 1:
        raise ValueError("Repayment rate must be between 0 and 1.")

    if not 0 <= customer["contacts_last_30_days"] <= 8:
        raise ValueError("Contact count must be between 0 and 8.")

    if not 1 <= customer["last_contact_days_ago"] <= 60:
        raise ValueError("Last contact days must be between 1 and 60.")

    if customer["income_band"] not in ["Low", "Medium", "High"]:
        raise ValueError("Invalid income band.")

    if customer["preferred_channel"] not in [
        "Call", "Whatsapp", "Email"
    ]:
        raise ValueError("Invalid contact channel.")

    if customer["payment_plan_active"] not in [0, 1]:
        raise ValueError("Payment plan must be 0 or 1.")

    predict_customer(customer)

except ValueError as error:
    print("\nInput error:", error)
