
import numpy as np
import pandas as pd
from pathlib import Path

# Reproducible synthetic dataset
rng = np.random.default_rng(42)

NUMBER_OF_CUSTOMERS = 5000

# Generate customer information
customer_id = [
    f"REC{i:05d}" for i in range(1, NUMBER_OF_CUSTOMERS + 1)
]

age = rng.integers(21, 66, NUMBER_OF_CUSTOMERS)

outstanding_amount = rng.integers(
    5000, 200001, NUMBER_OF_CUSTOMERS
)

days_overdue = rng.integers(
    1, 181, NUMBER_OF_CUSTOMERS
)

missed_payments = rng.integers(
    0, 9, NUMBER_OF_CUSTOMERS
)

prior_repayment_rate = np.round(
    rng.beta(2.5, 2.5, NUMBER_OF_CUSTOMERS), 2
)

contacts_last_30_days = rng.integers(
    0, 9, NUMBER_OF_CUSTOMERS
)

last_contact_days_ago = rng.integers(
    1, 61, NUMBER_OF_CUSTOMERS
)

income_band = rng.choice(
    ["Low", "Medium", "High"],
    NUMBER_OF_CUSTOMERS,
    p=[0.35, 0.45, 0.20]
)

preferred_channel = rng.choice(
    ["Call", "WhatsApp", "Email"],
    NUMBER_OF_CUSTOMERS,
    p=[0.40, 0.40, 0.20]
)

payment_plan_active = rng.choice(
    [0, 1],
    NUMBER_OF_CUSTOMERS,
    p=[0.75, 0.25]
)

# Simulate repayment probability
# These are illustrative assumptions, not real-world findings.

income_effect = np.select(
    [
        income_band == "Low",
        income_band == "Medium",
        income_band == "High"
    ],
    [-0.25, 0.10, 0.35],
    default=0
)

logit = (
    -0.45
    + 2.0 * prior_repayment_rate
    - 0.012 * days_overdue
    - 0.28 * missed_payments
    - 0.055 * (outstanding_amount / 10000)
    - 0.07 * contacts_last_30_days
    + 0.45 * payment_plan_active
    + income_effect
    + 0.15 * (last_contact_days_ago / 30)
)

repayment_probability = 1 / (1 + np.exp(-logit))

# Simulate repayment outcome within the next 30 days
repaid_next_30_days = rng.binomial(
    1, repayment_probability
)

# Build dataset
df = pd.DataFrame({
    "customer_id": customer_id,
    "age": age,
    "outstanding_amount": outstanding_amount,
    "days_overdue": days_overdue,
    "missed_payments": missed_payments,
    "prior_repayment_rate": prior_repayment_rate,
    "contacts_last_30_days": contacts_last_30_days,
    "last_contact_days_ago": last_contact_days_ago,
    "income_band": income_band,
    "preferred_channel": preferred_channel,
    "payment_plan_active": payment_plan_active,
    "repaid_next_30_days": repaid_next_30_days
})

# Save dataset
project_root = Path(__file__).resolve().parents[1]
data_folder = project_root / "data"
data_folder.mkdir(exist_ok=True)

output_file = data_folder / "recovery_customers.csv"

df.to_csv(output_file, index=False)

# Display summary
print("\n===== RECOVERIQ DATASET GENERATED =====")
print("Total customers:", len(df))
print("Total features:", len(df.columns) - 2)
print(
    "Customers repaid:",
    int(df["repaid_next_30_days"].sum())
)
print(
    "Customers not repaid:",
    int((df["repaid_next_30_days"] == 0).sum())
)
print(
    "Repayment rate:",
    round(df["repaid_next_30_days"].mean() * 100, 2),
    "%"
)
print("\nDataset saved at:", output_file)
print("\nFirst five records:")
print(df.head())
print("\nDataset information:")
print(df.info())
