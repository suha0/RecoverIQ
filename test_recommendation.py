
from src.recommendation import display_recommendation

customer = {
    "contacts_last_30_days": 2,
    "payment_plan_active": 0
}

# Example model score: 72%
repayment_score = 0.72

display_recommendation(repayment_score, customer)
