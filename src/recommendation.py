
def recommend_next_step(repayment_score, customer):
    """
    Generate a responsible customer-support recommendation.

    repayment_score: model score between 0 and 1
    customer: dictionary containing customer information
    """

    score = repayment_score * 100
    contacts = customer["contacts_last_30_days"]
    payment_plan = customer["payment_plan_active"]

    # Rule 1: Respect contact frequency limits
    if contacts >= 5:
        return {
            "segment": "CONTACT LIMIT REACHED",
            "priority": "Review",
            "channel": "No additional automated contact",
            "recommendation": (
                "Pause automated outreach and review contact history, "
                "consent, and applicable contact limits."
            )
        }

    # Rule 2: Support customers already on payment plans
    if payment_plan == 1:
        return {
            "segment": "ACTIVE PAYMENT PLAN",
            "priority": "Support",
            "channel": "Customer's consented channel",
            "recommendation": (
                "Provide payment-plan support and a self-service option. "
                "Avoid unnecessary reminders."
            )
        }

    # Rule 3: Higher model score
    if score >= 65:
        return {
            "segment": "HIGHER MODEL SCORE",
            "priority": "Normal",
            "channel": "Customer's consented preferred channel",
            "recommendation": (
                "Consider a polite payment reminder with a convenient "
                "self-service payment option."
            )
        }

    # Rule 4: Intermediate model score
    elif score >= 35:
        return {
            "segment": "MODERATE MODEL SCORE",
            "priority": "Support",
            "channel": "Customer's consented preferred channel",
            "recommendation": (
                "Provide a supportive reminder and explain available "
                "payment-plan or assistance options."
            )
        }

    # Rule 5: Lower model score
    else:
        return {
            "segment": "LOWER MODEL SCORE",
            "priority": "Human review",
            "channel": "No automatic channel selection",
            "recommendation": (
                "Refer for human review. Check whether the customer "
                "needs assistance or alternative repayment arrangements. "
                "Do not increase contact pressure based on this score."
            )
        }


def display_recommendation(repayment_score, customer):

    result = recommend_next_step(repayment_score, customer)

    print("\n====================================")
    print("       CUSTOMER RECOMMENDATION")
    print("====================================")

    print("Model score:", round(repayment_score * 100, 2), "%")
    print("Customer segment:", result["segment"])
    print("Priority:", result["priority"])
    print("Suggested channel:", result["channel"])

    print("\nRecommended next step:")
    print(result["recommendation"])

    return result
