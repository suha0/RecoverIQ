
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path
from src.recommendation import recommend_next_step


# --------------------------------------------------
# PATHS AND MODEL LOADING
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "recovery_customers.csv"
MODEL_PATH = ROOT / "models" / "recovery_model.joblib"

REPORT_PATH = ROOT / "reports"

METRICS_PATH = REPORT_PATH / "model_metrics.csv"

st.set_page_config(
    page_title="RecoverIQ | Debt Recovery Analytics",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

artifact = load_model()
model = artifact["pipeline"]
threshold = artifact["threshold"]
model_name = artifact["model_name"]

data = load_data()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("RecoverIQ")
st.sidebar.caption("AI-powered Recovery Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Customer Prediction",
        "Model Information"
    ]
)


# --------------------------------------------------
# DASHBOARD OVERVIEW
# --------------------------------------------------

if page == "Dashboard Overview":

    st.title("Recovery Analytics Dashboard")

    st.write(
        "Explore synthetic customer repayment patterns "
        "and machine learning insights."
    )

    total_customers = len(data)

    repayment_rate = (
        data["repaid_next_30_days"].mean() * 100
    )

    average_overdue = data["days_overdue"].mean()

    average_outstanding = data["outstanding_amount"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Repayment Rate", f"{repayment_rate:.1f}%")
    col3.metric("Average Overdue", f"{average_overdue:.1f} days")
    col4.metric(
        "Average Outstanding",
        f"₹{average_outstanding:,.0f}"
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Repayment Distribution")

        distribution = (
            data["repaid_next_30_days"]
            .value_counts()
            .rename(index={0: "Not Repaid", 1: "Repaid"})
        )

        st.bar_chart(distribution)

    with right:
        st.subheader("Customers by Preferred Channel")

        channel_data = data["preferred_channel"].value_counts()

        st.bar_chart(channel_data)

    st.subheader("Customer Dataset Preview")

    st.dataframe(data.head(10), use_container_width=True)

    st.caption(
        "All records are synthetically generated for demonstration."
    )


# --------------------------------------------------
# CUSTOMER PREDICTION
# --------------------------------------------------

elif page == "Customer Prediction":

    st.title("Customer Repayment Prediction")

    st.write(
        "Enter customer information to generate an "
        "experimental repayment score."
    )

    with st.form("customer_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Customer Age",
                min_value=21,
                max_value=65,
                value=30
            )

            outstanding_amount = st.number_input(
                "Outstanding Amount (₹)",
                min_value=1.0,
                value=25000.0,
                step=1000.0
            )

            days_overdue = st.number_input(
                "Days Overdue",
                min_value=1,
                max_value=180,
                value=30
            )

            missed_payments = st.number_input(
                "Missed Payments",
                min_value=0,
                max_value=8,
                value=1
            )

            prior_repayment_rate = st.slider(
                "Previous Repayment Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.6
            )

        with col2:

            contacts_last_30_days = st.number_input(
                "Contacts in Last 30 Days",
                min_value=0,
                max_value=8,
                value=2
            )

            last_contact_days_ago = st.number_input(
                "Days Since Last Contact",
                min_value=1,
                max_value=60,
                value=7
            )

            income_band = st.selectbox(
                "Income Band",
                ["Low", "Medium", "High"]
            )

            preferred_channel = st.selectbox(
                "Preferred Channel",
                ["Call", "WhatsApp", "Email"]
            )

            payment_plan_active = st.selectbox(
                "Active Payment Plan?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        submitted = st.form_submit_button(
            "Predict Repayment",
            use_container_width=True
        )

    if submitted:

        customer = {
            "age": age,
            "outstanding_amount": outstanding_amount,
            "days_overdue": days_overdue,
            "missed_payments": missed_payments,
            "prior_repayment_rate": prior_repayment_rate,
            "contacts_last_30_days": contacts_last_30_days,
            "last_contact_days_ago": last_contact_days_ago,
            "income_band": income_band,
            "preferred_channel": preferred_channel,
            "payment_plan_active": payment_plan_active
        }

        customer_df = pd.DataFrame([customer])

        score = model.predict_proba(customer_df)[0][1]

        prediction = int(score >= threshold)

        st.divider()

        st.subheader("Prediction Results")

        result_col1, result_col2 = st.columns(2)

        result_col1.metric(
            "Model Score",
            f"{score * 100:.2f}%"
        )

        result_col2.metric(
            "Model Classification",
            "Likely to Repay" if prediction == 1
            else "Less Likely to Repay"
        )

        st.progress(float(score))

        recommendation = recommend_next_step(
            score,
            customer
        )

        st.subheader("Customer Support Recommendation")

        st.write("**Segment:**", recommendation["segment"])
        st.write("**Priority:**", recommendation["priority"])
        st.write("**Suggested Channel:**", recommendation["channel"])

        st.info(recommendation["recommendation"])

        st.caption(
            "Experimental score from a model trained on synthetic data. "
            "It is not a calibrated real-world repayment probability "
            "or a basis for adverse customer treatment."
        )


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------


elif page == "Model Information":

    st.title("Model Information")

    st.subheader("Training Configuration")

    col1, col2, col3 = st.columns(3)

    col1.metric("Selected Model", model_name)
    col2.metric("Decision Threshold", f"{threshold:.2f}")
    col3.metric("Test Records", "1,000")

    st.write("**Prediction Target:** Repayment within the next 30 days")
    st.write("**Dataset:** 5,000 synthetic customer records")

    st.divider()

    st.subheader("Model Performance")

    if METRICS_PATH.exists():

        metrics_df = pd.read_csv(METRICS_PATH)

        metric_columns = st.columns(3)

        for index, row in metrics_df.iterrows():

            metric_columns[index % 3].metric(
                row["Metric"],
                f"{row['Score']:.2%}"
            )

    else:
        st.warning("Evaluation metrics not found. Run evaluate_model.py.")

    st.divider()

    st.subheader("Evaluation Visualizations")

    charts = [
        ("Confusion Matrix", "confusion_matrix.png"),
        ("ROC Curve", "roc_curve.png"),
        ("Precision-Recall Curve", "precision_recall_curve.png"),
        ("Feature Coefficients", "feature_coefficients.png")
    ]

    for index in range(0, len(charts), 2):

        col1, col2 = st.columns(2)

        for column, chart in zip(
            [col1, col2],
            charts[index:index + 2]
        ):

            title, filename = chart
            image_path = REPORT_PATH / filename

            with column:
                st.markdown(f"**{title}**")

                if image_path.exists():
                    st.image(
                        str(image_path),
                        use_container_width=True
                    )
                else:
                    st.info(f"{filename} not found.")

    st.divider()

    st.subheader("Project Limitations")

    st.warning(
        "RecoverIQ uses synthetic data. Its model scores are "
        "experimental and are not calibrated real-world repayment "
        "probabilities. Further validation, calibration, fairness "
        "testing, privacy review and monitoring are necessary "
        "before any real-world deployment."
    )


st.sidebar.divider()
st.sidebar.caption("RecoverIQ | Portfolio Project")
