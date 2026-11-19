
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def build_preprocessor():

    numerical_features = [
        "age",
        "outstanding_amount",
        "days_overdue",
        "missed_payments",
        "prior_repayment_rate",
        "contacts_last_30_days",
        "last_contact_days_ago",
        "payment_plan_active"
    ]

    categorical_features = [
        "income_band",
        "preferred_channel"
    ]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return preprocessor
