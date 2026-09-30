import numpy as np
import pandas as pd
import shap

from src.config import SHAP_IMPORTANCE_PATH
from v2_pipeline import RAW_FEATURES


def readable_feature_name(name):
    if name.startswith("num__"):
        labels = {
            "credit_score": "Credit score",
            "age": "Customer age",
            "tenure": "Customer tenure (years)",
            "balance": "Account balance",
            "products_number": "Number of products",
            "estimated_salary": "Estimated salary",
            "balance_per_product": "Balance per product",
            "salary_balance_ratio": "Salary-to-balance ratio",
        }
        feature = name.removeprefix("num__")
        return labels.get(feature, feature.replace("_", " ").title())
    if name.startswith("cat__"):
        encoded = name.removeprefix("cat__")
        for field in ("active_member", "credit_card", "age_group", "tenure_bucket", "country", "gender", "high_balance"):
            prefix = f"{field}_"
            if encoded.startswith(prefix):
                label = encoded.removeprefix(prefix)
                readable_field = {
                    "active_member": "Active customer",
                    "credit_card": "Credit card",
                    "age_group": "Age group",
                    "tenure_bucket": "Tenure group (years)",
                    "high_balance": "High balance",
                    "country": "Country",
                    "gender": "Gender",
                }[field]
                if field in {"active_member", "credit_card", "high_balance"}:
                    label = {"0": "No", "1": "Yes"}.get(label, label)
                return f"{readable_field}: {label}"
        return encoded.replace("_", " ").title()
    return str(name)


def load_global_importance():
    if not SHAP_IMPORTANCE_PATH.is_file():
        raise FileNotFoundError("Global SHAP artifact is not available.")
    importance = pd.read_csv(SHAP_IMPORTANCE_PATH)
    if not {"feature", "mean_absolute_shap", "mean_shap"}.issubset(importance.columns):
        raise ValueError("Global SHAP artifact has an unexpected schema.")
    importance["readable_feature"] = importance["feature"].map(readable_feature_name)
    return importance


def explain_customer(pipeline, raw_customer):
    if list(raw_customer.columns) != RAW_FEATURES:
        raw_customer = raw_customer.loc[:, RAW_FEATURES]
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    transformed = pipeline[:-1].transform(raw_customer)
    transformed = np.asarray(transformed)
    classifier = pipeline.named_steps["classifier"]
    explanation = shap.TreeExplainer(classifier)(transformed)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, 1]
    if values.shape != (1, len(feature_names)):
        raise ValueError("SHAP values do not match the transformed model features.")
    explanation.feature_names = list(feature_names)
    contributions = pd.DataFrame({
        "feature": feature_names,
        "readable_feature": [readable_feature_name(name) for name in feature_names],
        "shap_value": values[0],
    })
    return explanation, contributions
