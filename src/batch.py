import pandas as pd
import math

from src.config import get_model_metadata
from src.predictor import load_pipeline
from src.validation import validate_customer
from v2_pipeline import RAW_FEATURES


def validate_batch(frame):
    """Validate the raw CSV schema and values before inference."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Upload a CSV file containing customer rows.")
    missing = [name for name in RAW_FEATURES if name not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("The CSV has no customer rows.")

    customers = frame.loc[:, RAW_FEATURES].copy()
    for name in ("credit_score", "age", "tenure", "balance", "products_number", "credit_card", "active_member", "estimated_salary"):
        customers[name] = pd.to_numeric(customers[name], errors="coerce")
    errors = []
    for index, row in customers.iterrows():
        try:
            validate_customer(row.to_dict())
        except (ValueError, TypeError) as error:
            errors.append(f"Row {index + 2}: {error}")
            if len(errors) == 5:
                break
    if errors:
        suffix = " Additional errors may exist." if len(errors) == 5 else ""
        raise ValueError("Invalid customer data. " + "; ".join(errors) + suffix)
    return customers


def predict_batch(frame, threshold=None, model_name="gradient_boosting", pipeline=None):
    customers = validate_batch(frame)
    if threshold is None:
        threshold = get_model_metadata(model_name)["threshold"]
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Decision threshold must be between 0 and 1.")
    model = pipeline if pipeline is not None else load_pipeline(model_name)
    probabilities = model.predict_proba(customers)[:, 1]
    results = frame.copy()
    results["churn_probability"] = probabilities
    results["prediction"] = ["Likely churn" if value >= threshold else "Likely stay" for value in probabilities]
    results["risk_level"] = ["High" if value >= threshold else "Low" for value in probabilities]
    return results
