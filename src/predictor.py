import joblib
import math
import streamlit as st

from src.config import BASELINE_PIPELINE_PATH, XGBOOST_PIPELINE_PATH, get_model_metadata
from src.validation import validate_customer


@st.cache_resource(show_spinner="Loading saved churn model...")
def load_pipeline(model_name="gradient_boosting"):
    if model_name not in {"gradient_boosting", "xgboost"}:
        raise ValueError(f"Unknown model: {model_name}")
    path = BASELINE_PIPELINE_PATH if model_name == "gradient_boosting" else XGBOOST_PIPELINE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path.name}")
    return joblib.load(path)


def predict_customer(values, model_name="gradient_boosting", threshold=None, pipeline=None):
    customer = validate_customer(values)
    model = pipeline if pipeline is not None else load_pipeline(model_name)
    probability = float(model.predict_proba(customer)[0, 1])
    if threshold is None:
        threshold = get_model_metadata(model_name)["threshold"]
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Decision threshold must be between 0 and 1.")
    prediction = int(probability >= threshold)
    return {
        "prediction": prediction,
        "probability": probability,
        "threshold": threshold,
        "model_name": model_name,
    }
