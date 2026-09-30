import math

import pandas as pd

from v2_pipeline import RAW_FEATURES


COUNTRIES = ("France", "Germany", "Spain")
GENDERS = ("Female", "Male")
BINARY_FIELDS = ("credit_card", "active_member")


def validate_customer(values):
    missing = [name for name in RAW_FEATURES if name not in values]
    if missing:
        raise ValueError(f"Missing customer fields: {', '.join(missing)}")

    customer = {name: values[name] for name in RAW_FEATURES}
    if customer["country"] not in COUNTRIES:
        raise ValueError(f"Country must be one of: {', '.join(COUNTRIES)}")
    if customer["gender"] not in GENDERS:
        raise ValueError(f"Gender must be one of: {', '.join(GENDERS)}")

    for name in ("credit_score", "age", "tenure", "balance", "products_number", "estimated_salary"):
        value = customer[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")

    if not 300 <= customer["credit_score"] <= 850:
        raise ValueError("Credit score must be between 300 and 850")
    if not 18 <= customer["age"] <= 100:
        raise ValueError("Age must be between 18 and 100")
    if not 0 <= customer["tenure"] <= 10:
        raise ValueError("Tenure must be between 0 and 10 years")
    if customer["balance"] < 0 or customer["estimated_salary"] < 0:
        raise ValueError("Balance and salary cannot be negative")
    if customer["products_number"] not in (1, 2, 3, 4):
        raise ValueError("Products number must be an integer from 1 to 4")
    if any(customer[name] not in (0, 1) for name in BINARY_FIELDS):
        raise ValueError("Credit card and active member values must be 0 or 1")

    return pd.DataFrame([customer], columns=RAW_FEATURES)