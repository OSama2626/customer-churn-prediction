import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAW_FEATURES = [
    "credit_score", "country", "gender", "age", "tenure", "balance",
    "products_number", "credit_card", "active_member", "estimated_salary",
]
NUMERIC_FEATURES = [
    "credit_score", "age", "tenure", "balance", "products_number",
    "estimated_salary", "balance_per_product", "salary_balance_ratio",
]
CATEGORICAL_FEATURES = [
    "gender", "country", "credit_card", "active_member", "age_group",
    "tenure_bucket", "high_balance",
]


class ChurnFeatureEngineerV1(BaseEstimator, TransformerMixin):
    def __init__(self, feature_families=()):
        self.feature_families = feature_families

    def fit(self, X, y=None):
        required_columns = set(RAW_FEATURES)
        missing_columns = required_columns - set(X.columns)
        if missing_columns:
            raise ValueError(f"Missing raw columns: {sorted(missing_columns)}")
        if set(self.feature_families) - {"ratios", "interactions", "flags"}:
            raise ValueError(f"Unknown feature families: {self.feature_families}")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        ratio = (X["estimated_salary"] / X["balance"].replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
        self.salary_balance_ratio_mean_ = float(ratio.mean())
        if not np.isfinite(self.salary_balance_ratio_mean_):
            self.salary_balance_ratio_mean_ = 0.0
        self.high_balance_threshold_ = float(X["balance"].quantile(0.75))
        return self

    def transform(self, X):
        missing_columns = set(self.feature_names_in_) - set(X.columns)
        if missing_columns:
            raise ValueError(f"Missing raw columns: {sorted(missing_columns)}")

        transformed = X.loc[:, self.feature_names_in_].copy()
        transformed["balance_per_product"] = (
            transformed["balance"] / transformed["products_number"].replace(0, np.nan)
        ).fillna(0)
        ratio = (transformed["estimated_salary"] / transformed["balance"].replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
        transformed["salary_balance_ratio"] = ratio.fillna(self.salary_balance_ratio_mean_)
        transformed["age_group"] = pd.cut(
            transformed["age"],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
        )
        transformed["tenure_bucket"] = pd.cut(
            transformed["tenure"],
            bins=[0, 1, 2, 5, 10, 100],
            labels=["0", "1-2", "3-5", "6-10", "10+"],
        )
        transformed["high_balance"] = (
            transformed["balance"] > self.high_balance_threshold_
        ).astype(int)

        categorical_columns = [
            "gender", "country", "credit_card", "active_member",
            "age_group", "tenure_bucket", "high_balance",
        ]
        transformed[categorical_columns] = transformed[categorical_columns].astype(object)
        return transformed

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_ if input_features is None else input_features)
        names.extend([
            "balance_per_product", "salary_balance_ratio", "age_group",
            "tenure_bucket", "high_balance",
        ])
        return np.asarray(names, dtype=object)


def build_pipeline():
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("feature_engineering", ChurnFeatureEngineerV1()),
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(n_estimators=200, random_state=42)),
    ])