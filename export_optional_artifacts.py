import json

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from v2_pipeline import RAW_FEATURES, build_pipeline


XGB_CONFIG = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": 1,
}
SPLIT_CONFIG = {
    "test_size": 0.2,
    "stratify": "churn",
    "random_state": 42,
}


data = pd.read_csv("customer_data.csv")
X_train, X_test, y_train, y_test = train_test_split(
    data[RAW_FEATURES],
    data["churn"],
    test_size=SPLIT_CONFIG["test_size"],
    stratify=data["churn"],
    random_state=SPLIT_CONFIG["random_state"],
)

pipeline = build_pipeline()
pipeline.set_params(classifier=XGBClassifier(**XGB_CONFIG))
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "xgboost_v2_pipeline.pkl")

preprocessor = pipeline.named_steps["preprocessor"]
feature_names = preprocessor.get_feature_names_out()
train_matrix = pipeline[:-1].transform(X_train)
train_transformed = pd.DataFrame(train_matrix, columns=feature_names, index=X_train.index)
shap_sample = train_transformed.sample(n=min(1000, len(train_transformed)), random_state=42)
explainer = shap.TreeExplainer(pipeline.named_steps["classifier"])
explanation = explainer(shap_sample.to_numpy())
shap_values = np.asarray(explanation.values)
if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]
if shap_values.shape != shap_sample.shape:
    raise ValueError(
        f"SHAP dimensions {shap_values.shape} do not match transformed data {shap_sample.shape}"
    )

global_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_absolute_shap": np.abs(shap_values).mean(axis=0),
    "mean_shap": shap_values.mean(axis=0),
}).sort_values("mean_absolute_shap", ascending=False)
global_importance.to_csv("shap_global_importance.csv", index=False)

metadata = {
    "model_name": "XGBClassifier (untuned alternative)",
    "model_parameters": XGB_CONFIG,
    "target": "churn",
    "threshold": 0.30,
    "raw_features": RAW_FEATURES,
    "transformed_feature_names": list(feature_names),
    "transformed_feature_count": len(feature_names),
    "fit_split": "test_size=0.2, stratify=y, random_state=42",
    "shap_explainer": "TreeExplainer",
    "shap_sample_rows": len(shap_sample),
    "shap_source": "training split only, random_state=42",
    "shap_output": "raw model output (log-odds)",
}
with open("xgboost_v2_metadata.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)

print("Exported: xgboost_v2_pipeline.pkl")
print("Exported: xgboost_v2_metadata.json")
print("Exported: shap_global_importance.csv")
print(f"Training rows used for SHAP: {len(shap_sample)}")
print(f"Transformed feature count: {len(feature_names)}")