import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from v2_pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RAW_FEATURES, build_pipeline


data = pd.read_csv("customer_data.csv")
X = data[RAW_FEATURES]
y = data["churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline = build_pipeline()
cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc",
    n_jobs=-1,
)
pipeline.fit(X_train, y_train)
probabilities = pipeline.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

joblib.dump(pipeline, "baseline_v2_pipeline.pkl")

transformed_feature_names = list(
    pipeline.named_steps["preprocessor"].get_feature_names_out()
)
metrics = {
    "model_name": "GradientBoostingClassifier(n_estimators=200, random_state=42)",
    "target": "churn",
    "threshold": 0.5,
    "raw_features": RAW_FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "transformed_feature_names": transformed_feature_names,
    "transformed_feature_count": len(transformed_feature_names),
    "cv_roc_auc_mean": float(cv_scores.mean()),
    "cv_roc_auc_std": float(cv_scores.std()),
    "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
    "test_precision": float(precision_score(y_test, predictions)),
    "test_recall": float(recall_score(y_test, predictions)),
    "test_f1": float(f1_score(y_test, predictions)),
    "test_accuracy": float(accuracy_score(y_test, predictions)),
    "fit_split": "test_size=0.2, stratify=y, random_state=42",
    "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
}
with open("baseline_v2_metadata.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print("Exported: baseline_v2_pipeline.pkl")
print("Exported: baseline_v2_metadata.json")
print(f"CV ROC-AUC: {metrics['cv_roc_auc_mean']:.4f} +/- {metrics['cv_roc_auc_std']:.4f}")
print(f"Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
print(f"Test F1: {metrics['test_f1']:.4f}")
print(f"Transformed feature count: {metrics['transformed_feature_count']}")