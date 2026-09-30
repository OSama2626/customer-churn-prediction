import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from unittest.mock import Mock

from src.batch import predict_batch, validate_batch
from src.explainability import explain_customer, load_global_importance, readable_feature_name
from src.predictor import predict_customer
from src.validation import validate_customer
from v2_pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RAW_FEATURES


ROOT = Path(__file__).resolve().parents[1]


class V2PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = joblib.load(ROOT / "baseline_v2_pipeline.pkl")
        cls.xgboost_pipeline = joblib.load(ROOT / "xgboost_v2_pipeline.pkl")
        with (ROOT / "baseline_v2_metadata.json").open(encoding="utf-8") as file:
            cls.metadata = json.load(file)
        with (ROOT / "xgboost_v2_metadata.json").open(encoding="utf-8") as file:
            cls.xgboost_metadata = json.load(file)
        cls.data = pd.read_csv(ROOT / "customer_data.csv")

    def test_saved_pipeline_has_expected_raw_schema(self):
        self.assertEqual(len(RAW_FEATURES), 10)
        self.assertEqual(self.metadata["raw_features"], RAW_FEATURES)
        self.assertEqual(self.pipeline.named_steps["feature_engineering"].feature_names_in_.tolist(), RAW_FEATURES)

    def test_preprocessor_has_expected_feature_groups_and_order(self):
        self.assertEqual(self.metadata["numeric_features"], NUMERIC_FEATURES)
        self.assertEqual(self.metadata["categorical_features"], CATEGORICAL_FEATURES)
        names = self.pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
        self.assertEqual(len(names), 29)
        self.assertEqual(names, self.metadata["transformed_feature_names"])

    def test_predict_probability_accepts_raw_customer(self):
        customer = self.data[RAW_FEATURES].iloc[[0]]
        probability = self.pipeline.predict_proba(customer)
        prediction = self.pipeline.predict(customer)

        self.assertEqual(probability.shape, (1, 2))
        self.assertTrue(np.isfinite(probability).all())
        self.assertAlmostEqual(float(probability.sum()), 1.0, places=7)
        self.assertIn(int(prediction[0]), (0, 1))

    def test_missing_raw_column_is_rejected(self):
        customer = self.data[RAW_FEATURES].iloc[[0]].drop(columns=["country"])
        with self.assertRaisesRegex(ValueError, "Missing raw columns"):
            self.pipeline.predict_proba(customer)

    def test_customer_validation_rejects_invalid_and_accepts_boundaries(self):
        customer = self.data[RAW_FEATURES].iloc[0].to_dict()
        customer.update({"age": 18, "credit_score": 300, "tenure": 0, "products_number": 1})
        self.assertEqual(list(validate_customer(customer).columns), RAW_FEATURES)

        customer["age"] = 101
        with self.assertRaisesRegex(ValueError, "Age must be between"):
            validate_customer(customer)

    def test_threshold_050_and_030_classification(self):
        customer = self.data[RAW_FEATURES].iloc[0].to_dict()
        model = Mock()
        model.predict_proba.return_value = np.array([[0.60, 0.40]])

        standard = predict_customer(customer, "xgboost", 0.50, pipeline=model)
        recall_focused = predict_customer(customer, "xgboost", 0.30, pipeline=model)

        self.assertEqual(standard["prediction"], 0)
        self.assertEqual(recall_focused["prediction"], 1)
        self.assertEqual(standard["threshold"], 0.50)
        self.assertEqual(recall_focused["threshold"], 0.30)

    def test_default_threshold_comes_from_each_models_metadata(self):
        customer = self.data[RAW_FEATURES].iloc[0].to_dict()
        model = Mock()
        model.predict_proba.return_value = np.array([[0.60, 0.40]])

        baseline = predict_customer(customer, "gradient_boosting", pipeline=model)
        xgboost = predict_customer(customer, "xgboost", pipeline=model)

        self.assertEqual(baseline["threshold"], self.metadata["threshold"])
        self.assertEqual(xgboost["threshold"], self.xgboost_metadata["threshold"])

    def test_batch_validation_and_default_threshold_use_selected_model_metadata(self):
        customers = self.data[RAW_FEATURES].head(3)
        validated = validate_batch(customers)
        predictions = predict_batch(customers, model_name="xgboost", pipeline=self.xgboost_pipeline)

        self.assertEqual(list(validated.columns), RAW_FEATURES)
        self.assertEqual(len(predictions), 3)
        self.assertTrue(predictions["churn_probability"].between(0, 1).all())
        self.assertEqual(
            predictions["prediction"].tolist(),
            ["Likely churn" if p >= self.xgboost_metadata["threshold"] else "Likely stay"
             for p in predictions["churn_probability"]],
        )
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            validate_batch(customers.drop(columns=["country"]))

    def test_inference_does_not_refit_train_statistics(self):
        engineer = self.pipeline.named_steps["feature_engineering"]
        ratio_mean = engineer.salary_balance_ratio_mean_
        balance_threshold = engineer.high_balance_threshold_

        self.pipeline.predict_proba(self.data[RAW_FEATURES].iloc[:5])

        self.assertEqual(engineer.salary_balance_ratio_mean_, ratio_mean)
        self.assertEqual(engineer.high_balance_threshold_, balance_threshold)

    def test_saved_artifact_reproduces_recorded_baseline_metrics(self):
        X = self.data[RAW_FEATURES]
        y = self.data["churn"]
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        probabilities = self.pipeline.predict_proba(X_test)[:, 1]
        predictions = self.pipeline.predict(X_test)
        measured = {
            "test_roc_auc": roc_auc_score(y_test, probabilities),
            "test_precision": precision_score(y_test, predictions),
            "test_recall": recall_score(y_test, predictions),
            "test_f1": f1_score(y_test, predictions),
            "test_accuracy": accuracy_score(y_test, predictions),
        }

        for metric, actual in measured.items():
            self.assertAlmostEqual(actual, self.metadata[metric], places=10, msg=metric)
        self.assertAlmostEqual(measured["test_roc_auc"], 0.8716, places=4)
        self.assertAlmostEqual(measured["test_f1"], 0.5961, places=4)

    def test_xgboost_shap_names_match_model_transformed_features(self):
        customer = self.data[RAW_FEATURES].iloc[[0]]
        explanation, contributions = explain_customer(self.xgboost_pipeline, customer)
        transformed_names = self.xgboost_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()

        self.assertEqual(self.xgboost_metadata["shap_explainer"], "TreeExplainer")
        self.assertEqual(explanation.values.shape, (1, len(transformed_names)))
        self.assertEqual(contributions["feature"].tolist(), transformed_names)
        self.assertTrue(all(name.startswith(("num__", "cat__")) for name in transformed_names))
        self.assertIn("Number of products", [readable_feature_name("num__products_number")])

    def test_saved_global_shap_importance_has_readable_mapping(self):
        importance = load_global_importance()
        self.assertEqual(len(importance), self.xgboost_metadata["transformed_feature_count"])
        self.assertTrue(importance["readable_feature"].notna().all())


if __name__ == "__main__":
    unittest.main()
