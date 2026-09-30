import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class StreamlitAppTests(unittest.TestCase):
    def test_overview_starts_and_links_to_primary_workflows(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)

        self.assertFalse(app.exception)
        link_labels = {element.label for element in app if getattr(element, "label", None)}
        self.assertIn("Open customer prediction", link_labels)
        self.assertIn("Review model performance", link_labels)

    def test_each_page_starts_without_an_exception(self):
        pages = sorted((ROOT / "pages").glob("*.py"))
        self.assertEqual(len(pages), 4)
        for page in pages:
            with self.subTest(page=page.name):
                result = AppTest.from_file(str(page)).run(timeout=30)
                self.assertFalse(result.exception)

    def test_prediction_form_renders_a_result(self):
        page = AppTest.from_file(str(ROOT / "pages" / "1_Customer_Prediction.py")).run(timeout=30)
        page.button[0].click().run(timeout=30)

        self.assertFalse(page.exception)
        self.assertEqual(
            [metric.label for metric in page.metric],
            ["Churn probability", "Classification", "Applied threshold"],
        )

    def test_xgboost_metadata_threshold_renders_prediction(self):
        page = AppTest.from_file(str(ROOT / "pages" / "1_Customer_Prediction.py")).run(timeout=30)
        page.radio[0].set_value("XGBoost · detection option").run(timeout=30)
        page.button[0].click().run(timeout=30)

        self.assertFalse(page.exception)
        self.assertIn("0.30", [metric.value for metric in page.metric])

    def test_local_shap_workflow_renders_for_customer(self):
        page = AppTest.from_file(str(ROOT / "pages" / "3_Model_Insights.py")).run(timeout=30)
        page.button[0].click().run(timeout=30)

        self.assertFalse(page.exception)
        rendered_text = "\n".join(element.value for element in page.markdown)
        rendered_captions = "\n".join(element.value for element in page.caption)
        self.assertIn("XGBClassifier", page.info[0].value)
        self.assertIn("SHAP values are log-odds contributions", rendered_captions)


if __name__ == "__main__":
    unittest.main()
