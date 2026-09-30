import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
BASELINE_PIPELINE_PATH = ROOT / "baseline_v2_pipeline.pkl"
BASELINE_METADATA_PATH = ROOT / "baseline_v2_metadata.json"
XGBOOST_PIPELINE_PATH = ROOT / "xgboost_v2_pipeline.pkl"
XGBOOST_METADATA_PATH = ROOT / "xgboost_v2_metadata.json"
SHAP_IMPORTANCE_PATH = ROOT / "shap_global_importance.csv"


def load_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def get_model_metadata(model_name):
    paths = {
        "gradient_boosting": BASELINE_METADATA_PATH,
        "xgboost": XGBOOST_METADATA_PATH,
    }
    try:
        return load_json(paths[model_name])
    except KeyError as error:
        raise ValueError(f"Unknown model: {model_name}") from error


_baseline_metadata = load_json(BASELINE_METADATA_PATH)
BASELINE_METRICS = {
    "cv_roc_auc_mean": _baseline_metadata["cv_roc_auc_mean"],
    "cv_roc_auc_std": _baseline_metadata["cv_roc_auc_std"],
    "test_roc_auc": _baseline_metadata["test_roc_auc"],
    "precision": _baseline_metadata["test_precision"],
    "recall": _baseline_metadata["test_recall"],
    "f1": _baseline_metadata["test_f1"],
    "accuracy": _baseline_metadata["test_accuracy"],
}
