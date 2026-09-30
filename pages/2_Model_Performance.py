import streamlit as st

from src.config import BASELINE_METRICS, get_model_metadata
from src.ui import apply_styles, metric_panel, page_header


apply_styles()
metadata = get_model_metadata("gradient_boosting")
threshold = float(metadata["threshold"])
page_header(
    "Performance",
    "Recorded model performance",
    "Metrics below come from the saved reference-model metadata and describe the recorded evaluation, not guaranteed future performance.",
)

st.info(f"Reference model: **{metadata['model_name']}** · evaluated threshold: **{threshold:.2f}**")
columns = st.columns(3, gap="small")
with columns[0]:
    metric_panel(
        "Cross-validation ROC-AUC",
        f"{BASELINE_METRICS['cv_roc_auc_mean']:.4f}",
        f"± {BASELINE_METRICS['cv_roc_auc_std']:.4f}",
    )
with columns[1]:
    metric_panel("Held-out test ROC-AUC", f"{BASELINE_METRICS['test_roc_auc']:.4f}", "Ranking across thresholds")
with columns[2]:
    metric_panel("Held-out test precision", f"{BASELINE_METRICS['precision']:.4f}", f"At threshold {threshold:.2f}")

columns = st.columns(3, gap="small")
with columns[0]:
    metric_panel("Held-out test recall", f"{BASELINE_METRICS['recall']:.4f}", f"At threshold {threshold:.2f}")
with columns[1]:
    metric_panel("Held-out test F1", f"{BASELINE_METRICS['f1']:.4f}", f"At threshold {threshold:.2f}")
with columns[2]:
    metric_panel("Held-out test accuracy", f"{BASELINE_METRICS['accuracy']:.2%}", f"At threshold {threshold:.2f}")

st.subheader("How to read these results")
st.write("ROC-AUC measures how well the model ranks churners above non-churners across possible thresholds. It does not measure probability calibration.")
st.write("A probability is the pipeline's estimated churn score. At the saved threshold, scores at or above the threshold are classified as likely churn. Lowering a threshold generally flags more customers and may also increase false alarms.")
st.caption("The baseline metadata contains these CV and test metrics. The XGBoost metadata contains its model settings and threshold, but no evaluation metrics, so no XGBoost performance scores are reported here.")
st.warning("The test split was used during project model comparisons. These results are not an independent external validation.")
