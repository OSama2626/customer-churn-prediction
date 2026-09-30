import streamlit as st

from src.config import BASELINE_METRICS, get_model_metadata
from src.ui import apply_styles, metric_panel, page_header


baseline = get_model_metadata("gradient_boosting")
threshold = float(baseline["threshold"])
st.set_page_config(
    page_title="Churn Signal | Customer Retention Intelligence",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_styles()

st.sidebar.markdown("### CHURN SIGNAL")
st.sidebar.caption("Customer retention · V2")
st.sidebar.markdown("---")

page_header(
    "Overview",
    "Customer churn, made actionable.",
    "Explore customer churn scores from the saved V2 pipeline, with model evidence alongside each prediction.",
)

st.markdown("<div class='section-kicker'>Reference model · Gradient Boosting</div>", unsafe_allow_html=True)
st.info(f"Active overview model: **{baseline['model_name']}** · saved decision threshold: **{threshold:.2f}**")
metrics = st.columns(4, gap="small")
with metrics[0]:
    metric_panel("CV ROC-AUC", f"{BASELINE_METRICS['cv_roc_auc_mean']:.4f}", f"± {BASELINE_METRICS['cv_roc_auc_std']:.4f} · {baseline['cv']}")
with metrics[1]:
    metric_panel("Test ROC-AUC", f"{BASELINE_METRICS['test_roc_auc']:.4f}", baseline["fit_split"])
with metrics[2]:
    metric_panel("Test F1", f"{BASELINE_METRICS['f1']:.4f}", f"Threshold {threshold:.2f}")
with metrics[3]:
    metric_panel("Test accuracy", f"{BASELINE_METRICS['accuracy']:.2%}", f"Threshold {threshold:.2f}")

st.markdown("<div class='section-kicker'>Workspace</div>", unsafe_allow_html=True)
left, right = st.columns([1.1, 1], gap="large")
with left:
    st.subheader("From profile to signal")
    st.write("Enter a customer profile or upload a CSV to receive probabilities and threshold-based classifications.")
    st.page_link("pages/1_Customer_Prediction.py", label="Open customer prediction", icon=":material/person_search:")
with right:
    st.markdown(
        "<div class='status-strip'><strong>Evaluation context</strong><br>"
        "Cross-validation and held-out test results are reported separately. They summarize this experiment and do not guarantee future performance.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.page_link("pages/2_Model_Performance.py", label="Review model performance", icon=":material/monitoring:")
