import streamlit as st

from src.config import get_model_metadata
from src.ui import apply_styles, page_header


apply_styles()
baseline = get_model_metadata("gradient_boosting")
xgboost = get_model_metadata("xgboost")
page_header(
    "Methodology",
    "How the V2 models work",
    "The application runs saved pipelines. Feature engineering and preprocessing are fitted inside each saved pipeline.",
)

st.markdown("<div class='section-kicker'>Leakage-aware pipeline</div>", unsafe_allow_html=True)
st.write(
    "The data was split before fitting. Derived-feature statistics, imputation, scaling, and categorical encoding "
    "are learned by the pipeline during fitting. Cross-validation therefore fits these steps within each training fold."
)

st.markdown("<div class='section-kicker'>Input and derived features</div>", unsafe_allow_html=True)
st.write(
    "The models accept 10 raw customer fields: credit score, country, gender, age, tenure, account balance, "
    "number of products, credit card, active-member status, and estimated salary. The pipeline derives balance "
    "per product, salary-to-balance ratio, age group, tenure group, and a high-balance flag."
)

st.markdown("<div class='section-kicker'>Thresholds</div>", unsafe_allow_html=True)
st.write(f"Gradient Boosting uses its saved default threshold of **{baseline['threshold']:.2f}**. XGBoost uses its saved default threshold of **{xgboost['threshold']:.2f}**. The prediction page also offers 0.50 as an alternate XGBoost operating threshold.")
st.write("At or above the active threshold, a score is classified as likely churn. Lowering the threshold generally flags more customers and can also create more false alarms. The threshold does not change the score itself.")

st.markdown("<div class='section-kicker'>Limitations</div>", unsafe_allow_html=True)
st.write(
    "The held-out test set was used to compare project variants and is not independent external validation. "
    "Scores have not been calibrated as real-world probabilities. SHAP contributions describe model behavior, "
    "not causal effects. The data is a static sample and future customer populations may differ."
)
