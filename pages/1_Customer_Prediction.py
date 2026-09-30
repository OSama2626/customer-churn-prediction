import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.batch import predict_batch
from src.config import get_model_metadata
from src.explainability import explain_customer
from src.forms import customer_input_form
from src.predictor import load_pipeline, predict_customer
from src.ui import apply_styles, page_header
from src.validation import validate_customer
from v2_pipeline import RAW_FEATURES


apply_styles()
page_header(
    "Prediction",
    "Score a customer profile or a CSV.",
    "The saved V2 pipeline accepts the original customer fields and applies its fitted feature engineering and preprocessing.",
)

model_choice = st.radio(
    "Model",
    ["Gradient Boosting · reference", "XGBoost · detection option"],
    horizontal=True,
    key="prediction_model_choice",
)
model_name = "xgboost" if model_choice.startswith("XGBoost") else "gradient_boosting"
model_metadata = get_model_metadata(model_name)

if model_name == "xgboost":
    metadata_threshold = float(model_metadata["threshold"])
    alternate_thresholds = [value for value in (0.30, 0.50) if value != metadata_threshold]
    threshold_options = [f"{metadata_threshold:.2f} · metadata default"] + [
        f"{value:.2f} · alternate" for value in alternate_thresholds
    ]
    selected_threshold = st.radio(
        "Decision threshold",
        threshold_options,
        horizontal=True,
        key="xgb_threshold_choice",
    )
    threshold = float(selected_threshold.split(" · ", 1)[0])
else:
    threshold = float(model_metadata["threshold"])

st.info(f"Active model: **{model_metadata['model_name']}** · decision threshold: **{threshold:.2f}**")
st.caption("A customer is classified as likely to churn when probability meets or exceeds the threshold. The threshold changes the class decision, not the probability.")

submitted, customer_values = customer_input_form("prediction")
if submitted:
    try:
        result = predict_customer(
            customer_values,
            model_name=model_name,
            threshold=threshold,
            pipeline=load_pipeline(model_name),
        )
        st.session_state["last_prediction"] = {
            "values": customer_values,
            **result,
        }
    except (ValueError, FileNotFoundError) as error:
        st.error(str(error))

prediction = st.session_state.get("last_prediction")
if prediction and prediction["model_name"] == model_name and prediction["threshold"] == threshold:
    st.markdown("<div class='section-kicker'>Prediction result</div>", unsafe_allow_html=True)
    probability, decision, applied_threshold = st.columns(3)
    probability.metric("Churn probability", f"{prediction['probability']:.1%}")
    decision.metric("Classification", "Likely churn" if prediction["prediction"] else "Likely stay")
    applied_threshold.metric("Applied threshold", f"{threshold:.2f}")
    st.progress(prediction["probability"], text="Probability of churn")
    risk = "High" if prediction["prediction"] else "Low"
    st.write(f"Risk level: **{risk}**")

    st.markdown("<div class='section-kicker'>Why this score?</div>", unsafe_allow_html=True)
    try:
        explanation, contributions = explain_customer(
            load_pipeline(model_name), validate_customer(prediction["values"])
        )
        toward, away = st.columns(2)
        positive = contributions[contributions["shap_value"] > 0].nlargest(5, "shap_value")
        negative = contributions[contributions["shap_value"] < 0].nsmallest(5, "shap_value")
        with toward:
            st.caption("Factors pushing toward churn")
            if positive.empty:
                st.caption("No positive feature contributions for this customer.")
            else:
                st.dataframe(
                positive[["readable_feature", "shap_value"]].rename(columns={
                    "readable_feature": "Business factor", "shap_value": "SHAP (log-odds)"
                }),
                hide_index=True,
                width="stretch",
                )
        with away:
            st.caption("Factors pushing toward staying")
            if negative.empty:
                st.caption("No negative feature contributions for this customer.")
            else:
                st.dataframe(
                negative[["readable_feature", "shap_value"]].rename(columns={
                    "readable_feature": "Business factor", "shap_value": "SHAP (log-odds)"
                }),
                hide_index=True,
                width="stretch",
                )
        import shap

        shap.plots.waterfall(explanation[0], max_display=12, show=False)
        figure = plt.gcf()
        st.pyplot(figure, clear_figure=True)
        plt.close(figure)
        st.caption("Positive SHAP values push the model output toward churn; negative values push it toward staying. Values are in log-odds, not probability points, and do not show causation.")
    except (FileNotFoundError, ValueError, ImportError) as error:
        st.warning(f"Local explanation unavailable: {error}")

st.markdown("<div class='section-kicker'>Batch prediction</div>", unsafe_allow_html=True)
st.write("Upload a CSV with the 10 raw customer columns. Additional columns are retained in the downloadable results.")
with st.expander("Required CSV columns"):
    st.code(", ".join(RAW_FEATURES))
st.caption(f"Batch model: {model_metadata['model_name']} · threshold: {threshold:.2f}")
uploaded = st.file_uploader("Customer CSV", type=["csv"], key="batch_csv")
if uploaded is not None:
    try:
        batch_input = pd.read_csv(uploaded)
        batch_results = predict_batch(
            batch_input,
            threshold=threshold,
            model_name=model_name,
        )
        st.dataframe(batch_results, hide_index=True, width="stretch")
        st.download_button(
            "Download predictions CSV",
            batch_results.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv",
        )
    except (ValueError, FileNotFoundError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not process this CSV: {error}")
else:
    st.caption("No CSV uploaded yet. Use the required-column list above to prepare a file.")
