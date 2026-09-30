import matplotlib.pyplot as plt
import streamlit as st

from src.config import get_model_metadata
from src.explainability import explain_customer, load_global_importance
from src.forms import customer_input_form
from src.predictor import load_pipeline
from src.ui import apply_styles, page_header
from src.validation import validate_customer


apply_styles()
metadata = get_model_metadata("xgboost")
threshold = float(metadata["threshold"])
page_header(
    "Insights",
    "What influences the XGBoost score?",
    "Global SHAP summarizes a training sample. Local SHAP shows how transformed input factors contribute to one customer's model output.",
)
st.info(f"Active model: **{metadata['model_name']}** · saved decision threshold: **{threshold:.2f}**")

st.markdown("<div class='section-kicker'>Global SHAP · training sample</div>", unsafe_allow_html=True)
try:
    global_importance = load_global_importance()
    if global_importance.empty:
        st.info("No global explanation rows are available.")
    else:
        top = global_importance.head(15).copy()
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.barh(top["readable_feature"][::-1], top["mean_absolute_shap"][::-1], color="#2b765d")
        axis.set_xlabel("Mean absolute SHAP · raw model output (log-odds)")
        axis.set_ylabel("")
        axis.grid(axis="x", alpha=0.18)
        axis.set_axisbelow(True)
        figure.tight_layout()
        st.pyplot(figure, clear_figure=True)
        plt.close(figure)
        st.dataframe(
            top[["readable_feature", "mean_absolute_shap", "mean_shap"]].rename(columns={
                "readable_feature": "Business factor",
                "mean_absolute_shap": "Mean |SHAP| (log-odds)",
                "mean_shap": "Mean signed SHAP (log-odds)",
            }),
            hide_index=True,
            width="stretch",
        )
except (FileNotFoundError, ValueError) as error:
    st.warning(f"Global SHAP summary unavailable: {error}")

st.caption("Positive signed values push the model output toward churn; negative values push it toward staying. Importance shows model association, not causation.")
st.markdown("<div class='section-kicker'>Local explanation · customer profile</div>", unsafe_allow_html=True)
submitted, customer_values = customer_input_form("insights", "Explain this customer")
if submitted:
    try:
        pipeline = load_pipeline("xgboost")
        customer = validate_customer(customer_values)
        explanation, contributions = explain_customer(pipeline, customer)
        probability = float(pipeline.predict_proba(customer)[0, 1])
        prediction = probability >= threshold
        risk = "High" if prediction else "Low"
        st.write(f"Churn probability: **{probability:.1%}** · prediction: **{'Likely churn' if prediction else 'Likely stay'}** · risk: **{risk}**")

        toward, away = st.columns(2)
        with toward:
            st.caption("Factors pushing toward churn")
            st.dataframe(
                contributions[contributions["shap_value"] > 0]
                .nlargest(5, "shap_value")[["readable_feature", "shap_value"]]
                .rename(columns={"readable_feature": "Business factor", "shap_value": "SHAP (log-odds)"}),
                hide_index=True,
                width="stretch",
            )
        with away:
            st.caption("Factors pushing toward staying")
            st.dataframe(
                contributions[contributions["shap_value"] < 0]
                .nsmallest(5, "shap_value")[["readable_feature", "shap_value"]]
                .rename(columns={"readable_feature": "Business factor", "shap_value": "SHAP (log-odds)"}),
                hide_index=True,
                width="stretch",
            )

        import shap

        shap.plots.waterfall(explanation[0], max_display=12, show=False)
        figure = plt.gcf()
        st.pyplot(figure, clear_figure=True)
        plt.close(figure)
        st.caption(f"Classification uses the saved threshold of {threshold:.2f}. SHAP values are log-odds contributions, not probability points or causes.")
    except (FileNotFoundError, ValueError, ImportError) as error:
        st.error(f"Local explanation unavailable: {error}")
