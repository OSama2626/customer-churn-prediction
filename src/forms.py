import streamlit as st

from src.validation import COUNTRIES, GENDERS


def customer_input_form(key_prefix, submit_label="Analyze customer"):
    with st.form(f"{key_prefix}_customer_form"):
        st.markdown("<div class='section-kicker'>Customer profile</div>", unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            credit_score = st.number_input("Credit score", min_value=300, max_value=850, value=619, step=1, key=f"{key_prefix}_credit_score")
            country = st.selectbox("Geography", COUNTRIES, index=0, key=f"{key_prefix}_country")
            gender = st.selectbox("Gender", GENDERS, index=1, key=f"{key_prefix}_gender")
            age = st.number_input("Age", min_value=18, max_value=100, value=42, step=1, key=f"{key_prefix}_age")
            tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=2, step=1, key=f"{key_prefix}_tenure")
        with right:
            balance = st.number_input("Account balance", min_value=0.0, value=0.0, step=1000.0, format="%.2f", key=f"{key_prefix}_balance")
            products_number = st.selectbox("Products", [1, 2, 3, 4], index=0, key=f"{key_prefix}_products")
            credit_card = st.selectbox("Has credit card", ["Yes", "No"], index=0, key=f"{key_prefix}_card")
            active_member = st.selectbox("Active member", ["Yes", "No"], index=0, key=f"{key_prefix}_active")
            estimated_salary = st.number_input("Estimated salary", min_value=0.0, value=101348.88, step=1000.0, format="%.2f", key=f"{key_prefix}_salary")
        submitted = st.form_submit_button(submit_label, width="stretch")

    values = {
        "credit_score": credit_score,
        "country": country,
        "gender": gender,
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "products_number": products_number,
        "credit_card": int(credit_card == "Yes"),
        "active_member": int(active_member == "Yes"),
        "estimated_salary": estimated_salary,
    }
    return submitted, values