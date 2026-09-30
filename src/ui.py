from pathlib import Path

import streamlit as st

from src.config import ROOT


def apply_styles():
    css_path = ROOT / "assets" / "app.css"
    if css_path.is_file():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def page_header(section, title, lede):
    st.markdown(f"<div class='eyebrow'>Customer churn / {section}</div>", unsafe_allow_html=True)
    st.title(title)
    st.markdown(f"<div class='page-lede'>{lede}</div>", unsafe_allow_html=True)


def metric_panel(label, value, note):
    st.markdown(
        f"<div class='metric-panel'><div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>",
        unsafe_allow_html=True,
    )