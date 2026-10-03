"""FraudGuard Streamlit app: entry point.

Run with:  streamlit run streamlit_app.py

This file only sets up the page and the navigation; each page lives in app_pages/.
"""

import streamlit as st

st.set_page_config(page_title="FraudGuard", page_icon="🛡️", layout="wide")

pages = [
    st.Page("app_pages/1_model_performance.py", title="Model performance", icon="📊", default=True),
    st.Page("app_pages/2_threshold_explorer.py", title="Threshold explorer", icon="🎚️"),
    st.Page("app_pages/3_transaction_checker.py", title="Transaction checker", icon="🔍"),
    st.Page("app_pages/4_batch_scoring.py", title="Batch scoring", icon="📁"),
]

with st.sidebar:
    st.markdown("## 🛡️ FraudGuard")
    st.caption(
        "Credit card fraud detection with XGBoost, a cost-based threshold and SHAP explanations. "
        "Trained on 284,807 real (anonymised) European card transactions."
    )

st.navigation(pages).run()
