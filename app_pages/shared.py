"""Loading the saved files once and sharing them between pages.

@st.cache_resource keeps the loaded objects in memory, so moving a slider
doesn't reload the model from disk on every interaction.
"""

import joblib
import streamlit as st

from fraudguard.config import MODELS_DIR
from fraudguard.plotting import apply_style
from fraudguard.predict import load_model

MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "Isolation Forest"]

apply_style()  # same chart style as the notebooks


@st.cache_resource
def get_model() -> dict:
    """The final XGBoost pipeline, its threshold and the review cost it was tuned with."""
    return load_model()


@st.cache_resource
def get_test_predictions() -> dict:
    """Test-set labels, amounts and scores for all four models (from notebook 02)."""
    return joblib.load(MODELS_DIR / "test_predictions.joblib")


@st.cache_resource
def get_sample_transactions():
    """All 98 test frauds + 2,000 random normal test transactions (from notebook 03)."""
    return joblib.load(MODELS_DIR / "sample_transactions.joblib")
