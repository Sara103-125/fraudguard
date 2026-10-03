"""Page 1: how do the four models compare on the test set?"""

import matplotlib.pyplot as plt
import streamlit as st

from app_pages.shared import MODEL_NAMES, get_model, get_test_predictions
from fraudguard.plotting import plot_confusion_matrix, plot_pr_curves

st.title("📊 Model performance")
st.write(
    "All four models were trained on 80% of the data and evaluated on the remaining 20% "
    "(56,962 transactions, of which **98 are fraud**)."
)

data = get_test_predictions()
results, metrics = data["results"], data["metrics"]
threshold = get_model()["threshold"]

# --- Metrics table ---------------------------------------------------------
st.subheader("Metrics on the test set")
table = metrics.rename(columns={
    "precision": "Precision", "recall": "Recall", "f1": "F1",
    "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "accuracy": "Accuracy",
})
st.dataframe(
    table.style.format("{:.3f}").highlight_max(
        subset=["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"], color="#cde2fb"
    ),
    width="stretch",
)
st.caption(
    "Default decision rule: probability ≥ 0.5 (Isolation Forest uses its own cut-off). "
    "**Accuracy is misleading here:** always answering *not fraud* is 99.8% accurate and catches nothing. "
    "**PR-AUC** is the fairest single number for comparing models on such imbalanced data."
)

# --- Confusion matrix and PR curve -------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Confusion matrix")
    model_name = st.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index("XGBoost"))
    if model_name == "XGBoost":
        use_tuned = st.toggle(f"Use the cost-tuned threshold ({threshold})", value=True)
    else:
        use_tuned = False

    if use_tuned:
        y_pred = (results["score_XGBoost"] >= threshold).astype(int)
        title = f"XGBoost at threshold {threshold}"
    else:
        y_pred = results[f"pred_{model_name}"]
        title = f"{model_name} (default rule)"

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    plot_confusion_matrix(ax, results["Class"], y_pred, title=title)
    st.pyplot(fig)

with right:
    st.subheader("Precision-recall curves")
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_pr_curves(ax, results["Class"], {m: results[f"score_{m}"] for m in MODEL_NAMES})
    st.pyplot(fig)
    st.caption(
        "Each curve shows the trade-off as the threshold moves: catching more frauds (recall, →) "
        "costs precision (↓). A curve closer to the top-right corner is better."
    )
