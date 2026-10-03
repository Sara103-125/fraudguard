"""Page 3: score one transaction and explain the decision with SHAP."""

import matplotlib.pyplot as plt
import numpy as np
import shap
import streamlit as st

from app_pages.shared import get_model, get_sample_transactions
from fraudguard.config import FEATURES, TARGET
from fraudguard.explain import explain
from fraudguard.predict import predict

st.title("🔍 Transaction checker")
st.write(
    "Pick a transaction from the test set (the model never saw these during training) and see "
    "its fraud probability, the decision and **why** the model decided that."
)

bundle = get_model()
transactions = get_sample_transactions()
fraud_ids = transactions.index[transactions[TARGET] == 1]
normal_ids = transactions.index[transactions[TARGET] == 0]

# --- Choose a transaction -------------------------------------------------------
if "transaction_id" not in st.session_state:
    st.session_state.transaction_id = int(fraud_ids[0])

rng = np.random.default_rng()
b1, b2, b3 = st.columns([1, 1, 2])
if b1.button("🎲 Random fraud", width="stretch"):
    st.session_state.transaction_id = int(rng.choice(fraud_ids))
if b2.button("🎲 Random normal", width="stretch"):
    st.session_state.transaction_id = int(rng.choice(normal_ids))
b3.selectbox(
    "Or choose by transaction ID",
    options=[int(i) for i in transactions.index],
    key="transaction_id",
    format_func=lambda i: f"#{i} ({'fraud' if transactions.loc[i, TARGET] == 1 else 'normal'})",
)

row = transactions.loc[[st.session_state.transaction_id]]
prediction = predict(bundle, row).iloc[0]
is_fraud_actual = row[TARGET].iloc[0] == 1
flagged = prediction["is_fraud"] == 1

# --- Result -------------------------------------------------------------------
st.divider()
c = st.columns(4)
c[0].metric("Fraud probability", f"{prediction['fraud_probability']:.2%}")
c[1].metric("Decision", "🚨 Flag" if flagged else "✅ Approve",
            help=f"Flag when probability ≥ threshold {bundle['threshold']}")
c[2].metric("Actually was", "Fraud" if is_fraud_actual else "Normal")
c[3].metric("Amount", f"{row['Amount'].iloc[0]:,.2f}")

if flagged and is_fraud_actual:
    st.success("Correct: fraud caught.")
elif not flagged and not is_fraud_actual:
    st.success("Correct: genuine transaction approved.")
elif flagged:
    st.warning("False alarm: a genuine customer's transaction would be blocked and reviewed.")
else:
    st.error("Missed fraud: the bank would lose this amount.")

# --- SHAP explanation ----------------------------------------------------------
st.subheader("Why? SHAP explanation")
explanation = explain(bundle["pipeline"], row[FEATURES])[0]

left, right = st.columns([3, 2])
with left:
    shap.plots.waterfall(explanation, max_display=10, show=False)
    fig = plt.gcf()
    fig.set_size_inches(7, 5)
    st.pyplot(fig, clear_figure=True)
with right:
    contributions = (
        explanation.values if explanation.values.ndim == 1 else explanation.values[:, 0]
    )
    order = np.argsort(-np.abs(contributions))[:5]
    st.markdown("**Top 5 reasons**")
    for i in order:
        name, value, effect = explanation.feature_names[i], explanation.data[i], contributions[i]
        direction = "🔺 towards **fraud**" if effect > 0 else "🔻 towards **normal**"
        st.markdown(f"- `{name}` = {value:,.2f} pushed {direction} ({effect:+.2f})")
    st.caption(
        "Start from the model's average output E[f(X)] (bottom of the chart). Each bar is one feature's push; "
        "they add up to this transaction's score f(x), in log-odds. "
        "**Limitation:** V1–V28 are anonymised by the bank, so we can see *that* V14 was unusual, "
        "but not what V14 means."
    )
