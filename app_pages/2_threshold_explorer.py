"""Page 2: what happens if we move the decision threshold?"""

import matplotlib.pyplot as plt
import streamlit as st

from app_pages.shared import get_model, get_test_predictions
from fraudguard.cost import DEFAULT_THRESHOLDS, evaluate_threshold, threshold_table
from fraudguard.plotting import FRAUD_COLOR, MUTED_COLOR, NORMAL_COLOR

st.title("🎚️ Threshold explorer")
st.write(
    "XGBoost gives every transaction a fraud probability. The **threshold** decides when we raise an alert. "
    "Move the slider to see the trade-off on the 56,962 test transactions."
)

bundle = get_model()
results = get_test_predictions()["results"]
y_true, scores, amounts = results["Class"], results["score_XGBoost"], results["Amount"]


@st.cache_data
def cached_table(review_cost: float):
    return threshold_table(y_true, scores, amounts, review_cost=review_cost)


# --- Controls -----------------------------------------------------------------
c1, c2 = st.columns([3, 1])
with c1:
    threshold = st.select_slider(
        "Decision threshold (flag if fraud probability ≥ threshold)",
        options=list(DEFAULT_THRESHOLDS),
        value=bundle["threshold"],
    )
with c2:
    review_cost = st.number_input(
        "Cost of one false alarm", min_value=0.0, max_value=500.0,
        value=float(bundle["review_cost"]), step=1.0,
        help="Analyst time plus customer inconvenience when a genuine transaction is blocked.",
    )

result = evaluate_threshold(y_true, scores, amounts, threshold, review_cost)
default = evaluate_threshold(y_true, scores, amounts, 0.5, review_cost)

# --- Live metrics -------------------------------------------------------------
m = st.columns(5)
m[0].metric("Precision", f"{result['precision']:.1%}",
            help="Of the transactions we flagged, how many were really fraud?")
m[1].metric("Recall", f"{result['recall']:.1%}", help="Of all real frauds, how many did we catch?")
m[2].metric("Frauds caught", f"{result['frauds_caught']} / {result['frauds_caught'] + result['frauds_missed']}",
            delta=result["frauds_caught"] - default["frauds_caught"])
m[3].metric("False alarms", f"{result['false_alarms']:,}",
            delta=result["false_alarms"] - default["false_alarms"], delta_color="inverse")
m[4].metric("Estimated cost", f"{result['total_cost']:,.0f}",
            delta=f"{result['total_cost'] - default['total_cost']:,.0f}", delta_color="inverse")
st.caption(
    f"Deltas compare with the default threshold 0.5. Cost = missed fraud amounts "
    f"({result['missed_fraud_cost']:,.0f}) + {result['false_alarms']} false alarms × {review_cost:g} "
    f"({result['false_alarm_cost']:,.0f}). With no model at all, every fraud is missed: "
    f"cost {amounts[y_true == 1].sum():,.0f}."
)

# --- Charts -------------------------------------------------------------------
table = cached_table(review_cost)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(table["threshold"], table["precision"], color=NORMAL_COLOR, label="Precision")
ax1.plot(table["threshold"], table["recall"], color=FRAUD_COLOR, label="Recall")
ax1.set_ylim(0, 1.02)
ax1.set_title("Precision / recall trade-off")
ax1.legend(loc="lower right")

ax2.plot(table["threshold"], table["total_cost"], color=MUTED_COLOR)
ax2.set_title(f"Total cost (false alarm = {review_cost:g})")

for ax in (ax1, ax2):
    ax.set_xscale("log")
    ax.set_xlabel("Threshold (log scale)")
    ax.axvline(threshold, color=FRAUD_COLOR, linestyle="--", linewidth=1)
    ax.set_xticks([0.001, 0.01, 0.1, 0.5, 1], ["0.001", "0.01", "0.1", "0.5", "1"])
fig.tight_layout()
st.pyplot(fig)

st.info(
    f"The saved model uses threshold **{bundle['threshold']}**, chosen on *training* data with a false-alarm "
    f"cost of {bundle['review_cost']:g}. Picking the threshold by looking at this test set would be "
    "cheating: it would fit the noise of only 98 test frauds.",
    icon="ℹ️",
)
