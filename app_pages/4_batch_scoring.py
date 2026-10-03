"""Page 4: upload a CSV of transactions, get predictions back as a CSV."""

import pandas as pd
import streamlit as st

from app_pages.shared import get_model, get_sample_transactions
from fraudguard.config import FEATURES, TARGET
from fraudguard.predict import predict

MAX_ROWS = 300_000  # the whole original dataset fits

st.title("📁 Batch scoring")
st.write(
    "Upload a CSV with the same columns as the original dataset: "
    "`Time`, `V1` … `V28`, `Amount` (a `Class` column is optional). "
    "You'll get every row back with a fraud probability and a decision."
)

bundle = get_model()

# A small example file, so anyone can try the page without the dataset.
example = get_sample_transactions().sample(50, random_state=0)
st.download_button(
    "⬇️ Download an example CSV (50 test transactions)",
    data=example.to_csv(index=False).encode("utf-8"),
    file_name="example_transactions.csv",
    mime="text/csv",
)

uploaded = st.file_uploader("Upload transactions (CSV)", type="csv")
use_example = st.toggle("No file? Score the 50 example transactions instead")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as error:  # malformed file, wrong encoding, ...
        st.error(f"Could not read the file as CSV: {error}")
        st.stop()
elif use_example:
    df = example.reset_index(drop=True)
else:
    st.stop()

# --- Validate ----------------------------------------------------------------------

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error(f"The file is missing these columns: {', '.join(missing)}")
    st.stop()
if len(df) > MAX_ROWS:
    st.error(f"Please upload at most {MAX_ROWS:,} rows (this file has {len(df):,}).")
    st.stop()
non_numeric = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(df[c])]
if non_numeric or df[FEATURES].isna().any().any():
    st.error("All feature columns must be numeric with no empty cells. "
             f"Problem columns: {', '.join(non_numeric) or 'empty cells found'}")
    st.stop()

# --- Predict -------------------------------------------------------------------
with st.spinner(f"Scoring {len(df):,} transactions..."):
    output = predict(bundle, df).join(df)  # prediction columns first, then the original columns

flagged = int(output["is_fraud"].sum())
c = st.columns(3)
c[0].metric("Transactions", f"{len(output):,}")
c[1].metric("Flagged as fraud", f"{flagged:,}")
c[2].metric("Threshold used", bundle["threshold"])

if TARGET in df.columns:
    actual = df[TARGET] == 1
    caught = int((actual & (output["is_fraud"] == 1)).sum())
    st.info(f"The file contains a `{TARGET}` column: {caught} of {int(actual.sum())} actual frauds were flagged, "
            f"plus {flagged - caught} false alarms.", icon="ℹ️")

st.dataframe(
    output.sort_values("fraud_probability", ascending=False)
    .style.format({"fraud_probability": "{:.4f}"}),
    width="stretch",
    height=400,
)
st.download_button(
    "⬇️ Download predictions (CSV)",
    data=output.to_csv(index=False).encode("utf-8"),
    file_name="fraudguard_predictions.csv",
    mime="text/csv",
    type="primary",
)
