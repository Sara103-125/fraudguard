"""SHAP explanations for the XGBoost pipeline.

SHAP splits a single prediction into one contribution per feature:
    model output = base value (average prediction) + sum of feature contributions
For XGBoost the output is in log-odds, so positive contributions push the
transaction towards "fraud" and negative ones towards "normal".
"""

import pandas as pd
import shap


def explain(pipeline, X_raw: pd.DataFrame) -> shap.Explanation:
    """SHAP values for each row of X_raw (raw, unscaled transactions)."""
    X_scaled = pipeline["scale"].transform(X_raw)
    explainer = shap.TreeExplainer(pipeline["model"])  # fast, exact for tree models
    explanation = explainer(X_scaled)
    # The model sees scaled Time/Amount, but people should see the original values.
    explanation.data = X_raw[X_scaled.columns].to_numpy()
    return explanation
