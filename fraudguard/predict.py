"""Saving, loading and using the final model."""

import joblib
import pandas as pd

from fraudguard.config import FEATURES, MODELS_DIR

MODEL_PATH = MODELS_DIR / "fraudguard_model.joblib"


def save_model(pipeline, threshold: float, review_cost: float, path=MODEL_PATH) -> None:
    """Save the pipeline together with the threshold it must be used with."""
    bundle = {
        "pipeline": pipeline,
        "threshold": threshold,
        "review_cost": review_cost,
        "features": FEATURES,
    }
    joblib.dump(bundle, path, compress=3)


def load_model(path=MODEL_PATH) -> dict:
    return joblib.load(path)


def predict(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Fraud probability and decision for each transaction in X."""
    missing = set(bundle["features"]) - set(X.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    probability = bundle["pipeline"].predict_proba(X[bundle["features"]])[:, 1]
    return pd.DataFrame(
        {
            "fraud_probability": probability,
            "is_fraud": (probability >= bundle["threshold"]).astype(int),
        },
        index=X.index,
    )
