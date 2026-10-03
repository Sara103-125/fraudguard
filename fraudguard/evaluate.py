"""Scoring models and computing metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def fraud_scores(model, X) -> np.ndarray:
    """A 'how suspicious is this?' score per transaction: higher = more likely fraud.

    Classifiers give a probability between 0 and 1. Isolation Forest has no
    probabilities, so we use its anomaly score (flipped so higher = more anomalous).
    ROC-AUC and PR-AUC only care about the ranking, so both kinds of score work.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return -model.score_samples(X)


def default_predictions(model, X) -> np.ndarray:
    """0/1 predictions using each model's own default decision rule.

    Classifiers: probability >= 0.5. Isolation Forest: predict() returns -1 for
    anomalies and 1 for normal points, which we map to fraud=1 / normal=0.
    """
    if hasattr(model, "predict_proba"):
        return (model.predict_proba(X)[:, 1] >= 0.5).astype(int)
    return (model.predict(X) == -1).astype(int)


def compute_metrics(y_true, y_pred, scores) -> dict:
    """All the metrics we report. y_pred = 0/1 decisions, scores = fraud scores."""
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "accuracy": accuracy_score(y_true, y_pred),
    }
