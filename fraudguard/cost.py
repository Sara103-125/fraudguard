"""A simple business cost model for choosing the decision threshold.

Two kinds of mistake, with different costs:
- Missed fraud (false negative): the bank loses the transaction amount.
- False alarm (false positive): a genuine transaction is blocked and reviewed,
  which costs analyst time and annoys the customer. We use a fixed cost.

The model outputs a fraud probability; a transaction is flagged when
probability >= threshold. We pick the threshold with the lowest total cost.
"""

import numpy as np
import pandas as pd

# Assumed cost of reviewing one false alarm (analyst time + customer friction),
# in the same currency units as Amount. This is a business assumption: change it
# and the best threshold changes (try it in the app's Threshold explorer).
REVIEW_COST = 5.0

# Steps of 0.001 below 0.01 (where costs change fast), then steps of 0.01.
DEFAULT_THRESHOLDS = np.round(
    np.concatenate([np.arange(0.001, 0.01, 0.001), np.arange(0.01, 1.00, 0.01)]), 3
)


def evaluate_threshold(y_true, scores, amounts, threshold: float,
                       review_cost: float = REVIEW_COST) -> dict:
    """Outcome and cost of flagging every transaction with score >= threshold."""
    y_true = np.asarray(y_true)
    amounts = np.asarray(amounts)
    flagged = np.asarray(scores) >= threshold

    caught = flagged & (y_true == 1)
    missed = ~flagged & (y_true == 1)
    false_alarms = flagged & (y_true == 0)

    missed_amount = amounts[missed].sum()
    review_total = false_alarms.sum() * review_cost
    return {
        "threshold": threshold,
        "precision": caught.sum() / flagged.sum() if flagged.any() else 0.0,
        "recall": caught.sum() / (y_true == 1).sum(),
        "frauds_caught": int(caught.sum()),
        "frauds_missed": int(missed.sum()),
        "false_alarms": int(false_alarms.sum()),
        "missed_fraud_cost": float(missed_amount),
        "false_alarm_cost": float(review_total),
        "total_cost": float(missed_amount + review_total),
    }


def threshold_table(y_true, scores, amounts, review_cost: float = REVIEW_COST,
                    thresholds=DEFAULT_THRESHOLDS) -> pd.DataFrame:
    """evaluate_threshold() for many thresholds, one row each."""
    return pd.DataFrame(
        [evaluate_threshold(y_true, scores, amounts, t, review_cost) for t in thresholds]
    )


def best_threshold(y_true, scores, amounts, review_cost: float = REVIEW_COST,
                   thresholds=DEFAULT_THRESHOLDS) -> float:
    """The threshold with the lowest total cost (the lowest one wins ties)."""
    table = threshold_table(y_true, scores, amounts, review_cost, thresholds)
    return float(table.loc[table["total_cost"].idxmin(), "threshold"])
