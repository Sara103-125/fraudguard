"""Tests for the cost model, using a tiny example we can check by hand."""

import pytest

from fraudguard.cost import best_threshold, evaluate_threshold, threshold_table

# 2 frauds (amounts 100 and 50) and 3 normal transactions.
Y_TRUE = [1, 1, 0, 0, 0]
SCORES = [0.9, 0.3, 0.6, 0.2, 0.1]
AMOUNTS = [100, 50, 20, 30, 40]


def test_counts_and_cost_at_threshold_half():
    # Flagged: scores >= 0.5 -> transactions 0 (fraud) and 2 (normal).
    result = evaluate_threshold(Y_TRUE, SCORES, AMOUNTS, threshold=0.5, review_cost=5)

    assert result["frauds_caught"] == 1
    assert result["frauds_missed"] == 1  # the 50 fraud
    assert result["false_alarms"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["missed_fraud_cost"] == 50
    assert result["false_alarm_cost"] == 5
    assert result["total_cost"] == 55


def test_threshold_zero_flags_everything():
    result = evaluate_threshold(Y_TRUE, SCORES, AMOUNTS, threshold=0.0, review_cost=5)

    assert result["recall"] == 1.0
    assert result["false_alarms"] == 3
    assert result["total_cost"] == 15  # no missed fraud, 3 reviews x 5


def test_threshold_above_every_score_flags_nothing():
    result = evaluate_threshold(Y_TRUE, SCORES, AMOUNTS, threshold=0.99, review_cost=5)

    assert result["precision"] == 0.0  # no alerts: defined as 0, not a division error
    assert result["recall"] == 0.0
    assert result["total_cost"] == 150  # both frauds missed


@pytest.mark.parametrize(
    "review_cost, expected",
    [
        (5, 0.3),     # cheap reviews: worth flagging down to 0.3 to catch both frauds (cost 5)
        (100, 0.7),   # expensive reviews: only flag the most certain fraud (cost 50)
    ],
)
def test_best_threshold_depends_on_review_cost(review_cost, expected):
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.95]
    assert best_threshold(Y_TRUE, SCORES, AMOUNTS, review_cost, thresholds) == expected


def test_threshold_table_has_one_row_per_threshold():
    table = threshold_table(Y_TRUE, SCORES, AMOUNTS, thresholds=[0.1, 0.5, 0.9])

    assert list(table["threshold"]) == [0.1, 0.5, 0.9]
    assert table["recall"].is_monotonic_decreasing  # higher threshold, fewer frauds caught
