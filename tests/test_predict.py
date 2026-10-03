"""Tests for the saved model on real sample transactions (tests/fixtures/sample_transactions.csv).

The fixture holds 5 frauds the model is very sure about and 5 clearly normal
transactions from the test set, so these tests also catch a broken or
accidentally retrained model.
"""

from pathlib import Path

import pandas as pd
import pytest

from fraudguard.config import FEATURES, TARGET
from fraudguard.predict import load_model, predict

FIXTURE = Path(__file__).parent / "fixtures" / "sample_transactions.csv"


@pytest.fixture(scope="module")
def bundle():
    return load_model()


@pytest.fixture
def transactions():
    return pd.read_csv(FIXTURE)


def test_saved_model_bundle_is_complete(bundle):
    assert set(bundle) >= {"pipeline", "threshold", "review_cost", "features"}
    assert 0 < bundle["threshold"] < 1
    assert bundle["features"] == FEATURES


def test_predictions_have_expected_shape_and_range(bundle, transactions):
    result = predict(bundle, transactions)

    assert list(result.columns) == ["fraud_probability", "is_fraud"]
    assert len(result) == len(transactions)
    assert result["fraud_probability"].between(0, 1).all()
    assert set(result["is_fraud"]) <= {0, 1}


def test_known_frauds_and_normals_are_classified_correctly(bundle, transactions):
    result = predict(bundle, transactions)

    assert (result["is_fraud"] == transactions[TARGET]).all()


def test_decision_follows_threshold(bundle, transactions):
    result = predict(bundle, transactions)

    expected = (result["fraud_probability"] >= bundle["threshold"]).astype(int)
    assert (result["is_fraud"] == expected).all()


def test_column_order_and_extra_columns_do_not_matter(bundle, transactions):
    shuffled = transactions[[TARGET] + FEATURES[::-1]].assign(note="extra column")

    pd.testing.assert_frame_equal(predict(bundle, shuffled), predict(bundle, transactions))


def test_missing_column_raises_clear_error(bundle, transactions):
    with pytest.raises(ValueError, match="Amount"):
        predict(bundle, transactions.drop(columns="Amount"))
