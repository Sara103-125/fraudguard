"""Tests for the train/test split and the scaler (use small fake data, no download needed)."""

import numpy as np
import pandas as pd
import pytest

from fraudguard.config import FEATURES, PCA_FEATURES, TARGET
from fraudguard.preprocessing import make_scaler, split_data


@pytest.fixture
def fake_transactions() -> pd.DataFrame:
    """1,000 random transactions with the real column names, 2% fraud."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(1000, len(PCA_FEATURES))), columns=PCA_FEATURES)
    df["Time"] = rng.uniform(0, 172_800, 1000)
    df["Amount"] = rng.exponential(80, 1000)
    df[TARGET] = 0
    df.loc[:19, TARGET] = 1
    return df


def test_split_keeps_fraud_rate(fake_transactions):
    X_train, X_test, y_train, y_test = split_data(fake_transactions, test_size=0.2)

    assert len(X_test) == 200
    assert y_train.mean() == pytest.approx(0.02)  # stratified: same fraud rate in both parts
    assert y_test.mean() == pytest.approx(0.02)


def test_split_has_no_overlap_and_only_features(fake_transactions):
    X_train, X_test, _, _ = split_data(fake_transactions)

    assert set(X_train.index).isdisjoint(X_test.index)  # no transaction in both sets
    assert list(X_train.columns) == FEATURES  # the label is not a feature


def test_scaler_only_changes_time_and_amount(fake_transactions):
    X = fake_transactions[FEATURES]
    scaled = make_scaler().fit_transform(X)

    pd.testing.assert_frame_equal(scaled[PCA_FEATURES], X[PCA_FEATURES])  # V1-V28 untouched
    # RobustScaler centres on the median: the scaled median is 0.
    assert scaled["Amount"].median() == pytest.approx(0)
    assert scaled["Time"].median() == pytest.approx(0)


def test_scaler_uses_training_statistics_only(fake_transactions):
    X = fake_transactions[FEATURES]
    scaler = make_scaler().fit(X)
    new_data = X.head(5).assign(Amount=1_000_000)  # extreme values must not refit the scaler

    expected = (1_000_000 - X["Amount"].median()) / (
        X["Amount"].quantile(0.75) - X["Amount"].quantile(0.25)
    )
    assert scaler.transform(new_data)["Amount"].iloc[0] == pytest.approx(expected)
