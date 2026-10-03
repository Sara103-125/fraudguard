"""Train/test split and feature scaling."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from fraudguard.config import FEATURES, RANDOM_STATE, TARGET

# V1-V28 are already PCA outputs (centred, similar scale). Only these two are raw.
SCALED_COLUMNS = ["Time", "Amount"]


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Stratified split: train and test keep the same 0.17% fraud rate.

    Without stratify, a random 20% test set could end up with noticeably more or
    fewer frauds by chance, which makes the evaluation noisy.
    """
    return train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=test_size,
        stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )


def make_scaler() -> ColumnTransformer:
    """Scale Time and Amount; pass V1-V28 through unchanged.

    RobustScaler uses the median and inter-quartile range instead of mean and
    standard deviation, so a few huge amounts (up to 25,691) don't squash
    everything else. It is always fitted inside a Pipeline, i.e. on training
    data only, so no information from the test set leaks into it.
    """
    scaler = ColumnTransformer(
        [("robust", RobustScaler(), SCALED_COLUMNS)],
        remainder="passthrough",
        verbose_feature_names_out=False,  # keep plain column names like "V14"
    )
    return scaler.set_output(transform="pandas")  # DataFrame out, so SHAP shows names
