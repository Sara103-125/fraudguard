"""Loading the dataset."""

import pandas as pd

from fraudguard.config import DATA_PATH


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load the raw transactions CSV (run scripts/download_data.py first)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path)
