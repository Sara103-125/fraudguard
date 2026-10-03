"""Download the Credit Card Fraud Detection dataset into data/raw/creditcard.csv.

Source: OpenML dataset 1597 ("creditcard"), a public mirror of the Kaggle
dataset by the Machine Learning Group of ULB. No account or API key needed.

We download the raw ARFF file directly instead of using sklearn's
fetch_openml, because fetch_openml drops the `Time` column (OpenML marks it
as a "row id" attribute) and we need Time for the EDA.

Usage:
    python scripts/download_data.py

If OpenML is unavailable, download the CSV manually from Kaggle instead
(see the README) and put it at data/raw/creditcard.csv.
"""

import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CSV_PATH = RAW_DIR / "creditcard.csv"
ARFF_PATH = RAW_DIR / "creditcard.arff"
ARFF_URL = "https://www.openml.org/data/v1/download/1673544/creditcard.arff"

COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
EXPECTED_ROWS = 284_807
EXPECTED_FRAUDS = 492


def download_arff() -> None:
    print(f"Downloading {ARFF_URL}\n(about 150 MB, may take a few minutes)...")
    urllib.request.urlretrieve(ARFF_URL, ARFF_PATH)


def arff_to_dataframe(path: Path) -> pd.DataFrame:
    """An ARFF file is a CSV with a header block that ends at the line '@data'."""
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            if line.strip().lower() == "@data":
                break
    df = pd.read_csv(path, skiprows=line_number + 1, header=None, names=COLUMNS)
    df["Class"] = df["Class"].astype(str).str.strip("'").astype(int)  # '0'/'1' -> 0/1
    return df


def validate(df: pd.DataFrame) -> None:
    assert list(df.columns) == COLUMNS, "Unexpected columns"
    assert len(df) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {len(df)}"
    assert df["Class"].sum() == EXPECTED_FRAUDS, f"Expected {EXPECTED_FRAUDS} frauds"
    assert not df.isna().any().any(), "Dataset contains missing values"


def main() -> None:
    if CSV_PATH.exists():
        print(f"Already downloaded: {CSV_PATH}")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download_arff()
    df = arff_to_dataframe(ARFF_PATH)
    validate(df)
    df.to_csv(CSV_PATH, index=False)
    ARFF_PATH.unlink()  # the CSV is all we need
    print(f"Saved {len(df):,} rows ({df['Class'].sum()} frauds) to {CSV_PATH}")


if __name__ == "__main__":
    main()
