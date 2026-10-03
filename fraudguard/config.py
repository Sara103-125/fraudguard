"""Project-wide constants: paths, column names and the random seed."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

TARGET = "Class"  # 1 = fraud, 0 = normal
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]  # anonymised by the data provider
FEATURES = ["Time"] + PCA_FEATURES + ["Amount"]  # same order as the original CSV

RANDOM_STATE = 42  # fixed seed so every run gives the same results
