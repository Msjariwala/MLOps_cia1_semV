import pandas as pd
from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Dataset path
DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"


def load_data():
    """Load the original AI4I dataset."""
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset successfully loaded.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


def validate_data(df):
    """Perform basic data validation."""

    print("\n--- DATA VALIDATION ---")

    # Missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Duplicate rows
    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    # Data types
    print("\nData types:")
    print(df.dtypes)

    # Target distribution
    print("\nMachine failure distribution:")
    print(df["Machine failure"].value_counts())

    # Required target values
    print("\nUnique target values:")
    print(df["Machine failure"].unique())


if __name__ == "__main__":

    df = load_data()

    print("\nFirst 5 records:")
    print(df.head())

    validate_data(df)