import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

def load_data():
    """Load the original AI4I dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


# --------------------------------------------------
# Preprocess dataset
# --------------------------------------------------

def preprocess_data(df):
    """
    Prepare the dataset for machine learning.

    Returns:
        X_train_scaled
        X_test_scaled
        y_train
        y_test
        scaler
    """

    # ----------------------------------------------
    # 1. Remove identifier columns
    # ----------------------------------------------

    df = df.drop(
        columns=["UDI", "Product ID"]
    )

    # ----------------------------------------------
    # 2. Remove failure-type columns
    # ----------------------------------------------
    # These columns directly describe different
    # types of machine failures and can cause
    # target leakage.

    df = df.drop(
        columns=["TWF", "HDF", "PWF", "OSF", "RNF"]
    )

    # ----------------------------------------------
    # 3. Separate features and target
    # ----------------------------------------------

    X = df.drop(
        columns=["Machine failure"]
    )

    y = df["Machine failure"]

    # ----------------------------------------------
    # 4. Convert categorical feature
    # ----------------------------------------------

    X = pd.get_dummies(
        X,
        columns=["Type"],
        drop_first=True
    )

    # ----------------------------------------------
    # 5. Train-test split
    # ----------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ----------------------------------------------
    # 6. Feature scaling
    # ----------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )


# --------------------------------------------------
# Test preprocessing
# --------------------------------------------------

if __name__ == "__main__":

    df = load_data()

    print("Original dataset shape:")
    print(df.shape)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    ) = preprocess_data(df)

    print("\nPreprocessing completed.")

    print("\nTraining feature shape:")
    print(X_train.shape)

    print("\nTesting feature shape:")
    print(X_test.shape)

    print("\nTraining target shape:")
    print(y_train.shape)

    print("\nTesting target shape:")
    print(y_test.shape)