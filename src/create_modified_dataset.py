import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data" / "ai4i2020.csv"
OUTPUT_PATH = BASE_DIR / "data" / "ai4i2020_processed.csv"


# Load original dataset
df = pd.read_csv(INPUT_PATH)

print("Original dataset shape:")
print(df.shape)


# Remove identifier columns
df = df.drop(columns=["UDI", "Product ID"])


# Remove failure-mode columns that can cause target leakage
df = df.drop(columns=["TWF", "HDF", "PWF", "OSF", "RNF"])


# Convert Type into numerical dummy variables
df = pd.get_dummies(
    df,
    columns=["Type"],
    drop_first=True
)


# Save modified dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nModified dataset shape:")
print(df.shape)

print("\nModified dataset saved to:")
print(OUTPUT_PATH)