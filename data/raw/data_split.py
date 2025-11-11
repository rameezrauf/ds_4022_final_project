import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH   = "./data/raw/raw_data.csv"
TRAIN_PATH = "./data/train_NYC_inspection.parquet"
TEST_PATH  = "./data/test_NYC_inspection.parquet"

START_DATE = "2024-01-01"
END_DATE   = "2025-12-31"

DATE_COL = "INSPECTION DATE"

# Columns needed for modeling + evaluation
KEEP_COLS = [
    "CUISINE DESCRIPTION",
    "BORO",
    "ZIPCODE",
    "Latitude",
    "Longitude",
    "INSPECTION DATE",
    "BUILDING",
    "STREET",
    # Needed to build our Pass/Fail label
    "SCORE",
    "GRADE"
]

def main():
    # Load raw data
    df = pd.read_csv(RAW_PATH, low_memory=False)
    print(f"Loaded dataset with {len(df):,} rows.")

    # Parse inspection date
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", format="mixed")

    # Drop rows with invalid dates
    before = len(df)
    df = df.dropna(subset=[DATE_COL])
    print(f"Dropped {before - len(df):,} rows due to invalid inspection dates.")

    # Filter to 2023–2025
    mask = (df[DATE_COL] >= START_DATE) & (df[DATE_COL] <= END_DATE)
    df = df.loc[mask].copy()
    print(f"Rows in selected date range ({START_DATE} to {END_DATE}): {len(df):,}")

    # Keep only the necessary columns
    df = df[KEEP_COLS]
    print(f"Remaining columns after filtering: {list(df.columns)}")

    # Train-test split (80/20)
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        shuffle=True
    )

    # Save as Parquet
    train_df.to_parquet(TRAIN_PATH, index=False)
    test_df.to_parquet(TEST_PATH, index=False)

    print(f"Train set: {len(train_df):,} rows → {TRAIN_PATH}")
    print(f"Test set:  {len(test_df):,} rows → {TEST_PATH}")
    print("Done.")

if __name__ == "__main__":
    main()