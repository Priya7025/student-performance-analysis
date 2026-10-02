import pandas as pd
import numpy as np
from pathlib import Path

def load_data(filepath):
    """Load the raw CSV file into a pandas DataFrame."""
    df = pd.read_csv(filepath)
    return df

def inspect_data(df):
    """Print key diagnostic info about the dataset."""
    print("Shape:", df.shape)
    print("\nColumn Info:")
    print(df.info())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nStatistical summary:")
    print(df.describe())

def explore_categoricals(df):
    """Print unique values and counts for each categorical column."""
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        print(f"\n--- {col} ---")
        print(df[col].value_counts(dropna=False))



def clean_data(df):
    """Clean the dataset: fix invalid values, detect outliers, impute missing values."""
    df = df.copy()  # never modify the original DataFrame in place

    # ---- 1. Fix invalid Exam_Score (should be 0-100) ----
    invalid_scores = ~df["Exam_Score"].between(0, 100)
    print(f"Invalid Exam_Score values found: {invalid_scores.sum()}")
    df.loc[invalid_scores, "Exam_Score"] = np.nan

    # ---- 2. Check Hours_Studied for outliers using IQR ----
    Q1 = df["Hours_Studied"].quantile(0.25)
    Q3 = df["Hours_Studied"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    print(f"Hours_Studied IQR bounds: [{lower}, {upper}]")
    outliers = ~df["Hours_Studied"].between(lower, upper)
    print(f"Hours_Studied outliers found: {outliers.sum()}")
    df.loc[outliers, "Hours_Studied"] = np.nan

    # ---- 3. Impute missing NUMERIC values with median ----
    numeric_cols = ["Hours_Studied", "Exam_Score"]
    for col in numeric_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"{col}: filled {missing} missing value(s) with median = {median_val}")

    # ---- 4. Impute missing CATEGORICAL values with mode ----
    cat_cols = ["Teacher_Quality", "Parental_Education_Level", "Distance_from_Home"]
    for col in cat_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"{col}: filled {missing} missing value(s) with mode = '{mode_val}'")

    return df

if __name__ == "__main__":
    df = load_data("data/StudentPerformanceFactors.csv")
    inspect_data(df)
    explore_categoricals(df)

    df_clean = clean_data(df)
    df_clean.to_csv("data/StudentPerformance_clean.csv", index=False)
    print("\nCleaned data saved to data/StudentPerformance_clean.csv")