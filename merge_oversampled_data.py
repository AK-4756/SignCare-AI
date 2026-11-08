"""Merge Kaggle landmarks with five copies of the personal dataset."""

import os
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PERSONAL_DATA_PATH = BASE_DIR / "Data" / "alphabet_data.csv"
KAGGLE_DATA_PATH = BASE_DIR / "kaggle_augmented_data.csv"
OUTPUT_PATH = BASE_DIR / "combined_training_data_oversampled.csv"
FEATURE_COUNT = 63
PERSONAL_REPETITIONS = 5


def main():
    if not PERSONAL_DATA_PATH.is_file():
        raise FileNotFoundError(f"Personal dataset not found: {PERSONAL_DATA_PATH}")
    if not KAGGLE_DATA_PATH.is_file():
        raise FileNotFoundError(f"Kaggle dataset not found: {KAGGLE_DATA_PATH}")

    print(f"Loading personal dataset: {PERSONAL_DATA_PATH}")
    personal = pd.read_csv(PERSONAL_DATA_PATH)
    print(f"Loading augmented Kaggle dataset: {KAGGLE_DATA_PATH}")
    kaggle = pd.read_csv(KAGGLE_DATA_PATH)

    if "label" not in personal.columns or "label" not in kaggle.columns:
        raise ValueError("Both input datasets must have a 'label' column.")
    if personal.shape[1] != FEATURE_COUNT + 1:
        raise ValueError(
            f"Personal dataset must have {FEATURE_COUNT} features plus label; "
            f"found {personal.shape[1]} columns."
        )
    if kaggle.shape[1] != FEATURE_COUNT + 1:
        raise ValueError(
            f"Kaggle dataset must have {FEATURE_COUNT} features plus label; "
            f"found {kaggle.shape[1]} columns."
        )

    expected_features = list(personal.columns[:-1])
    if list(kaggle.columns[:-1]) != expected_features:
        print("Aligning Kaggle feature headers to the personal dataset order.")
        kaggle.columns = expected_features + ["label"]

    print(f"Personal dataset shape: {personal.shape}")
    print(f"Kaggle dataset shape: {kaggle.shape}")
    print(f"Repeating personal dataset {PERSONAL_REPETITIONS} times...")
    personal_oversampled = pd.concat(
        [personal] * PERSONAL_REPETITIONS,
        ignore_index=True,
    )
    combined = pd.concat(
        [personal_oversampled, kaggle],
        ignore_index=True,
    )

    print("Cleaning non-finite, missing, and empty rows...")
    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined["label"] = combined["label"].astype("string").str.strip()
    combined.loc[combined["label"] == "", "label"] = pd.NA
    for column in expected_features:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    combined.dropna(subset=expected_features + ["label"], inplace=True)

    print(f"Combined output shape: {combined.shape}")
    if combined.empty:
        raise ValueError("No valid rows remain after cleaning the merged dataset.")

    temporary_path = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")
    try:
        combined.to_csv(temporary_path, index=False)
        os.replace(temporary_path, OUTPUT_PATH)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    print(f"Saved oversampled dataset to {OUTPUT_PATH}")
    print(f"Class counts: {combined['label'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
