"""Train a static sign classifier from the combined landmark dataset."""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "combined_training_data_oversampled.csv"
MODEL_PATH = BASE_DIR / "model" / "sign_language_model_combined.p"
FEATURE_COUNT = 63
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Combined dataset not found: {DATA_PATH}")

    print(f"Loading combined dataset from {DATA_PATH}...")
    dataframe = pd.read_csv(DATA_PATH)
    if "label" not in dataframe.columns:
        raise ValueError(f"Dataset is missing the required 'label' column: {DATA_PATH}")

    X = dataframe.drop(columns=["label"])
    if dataframe["label"].isna().any():
        raise ValueError("Dataset contains missing labels")
    y = dataframe["label"].astype(str).str.strip()
    if X.shape[1] != FEATURE_COUNT:
        raise ValueError(
            f"Expected {FEATURE_COUNT} feature columns, found {X.shape[1]}"
        )
    if (y == "").any():
        raise ValueError("Dataset contains empty labels")

    try:
        X = X.astype(np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Dataset features must be numeric: {error}") from error
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Dataset contains non-finite feature values")

    print(f"Dataset shape: {len(dataframe):,} rows, {X.shape[1]} features")
    print(f"Labels: {y.nunique()} classes")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(
        f"Training samples: {len(X_train):,}; "
        f"test samples: {len(X_test):,} ({TEST_SIZE:.0%})"
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    print("Training RandomForestClassifier...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {accuracy:.2%}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".tmp")
    try:
        with temporary_path.open("wb") as model_file:
            pickle.dump(
                {
                    "model": model,
                    "source": "combined_training_data_oversampled.csv",
                    "feature_count": FEATURE_COUNT,
                    "classes": sorted(model.classes_.tolist()),
                    "test_size": TEST_SIZE,
                    "random_state": RANDOM_STATE,
                    "test_accuracy": float(accuracy),
                },
                model_file,
            )
        os.replace(temporary_path, MODEL_PATH)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    print(f"Trained model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
