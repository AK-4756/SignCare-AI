"""Train the live static-letter classifier from the Kaggle landmark dataset."""

import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate_datasets import (
    BASE_DIR,
    KAGGLE_DATA_DIR,
    SPLIT_SEED,
    grouped_train_test_split,
    load_kaggle_dataset,
    mirror_horizontal_features,
)


MODEL_PATH = BASE_DIR / "model" / "sign_language_model_kaggle.p"


def train_kaggle_model(output_path=MODEL_PATH):
    features, labels = load_kaggle_dataset()
    train_indices, test_indices, group_count = grouped_train_test_split(
        features,
        labels,
    )
    training_features = np.concatenate(
        (features[train_indices], mirror_horizontal_features(features[train_indices]))
    )
    training_labels = np.concatenate((labels[train_indices], labels[train_indices]))
    evaluation_model = RandomForestClassifier(
        n_estimators=100,
        random_state=SPLIT_SEED,
        n_jobs=-1,
    )
    evaluation_model.fit(training_features, training_labels)
    direct_accuracy = accuracy_score(
        labels[test_indices],
        evaluation_model.predict(features[test_indices]),
    )
    mirrored_accuracy = accuracy_score(
        labels[test_indices],
        evaluation_model.predict(mirror_horizontal_features(features[test_indices])),
    )

    final_training_features = np.concatenate(
        (features, mirror_horizontal_features(features))
    )
    final_training_labels = np.concatenate((labels, labels))
    final_model = RandomForestClassifier(
        n_estimators=100,
        random_state=SPLIT_SEED,
        n_jobs=-1,
    )
    final_model.fit(final_training_features, final_training_labels)

    artifact = {
        "model": final_model,
        "source": "Kaggle landmark dataset",
        "dataset_path": str(KAGGLE_DATA_DIR),
        "sample_count": len(labels),
        "mirror_augmentation": True,
        "pose_group_count": group_count,
        "feature_count": int(features.shape[1]),
        "classes": sorted(set(labels)),
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary_path.open("wb") as model_file:
            pickle.dump(artifact, model_file)
        temporary_path.replace(destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return {
        "direct_accuracy": float(direct_accuracy),
        "mirrored_accuracy": float(mirrored_accuracy),
        "sample_count": len(labels),
        "pose_group_count": group_count,
        "model_path": str(destination),
    }


if __name__ == "__main__":
    result = train_kaggle_model()
    print("Mirror-augmented Kaggle static-letter model training complete.")
    print(f"  Held-out accuracy, original orientation: {result['direct_accuracy']:.2%}")
    print(f"  Held-out accuracy, mirrored orientation: {result['mirrored_accuracy']:.2%}")
    print(f"  Samples used for final model: {result['sample_count']}")
    print(f"  Mirror-aware pose groups: {result['pose_group_count']}")
    print(f"  Model saved to: {result['model_path']}")
