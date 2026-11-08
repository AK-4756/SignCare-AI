"""Compare own and Kaggle landmark models, including mirrored hand poses."""

import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold


BASE_DIR = Path(__file__).resolve().parent
OWN_DATA_PATH = BASE_DIR / "Data" / "alphabet_data.csv"
KAGGLE_DATA_DIR = BASE_DIR / "archive (10)" / "landmarks"
OWN_MODEL_PATH = BASE_DIR / "model" / "sign_language_model.p"
KAGGLE_MODEL_PATH = BASE_DIR / "model" / "sign_language_model_kaggle.p"
LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
FEATURE_COUNT = 63
SPLIT_SEED = 42


def _validate_features(features, source):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != FEATURE_COUNT:
        raise ValueError(
            f"{source} must contain {FEATURE_COUNT} features per sample; "
            f"got shape {features.shape}"
        )
    if not np.isfinite(features).all():
        raise ValueError(f"{source} contains non-finite feature values")
    return features


def load_own_dataset(path=OWN_DATA_PATH):
    dataframe = pd.read_csv(path)
    if "label" not in dataframe.columns:
        raise ValueError(f"Missing 'label' column in {path}")
    labels = dataframe["label"].astype(str).str.strip().str.upper().to_numpy()
    features = _validate_features(
        dataframe.drop(columns=["label"]).to_numpy(),
        str(path),
    )
    if set(labels) != set(LABELS):
        raise ValueError(f"{path} must contain all labels A-Z")
    return features, labels


def load_kaggle_dataset(path=KAGGLE_DATA_DIR):
    root = Path(path)
    features = []
    labels = []

    for label in LABELS:
        label_dir = root / label
        if not label_dir.is_dir():
            raise ValueError(f"Missing Kaggle label folder: {label_dir}")
        for sample_path in sorted(label_dir.glob("*.npy")):
            sample = np.load(sample_path, allow_pickle=False)
            sample = np.asarray(sample, dtype=np.float32)
            if sample.size != FEATURE_COUNT:
                raise ValueError(
                    f"{sample_path} must contain {FEATURE_COUNT} coordinates; "
                    f"got shape {sample.shape}"
                )
            landmarks = sample.reshape(21, 3)
            landmarks = landmarks - landmarks[:1, :]
            features.append(landmarks.reshape(FEATURE_COUNT))
            labels.append(label)

    if not features:
        raise ValueError(f"No Kaggle landmark samples found under {root}")

    return _validate_features(features, str(root)), np.asarray(labels)


def build_pose_groups(features):
    """Keep exact duplicate and horizontal-mirror poses in the same split."""
    groups = []
    for feature_row in features:
        landmarks = feature_row.reshape(21, 3)
        mirrored = landmarks.copy()
        mirrored[:, 0] *= -1
        direct_key = np.round(landmarks, decimals=6).astype("<f4").tobytes()
        mirrored_key = np.round(mirrored, decimals=6).astype("<f4").tobytes()
        canonical_key = min(direct_key, mirrored_key)
        groups.append(hashlib.sha256(canonical_key).hexdigest())
    return np.asarray(groups)


def mirror_horizontal_features(features):
    """Mirror wrist-relative landmarks across the vertical image axis."""
    features = _validate_features(features, "Features to mirror")
    mirrored = features.reshape(-1, 21, 3).copy()
    mirrored[:, :, 0] *= -1
    return mirrored.reshape(-1, FEATURE_COUNT)


def grouped_train_test_split(features, labels):
    groups = build_pose_groups(features)
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SPLIT_SEED,
    )
    train_indices, test_indices = next(
        splitter.split(features, labels, groups=groups)
    )
    if set(groups[train_indices]).intersection(groups[test_indices]):
        raise RuntimeError("Pose-group leakage detected between train and test")
    return train_indices, test_indices, len(np.unique(groups))


def train_comparison_model(features, labels, mirror_augmentation=False):
    train_indices, test_indices, group_count = grouped_train_test_split(
        features,
        labels,
    )
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    if mirror_augmentation:
        train_features = np.concatenate(
            (train_features, mirror_horizontal_features(train_features))
        )
        train_labels = np.concatenate((train_labels, train_labels))

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=SPLIT_SEED,
        n_jobs=-1,
    )
    model.fit(train_features, train_labels)
    return (
        model,
        features[test_indices],
        labels[test_indices],
        group_count,
        len(train_labels),
    )


def predict(model, features):
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        if len(feature_names) != features.shape[1]:
            raise ValueError(
                f"Model expects {len(feature_names)} features, "
                f"but the dataset contains {features.shape[1]}"
            )
        features = pd.DataFrame(features, columns=feature_names)
    return model.predict(features)


def report_metrics(name, model, features, labels):
    predictions = predict(model, features)
    print(f"\n{name}")
    print(f"  Samples: {len(labels)}")
    print(f"  Accuracy: {accuracy_score(labels, predictions):.2%}")
    print(f"  Macro F1: {f1_score(labels, predictions, average='macro'):.4f}")
    print(
        classification_report(
            labels,
            predictions,
            labels=list(LABELS),
            zero_division=0,
            digits=3,
        )
    )


def load_saved_model(path=OWN_MODEL_PATH):
    with Path(path).open("rb") as model_file:
        artifact = pickle.load(model_file)
    model = artifact.get("model") if isinstance(artifact, dict) else artifact
    if not hasattr(model, "predict") or not hasattr(model, "classes_"):
        raise ValueError(f"No compatible classifier found in {path}")
    if set(map(str, model.classes_)) != set(LABELS):
        raise ValueError(f"Saved model at {path} does not contain all A-Z classes")
    return model


def report_mirror_metrics(name, model, features, labels):
    mirrored_features = mirror_horizontal_features(features)
    combined_features = np.concatenate((features, mirrored_features))
    combined_labels = np.concatenate((labels, labels))
    print(f"\n{name}")
    print(f"  Samples per orientation: {len(labels)}")
    print(
        "  Accuracy, original orientation: "
        f"{accuracy_score(labels, predict(model, features)):.2%}"
    )
    print(
        "  Accuracy, mirrored orientation: "
        f"{accuracy_score(labels, predict(model, mirrored_features)):.2%}"
    )
    print(
        "  Macro F1, both orientations combined: "
        f"{f1_score(combined_labels, predict(model, combined_features), average='macro'):.4f}"
    )


def main():
    own_features, own_labels = load_own_dataset()
    kaggle_features, kaggle_labels = load_kaggle_dataset()
    own_saved_model = load_saved_model(OWN_MODEL_PATH)
    kaggle_saved_model = load_saved_model(KAGGLE_MODEL_PATH)

    (
        own_model,
        own_test_features,
        own_test_labels,
        own_groups,
        own_train_count,
    ) = train_comparison_model(
        own_features,
        own_labels,
    )
    (
        kaggle_model,
        kaggle_test_features,
        kaggle_test_labels,
        kaggle_groups,
        kaggle_train_count,
    ) = train_comparison_model(
        kaggle_features,
        kaggle_labels,
        mirror_augmentation=True,
    )

    print("Dataset comparison")
    print(f"  Own dataset: {len(own_labels)} samples, {own_groups} mirror-aware pose groups")
    print(
        f"  Kaggle dataset: {len(kaggle_labels)} samples, "
        f"{kaggle_groups} mirror-aware pose groups"
    )
    print("  Split: 1 of 5 stratified, mirror-aware grouped folds (seed 42)")
    print("  Kaggle x/y/z coordinates are converted to wrist-relative features.")
    print(
        "  Kaggle training mirrors x coordinates only; y, z, and labels stay unchanged."
    )
    print(
        "  Splitting occurs before augmentation so synthetic mirrors of test poses "
        "cannot enter training."
    )
    print(
        f"  Training rows: own={own_train_count}, "
        f"Kaggle after mirror augmentation={kaggle_train_count}"
    )
    print("  Saved model files are only loaded; neither is overwritten.")

    report_mirror_metrics(
        "Saved original own-data model -> Kaggle dataset (external; both orientations)",
        own_saved_model,
        kaggle_features,
        kaggle_labels,
    )
    report_metrics(
        "Fresh own-data model -> held-out own pose groups",
        own_model,
        own_test_features,
        own_test_labels,
    )
    report_mirror_metrics(
        "Fresh mirror-augmented Kaggle model -> held-out Kaggle pose groups",
        kaggle_model,
        kaggle_test_features,
        kaggle_test_labels,
    )
    report_mirror_metrics(
        "Saved mirror-augmented Kaggle model -> own dataset (cross-dataset)",
        kaggle_saved_model,
        own_features,
        own_labels,
    )
    report_mirror_metrics(
        "Fresh own-data model -> Kaggle dataset (cross-dataset)",
        own_model,
        kaggle_features,
        kaggle_labels,
    )
    report_mirror_metrics(
        "Fresh mirror-augmented Kaggle model -> own dataset (cross-dataset)",
        kaggle_model,
        own_features,
        own_labels,
    )

    print(
        "\nInterpretation: random/grouped dataset splits do not prove "
        "unseen-person accuracy when participant IDs are unavailable. "
        "For that claim, evaluate on recordings from people excluded from "
        "training, captured with this app's camera pipeline."
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, pickle.UnpicklingError) as error:
        raise SystemExit(f"Dataset evaluation failed: {error}") from error
