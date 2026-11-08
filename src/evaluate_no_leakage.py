"""Leakage-controlled validation, feature-scale ablation, and model training."""

import argparse
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parent.parent
PERSONAL_PATH = ROOT / "Data" / "alphabet_data.csv"
KAGGLE_PATH = ROOT / "kaggle_augmented_data.csv"
KAGGLE_IMAGES_PATH = (
    ROOT / "archive (11)" / "asl_alphabet_train" / "asl_alphabet_train"
)
PRODUCTION_OUTPUT = ROOT / "model" / "sign_language_model_combined_splitfirst.p"
FEATURE_COUNT = 63
SPLIT_COUNT = 5
SEED = 42
PERSONAL_REPETITIONS = 5
MODEL_JOBS = 6
ALPHABET_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
def load_dataset(path):
    dataframe = pd.read_csv(path)
    if "label" not in dataframe.columns or dataframe.shape[1] != FEATURE_COUNT + 1:
        raise ValueError(f"Unexpected dataset schema: {path}")
    labels = dataframe["label"].astype(str).str.strip().to_numpy()
    features = dataframe.drop(columns=["label"]).to_numpy(dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite features found in {path}")
    return features, labels


def mirror_group_ids(features):
    """Keep exact duplicate and horizontal-mirror poses in one group."""
    direct = np.rint(
        np.asarray(features, dtype=np.float64) * 100_000
    ).astype(np.int32)
    mirrored = direct.copy()
    mirrored[:, 0::3] *= -1
    different = direct != mirrored
    first_difference = np.argmax(different, axis=1)
    row_indices = np.arange(len(direct))
    use_mirrored = (
        mirrored[row_indices, first_difference]
        < direct[row_indices, first_difference]
    )
    canonical = np.where(use_mirrored[:, None], mirrored, direct)
    return np.unique(canonical, axis=0, return_inverse=True)[1]


def grouped_folds(features, labels, groups):
    splitter = StratifiedGroupKFold(
        n_splits=SPLIT_COUNT,
        shuffle=True,
        random_state=SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        folds = list(splitter.split(features, labels, groups=groups))
    for train_indices, test_indices in folds:
        if set(groups[train_indices]).intersection(groups[test_indices]):
            raise RuntimeError("A duplicate/mirror group crossed a fold boundary.")
    return folds


def first_grouped_fold(features, labels, groups):
    splitter = StratifiedGroupKFold(
        n_splits=SPLIT_COUNT,
        shuffle=True,
        random_state=SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        train_indices, test_indices = next(
            splitter.split(features, labels, groups=groups)
        )
    if set(groups[train_indices]).intersection(groups[test_indices]):
        raise RuntimeError("A duplicate/mirror group crossed the split.")
    return train_indices, test_indices


def normalize_by_palm_scale(features):
    landmarks = np.asarray(features, dtype=np.float32).reshape(-1, 21, 3).copy()
    wrist_relative = landmarks - landmarks[:, :1, :]
    scale = np.linalg.norm(wrist_relative[:, 9, :], axis=1)
    invalid = ~np.isfinite(scale) | (scale <= 1e-8)
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} rows have invalid palm scale.")
    wrist_relative /= scale[:, None, None]
    return wrist_relative.reshape(-1, FEATURE_COUNT)


def new_model():
    return RandomForestClassifier(
        n_estimators=100,
        random_state=SEED,
        n_jobs=MODEL_JOBS,
    )


def train_and_score(name, train_x, train_y, test_x, test_y):
    model = new_model()
    started = time.perf_counter()
    model.fit(train_x, train_y)
    fit_seconds = time.perf_counter() - started
    predicted = model.predict(test_x)
    accuracy = accuracy_score(test_y, predicted)
    macro_f1 = f1_score(
        test_y,
        predicted,
        labels=ALPHABET_LABELS if set(test_y).issubset(ALPHABET_LABELS) else None,
        average="macro",
        zero_division=0,
    )
    print(
        f"{name}: train={len(train_y)} test={len(test_y)} "
        f"fit_seconds={fit_seconds:.3f} accuracy={accuracy:.6f} "
        f"macro_f1={macro_f1:.6f}",
        flush=True,
    )
    return accuracy, macro_f1, fit_seconds, model


def summarize_scores(name, scores):
    accuracy = np.asarray([score[0] for score in scores], dtype=np.float64)
    macro_f1 = np.asarray([score[1] for score in scores], dtype=np.float64)
    print(
        f"{name} mean±sample-SD: "
        f"accuracy={accuracy.mean():.6f}±{accuracy.std(ddof=1):.6f} "
        f"macro_f1={macro_f1.mean():.6f}±{macro_f1.std(ddof=1):.6f}",
        flush=True,
    )


def count_kaggle_detection_failures(kaggle_labels):
    counts = {}
    for label in sorted(set(kaggle_labels)):
        folder_name = label.lower() if label.lower() in {"del", "nothing", "space"} else label
        label_dir = KAGGLE_IMAGES_PATH / folder_name
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Kaggle image label directory missing: {label_dir}")
        image_count = sum(
            1
            for path in label_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".jpg"
        )
        successful_orientation_rows = int(np.count_nonzero(kaggle_labels == label))
        attempts = 2 * image_count
        failures = attempts - successful_orientation_rows
        if failures < 0:
            raise ValueError(
                f"More saved rows than two-per-image attempts for class {label}"
            )
        counts[label] = (image_count, successful_orientation_rows, failures)
    print("Kaggle MediaPipe detection misses (two orientations attempted per JPG):")
    for label, (images, rows, failures) in counts.items():
        print(
            f"  {label}: jpg_images={images} successful_orientation_rows={rows} "
            f"failed_orientation_detections={failures}",
            flush=True,
        )
    print(
        "  A per-source-image count of images with at least one/both failures "
        "cannot be reconstructed: the CSV does not retain source image IDs.",
        flush=True,
    )


def repeated_personal_training_rows(features, labels, sources, train_indices):
    personal_indices = train_indices[sources[train_indices] == "personal"]
    kaggle_indices = train_indices[sources[train_indices] == "kaggle"]
    train_x = np.concatenate(
        (
            np.tile(features[personal_indices], (PERSONAL_REPETITIONS, 1)),
            features[kaggle_indices],
        ),
        axis=0,
    )
    train_y = np.concatenate(
        (
            np.tile(labels[personal_indices], PERSONAL_REPETITIONS),
            labels[kaggle_indices],
        )
    )
    return train_x, train_y, personal_indices, kaggle_indices


def evaluate_combined_five_fold(raw_features, labels, sources, scale_features):
    groups = mirror_group_ids(raw_features)
    folds = grouped_folds(raw_features, labels, groups)
    score_rows = []
    for variant, features in (
        ("combined_raw", raw_features),
        ("combined_palm_scale", scale_features),
    ):
        scores = []
        for fold_index, (train_indices, test_indices) in enumerate(folds, start=1):
            train_x, train_y, personal_indices, kaggle_indices = (
                repeated_personal_training_rows(
                    features, labels, sources, train_indices
                )
            )
            accuracy, macro_f1, fit_seconds, _model = train_and_score(
                f"{variant} fold={fold_index} "
                f"train_personal={len(personal_indices)} "
                f"train_kaggle={len(kaggle_indices)}",
                train_x,
                train_y,
                features[test_indices],
                labels[test_indices],
            )
            scores.append((accuracy, macro_f1, fit_seconds))
        summarize_scores(variant, scores)
        score_rows.append((variant, scores))
    return score_rows


def evaluate_cross_dataset(personal_x, personal_y, kaggle_x, kaggle_y):
    all_groups = mirror_group_ids(np.concatenate((personal_x, kaggle_x), axis=0))
    p_groups = all_groups[: len(personal_x)]
    k_groups = all_groups[len(personal_x) :]
    p_fold = first_grouped_fold(personal_x, personal_y, p_groups)
    k_fold = first_grouped_fold(kaggle_x, kaggle_y, k_groups)
    p_train, p_test = p_fold
    k_train, k_test = k_fold

    shared_groups = set(p_groups).intersection(k_groups)
    if shared_groups:
        raise RuntimeError(
            f"Cross-dataset duplicate/mirror overlap found: {len(shared_groups)}"
        )
    print(
        "Cross-dataset evaluation uses first grouped fold from five-fold "
        f"partitions; shared duplicate/mirror groups={len(shared_groups)}",
        flush=True,
    )

    for name, train_x, train_y, test_x, test_y in (
        ("kaggle_to_personal", kaggle_x[k_train], kaggle_y[k_train], personal_x[p_test], personal_y[p_test]),
        ("personal_to_kaggle", personal_x[p_train], personal_y[p_train], kaggle_x[k_test], kaggle_y[k_test]),
    ):
        for variant, transform in (
            ("raw", lambda values: values),
            ("palm_scale", normalize_by_palm_scale),
        ):
            train_and_score(
                f"{name}_{variant}",
                transform(train_x),
                train_y,
                transform(test_x),
                test_y,
            )


def train_split_first_production_model(
    personal_x,
    personal_y,
    kaggle_x,
    kaggle_y,
    overwrite_existing=False,
):
    if PRODUCTION_OUTPUT.exists() and not overwrite_existing:
        raise FileExistsError(
            f"Refusing to overwrite existing model artifact: {PRODUCTION_OUTPUT}"
        )

    features = np.concatenate((personal_x, kaggle_x), axis=0)
    labels = np.concatenate((personal_y, kaggle_y), axis=0)
    sources = np.concatenate(
        (
            np.repeat("personal", len(personal_y)),
            np.repeat("kaggle", len(kaggle_y)),
        )
    )
    groups = mirror_group_ids(features)
    train_indices, test_indices = first_grouped_fold(features, labels, groups)
    train_x, train_y, personal_indices, kaggle_indices = (
        repeated_personal_training_rows(features, labels, sources, train_indices)
    )
    accuracy, macro_f1, fit_seconds, model = train_and_score(
        "production_splitfirst_raw",
        train_x,
        train_y,
        features[test_indices],
        labels[test_indices],
    )

    artifact = {
        "model": model,
        "source": "alphabet_data.csv x5 + kaggle_augmented_data.csv; grouped split first",
        "feature_count": FEATURE_COUNT,
        "classes": sorted(model.classes_.tolist()),
        "evaluation": "first StratifiedGroupKFold fold; duplicates/mirrors grouped",
        "test_accuracy": float(accuracy),
        "test_macro_f1": float(macro_f1),
        "fit_seconds": float(fit_seconds),
        "personal_training_rows_before_oversampling": int(len(personal_indices)),
        "personal_training_repetitions": PERSONAL_REPETITIONS,
        "kaggle_training_rows": int(len(kaggle_indices)),
        "test_rows": int(len(test_indices)),
        "n_estimators": 100,
        "random_state": SEED,
        "n_jobs": MODEL_JOBS,
    }
    PRODUCTION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = PRODUCTION_OUTPUT.with_suffix(
        PRODUCTION_OUTPUT.suffix + ".tmp"
    )
    try:
        with temporary_path.open("wb") as model_file:
            pickle.dump(artifact, model_file)
        os.replace(temporary_path, PRODUCTION_OUTPUT)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    print(
        f"production_artifact={PRODUCTION_OUTPUT} "
        f"bytes={PRODUCTION_OUTPUT.stat().st_size}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-production-only",
        action="store_true",
        help="Retrain the split-first artifact without repeating evaluations.",
    )
    arguments = parser.parse_args()
    personal_all_x, personal_all_y = load_dataset(PERSONAL_PATH)
    kaggle_all_x, kaggle_all_y = load_dataset(KAGGLE_PATH)
    if arguments.train_production_only:
        train_split_first_production_model(
            personal_all_x,
            personal_all_y,
            kaggle_all_x,
            kaggle_all_y,
            overwrite_existing=True,
        )
        return
    count_kaggle_detection_failures(kaggle_all_y)

    personal_x, personal_y = personal_all_x, personal_all_y
    kaggle_x, kaggle_y = kaggle_all_x, kaggle_all_y
    personal_mask = np.isin(personal_y, ALPHABET_LABELS)
    kaggle_mask = np.isin(kaggle_y, ALPHABET_LABELS)
    personal_x, personal_y = personal_x[personal_mask], personal_y[personal_mask]
    kaggle_x, kaggle_y = kaggle_x[kaggle_mask], kaggle_y[kaggle_mask]

    combined_x = np.concatenate((personal_x, kaggle_x), axis=0)
    combined_y = np.concatenate((personal_y, kaggle_y), axis=0)
    combined_sources = np.concatenate(
        (
            np.repeat("personal", len(personal_y)),
            np.repeat("kaggle", len(kaggle_y)),
        )
    )
    scaled_combined_x = normalize_by_palm_scale(combined_x)
    print(
        f"combined_A-Z_rows={len(combined_y)} "
        f"personal_rows={len(personal_y)} kaggle_rows={len(kaggle_y)} "
        f"feature_count={FEATURE_COUNT} folds={SPLIT_COUNT} "
        f"scale=Euclidean_distance_wrist_to_middle_MCP_Landmark_9",
        flush=True,
    )
    evaluate_combined_five_fold(
        combined_x,
        combined_y,
        combined_sources,
        scaled_combined_x,
    )
    evaluate_cross_dataset(personal_x, personal_y, kaggle_x, kaggle_y)

    print("Training split-first 29-class production model.")
    train_split_first_production_model(
        personal_all_x,
        personal_all_y,
        kaggle_all_x,
        kaggle_all_y,
    )


if __name__ == "__main__":
    main()
