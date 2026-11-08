"""Utilities for training and running motion-based ASL letter recognition."""

import math
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


FEATURE_COUNT = 63
SEQUENCE_LENGTH = 24
MOTION_LABELS = ("J", "Z", "OTHER")
MIN_SAMPLES_PER_LABEL = 30
MIN_VALID_MOTION_FRAMES = math.ceil(SEQUENCE_LENGTH * 0.75)
MAX_MISSING_MOTION_GAP = 4


def interpolate_missing_frames(frames):
    """Interpolate brief detection gaps without changing the original time axis."""
    if not isinstance(frames, (list, tuple)) or len(frames) != SEQUENCE_LENGTH:
        raise ValueError(f"Expected exactly {SEQUENCE_LENGTH} captured frames")

    valid_indices = []
    valid_frames = []
    for index, frame in enumerate(frames):
        if frame is None:
            continue
        vector = np.asarray(frame, dtype=np.float32)
        if vector.shape != (FEATURE_COUNT,):
            raise ValueError(
                f"Frame {index} must have {FEATURE_COUNT} features; got {vector.shape}"
            )
        if not np.isfinite(vector).all():
            raise ValueError(f"Frame {index} contains non-finite landmark values")
        valid_indices.append(index)
        valid_frames.append(vector)

    if len(valid_frames) < MIN_VALID_MOTION_FRAMES:
        raise ValueError(
            f"Only {len(valid_frames)}/{SEQUENCE_LENGTH} frames had a valid hand; "
            f"at least {MIN_VALID_MOTION_FRAMES} are required. Improve lighting, "
            "keep the hand inside the frame, and repeat the sign more slowly."
        )

    missing_gap = 0
    longest_missing_gap = 0
    valid_set = set(valid_indices)
    for index in range(SEQUENCE_LENGTH):
        if index in valid_set:
            missing_gap = 0
        else:
            missing_gap += 1
            longest_missing_gap = max(longest_missing_gap, missing_gap)
    if longest_missing_gap > MAX_MISSING_MOTION_GAP:
        raise ValueError(
            f"The hand was not tracked for {longest_missing_gap} consecutive frames. "
            "Keep the hand visible and repeat the sign at a slower, steady pace."
        )

    time_axis = np.arange(SEQUENCE_LENGTH, dtype=np.float32)
    interpolated = np.empty((SEQUENCE_LENGTH, FEATURE_COUNT), dtype=np.float32)
    valid_matrix = np.asarray(valid_frames, dtype=np.float32)
    for feature_index in range(FEATURE_COUNT):
        interpolated[:, feature_index] = np.interp(
            time_axis,
            valid_indices,
            valid_matrix[:, feature_index],
        )
    return interpolated, len(valid_frames)


def normalize_sequence(frames):
    """Resample a landmark sequence and scale each frame by palm length."""
    sequence = np.asarray(frames, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_COUNT:
        raise ValueError(
            f"Expected frames with {FEATURE_COUNT} features, got shape {sequence.shape}"
        )
    if sequence.shape[0] < 2:
        raise ValueError("A motion sequence must contain at least two frames")
    if not np.isfinite(sequence).all():
        raise ValueError("Motion sequence contains non-finite values")

    landmarks = sequence.reshape(-1, 21, 3)
    landmarks = landmarks - landmarks[:, :1, :]
    middle_mcp_lengths = np.linalg.norm(landmarks[:, 9, :], axis=1)
    index_mcp_lengths = np.linalg.norm(landmarks[:, 5, :], axis=1)
    palm_lengths = np.maximum(middle_mcp_lengths, index_mcp_lengths)
    if np.any(palm_lengths < 1e-6):
        raise ValueError("Motion sequence contains an invalid palm-length measurement")
    landmarks = landmarks / palm_lengths[:, np.newaxis, np.newaxis]

    source_positions = np.linspace(0.0, 1.0, num=len(landmarks))
    target_positions = np.linspace(0.0, 1.0, num=SEQUENCE_LENGTH)
    resampled = np.empty((SEQUENCE_LENGTH, FEATURE_COUNT), dtype=np.float32)
    flattened = landmarks.reshape(len(landmarks), FEATURE_COUNT)
    for feature_index in range(FEATURE_COUNT):
        resampled[:, feature_index] = np.interp(
            target_positions,
            source_positions,
            flattened[:, feature_index],
        )
    return resampled


def load_motion_dataset(data_dir):
    """Load labeled sequences from J, Z, and OTHER subdirectories."""
    data_path = Path(data_dir)
    samples = []
    labels = []

    for label in MOTION_LABELS:
        label_dir = data_path / label
        if not label_dir.exists():
            continue
        for sample_path in sorted(label_dir.glob("*.npy")):
            sample = np.load(sample_path, allow_pickle=False)
            samples.append(normalize_sequence(sample).reshape(-1))
            labels.append(label)

    if not samples:
        raise ValueError(f"No motion samples found under {data_path}")

    return np.asarray(samples, dtype=np.float32), np.asarray(labels)


def train_motion_model(data_dir, output_path):
    """Train and save the motion classifier after evaluating a held-out split."""
    features, labels = load_motion_dataset(data_dir)
    label_counts = {label: int(np.count_nonzero(labels == label)) for label in MOTION_LABELS}
    insufficient = {
        label: count
        for label, count in label_counts.items()
        if count < MIN_SAMPLES_PER_LABEL
    }
    if insufficient:
        details = ", ".join(f"{label}={count}" for label, count in insufficient.items())
        raise ValueError(
            "At least "
            f"{MIN_SAMPLES_PER_LABEL} samples are required for each of "
            f"{', '.join(MOTION_LABELS)}; found {details}"
        )

    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    evaluation_model = RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    evaluation_model.fit(train_features, train_labels)
    predictions = evaluation_model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(
        test_labels,
        predictions,
        labels=list(MOTION_LABELS),
        zero_division=0,
    )

    final_model = RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(features, labels)

    model_path = Path(output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as model_file:
        pickle.dump(
            {
                "model": final_model,
                "sequence_length": SEQUENCE_LENGTH,
                "feature_count": FEATURE_COUNT,
                "labels": list(MOTION_LABELS),
                "sample_counts": label_counts,
            },
            model_file,
        )

    return {
        "accuracy": float(accuracy),
        "report": report,
        "sample_counts": label_counts,
        "model_path": str(model_path),
    }
