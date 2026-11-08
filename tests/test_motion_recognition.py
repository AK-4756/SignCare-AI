import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.motion_recognition import (
    FEATURE_COUNT,
    MAX_MISSING_MOTION_GAP,
    MIN_VALID_MOTION_FRAMES,
    MIN_SAMPLES_PER_LABEL,
    MOTION_LABELS,
    SEQUENCE_LENGTH,
    interpolate_missing_frames,
    load_motion_dataset,
    normalize_sequence,
    train_motion_model,
)


def make_sequence(frame_count=SEQUENCE_LENGTH):
    landmarks = np.zeros((frame_count, 21, 3), dtype=np.float32)
    landmarks[:, 9, 1] = 0.1
    landmarks[:, 4, 0] = np.linspace(0.0, 0.2, frame_count)
    return landmarks.reshape(frame_count, FEATURE_COUNT)


class MotionRecognitionTests(unittest.TestCase):
    def test_normalize_sequence_resamples_and_scales(self):
        normalized = normalize_sequence(make_sequence(frame_count=5))

        self.assertEqual(normalized.shape, (SEQUENCE_LENGTH, FEATURE_COUNT))
        self.assertTrue(np.isfinite(normalized).all())

    def test_normalize_sequence_accepts_legacy_12_frame_samples(self):
        normalized = normalize_sequence(make_sequence(frame_count=12))

        self.assertEqual(normalized.shape, (24, FEATURE_COUNT))
        self.assertTrue(np.isfinite(normalized).all())

    def test_normalize_sequence_rejects_non_finite_input(self):
        sequence = make_sequence()
        sequence[0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "non-finite"):
            normalize_sequence(sequence)

    def test_interpolate_missing_frames_preserves_sequence_length(self):
        frames = list(make_sequence())
        for index in range(8, 12):
            frames[index] = None

        interpolated, valid_count = interpolate_missing_frames(frames)

        self.assertEqual(interpolated.shape, (SEQUENCE_LENGTH, FEATURE_COUNT))
        self.assertEqual(valid_count, SEQUENCE_LENGTH - 4)
        self.assertTrue(np.isfinite(interpolated).all())

    def test_interpolate_missing_frames_rejects_insufficient_coverage(self):
        frames = list(make_sequence())
        for index in range(SEQUENCE_LENGTH - MIN_VALID_MOTION_FRAMES + 1):
            frames[index] = None

        with self.assertRaisesRegex(ValueError, "at least"):
            interpolate_missing_frames(frames)

    def test_interpolate_missing_frames_rejects_long_tracking_gaps(self):
        frames = list(make_sequence())
        for index in range(MAX_MISSING_MOTION_GAP + 1, 2 * MAX_MISSING_MOTION_GAP + 2):
            frames[index] = None

        with self.assertRaisesRegex(ValueError, "consecutive frames"):
            interpolate_missing_frames(frames)

    def test_normalize_sequence_uses_index_mcp_when_middle_mcp_is_unavailable(self):
        sequence = make_sequence(frame_count=5).reshape(5, 21, 3)
        sequence[:, 9, :] = 0.0
        sequence[:, 5, 0] = 0.1

        normalized = normalize_sequence(sequence.reshape(5, FEATURE_COUNT))

        self.assertEqual(normalized.shape, (SEQUENCE_LENGTH, FEATURE_COUNT))
        self.assertTrue(np.isfinite(normalized).all())

    def test_dataset_loader_reads_labeled_sequence_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for label in MOTION_LABELS:
                label_dir = Path(temp_dir) / label
                label_dir.mkdir()
                np.save(label_dir / "sample.npy", make_sequence())

            features, labels = load_motion_dataset(temp_dir)

        self.assertEqual(features.shape, (3, SEQUENCE_LENGTH * FEATURE_COUNT))
        self.assertEqual(set(labels), set(MOTION_LABELS))

    def test_training_requires_all_three_labels_with_enough_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for label in MOTION_LABELS:
                label_dir = Path(temp_dir) / label
                label_dir.mkdir()
                np.save(label_dir / "sample.npy", make_sequence())

            with self.assertRaisesRegex(
                ValueError,
                f"At least {MIN_SAMPLES_PER_LABEL} samples",
            ):
                train_motion_model(temp_dir, Path(temp_dir) / "model.p")

    def test_training_saves_a_motion_model_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for label in MOTION_LABELS:
                label_dir = Path(temp_dir) / label
                label_dir.mkdir()
                for sample_number in range(MIN_SAMPLES_PER_LABEL):
                    sequence = make_sequence()
                    sequence[:, sample_number % FEATURE_COUNT] += sample_number * 0.001
                    np.save(label_dir / f"{sample_number}.npy", sequence)

            model_path = Path(temp_dir) / "model.p"
            result = train_motion_model(temp_dir, model_path)

            self.assertTrue(model_path.is_file())
            self.assertGreaterEqual(result["accuracy"], 0.0)
            self.assertEqual(
                result["sample_counts"],
                {label: MIN_SAMPLES_PER_LABEL for label in MOTION_LABELS},
            )


if __name__ == "__main__":
    unittest.main()
