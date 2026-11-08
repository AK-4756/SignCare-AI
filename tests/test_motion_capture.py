import asyncio
import base64
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.responses import JSONResponse
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app as backend
from src.motion_recognition import FEATURE_COUNT, SEQUENCE_LENGTH


def make_test_frame():
    image = Image.new("RGB", (64, 64), color=(40, 80, 120))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def make_landmarks():
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[9, 1] = 0.1
    landmarks[4, 0] = 0.2
    return landmarks.reshape(FEATURE_COUNT).tolist()


class FakePredictor:
    def __init__(self, missing_indices=()):
        self.missing_indices = set(missing_indices)
        self.frame_index = 0

    def extract_landmarks(self, frame):
        index = self.frame_index
        self.frame_index += 1
        if index in self.missing_indices:
            return None, False, None, None, None
        return make_landmarks(), True, None, None, None


class MotionCaptureEndpointTests(unittest.TestCase):
    def test_video_capture_saves_ordered_landmark_sequence(self):
        images = [make_test_frame()] * SEQUENCE_LENGTH
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(backend, "predictor", FakePredictor()),
                patch.object(backend, "MOTION_DATA_DIR", temp_dir),
            ):
                result = asyncio.run(
                    backend.save_motion_sample_from_video(
                        {"label": "J", "images": images}
                    )
                )

            self.assertTrue(result["saved"])
            self.assertEqual(result["valid_frames"], SEQUENCE_LENGTH)
            self.assertEqual(result["total_frames"], SEQUENCE_LENGTH)
            sample_path = next((Path(temp_dir) / "J").glob("*.npy"))
            self.assertEqual(
                np.load(sample_path, allow_pickle=False).shape,
                (SEQUENCE_LENGTH, FEATURE_COUNT),
            )

    def test_video_capture_reports_insufficient_hand_coverage(self):
        images = [make_test_frame()] * SEQUENCE_LENGTH
        missing_indices = range(SEQUENCE_LENGTH - 17)
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    backend,
                    "predictor",
                    FakePredictor(missing_indices),
                ),
                patch.object(backend, "MOTION_DATA_DIR", temp_dir),
            ):
                result = asyncio.run(
                    backend.save_motion_sample_from_video(
                        {"label": "Z", "images": images}
                    )
                )

            self.assertIsInstance(result, JSONResponse)
            self.assertEqual(result.status_code, 400)
            self.assertFalse((Path(temp_dir) / "Z").exists())


if __name__ == "__main__":
    unittest.main()
