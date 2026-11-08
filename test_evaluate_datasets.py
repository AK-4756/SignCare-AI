import unittest

import numpy as np

from evaluate_datasets import build_pose_groups, mirror_horizontal_features


class DatasetEvaluationTests(unittest.TestCase):
    def test_pose_groups_join_exact_horizontal_mirrors(self):
        landmarks = np.arange(63, dtype=np.float32).reshape(21, 3) / 100
        mirrored = landmarks.copy()
        mirrored[:, 0] *= -1

        groups = build_pose_groups(
            np.asarray([landmarks.reshape(-1), mirrored.reshape(-1)])
        )

        self.assertEqual(groups[0], groups[1])

    def test_horizontal_mirroring_negates_only_landmark_x_coordinates(self):
        features = np.arange(63, dtype=np.float32).reshape(1, 63)

        mirrored = mirror_horizontal_features(features).reshape(21, 3)
        original = features.reshape(21, 3)

        np.testing.assert_array_equal(mirrored[:, 0], -original[:, 0])
        np.testing.assert_array_equal(mirrored[:, 1:], original[:, 1:])

    def test_horizontal_mirroring_rejects_invalid_feature_shape(self):
        with self.assertRaisesRegex(ValueError, "63 features"):
            mirror_horizontal_features(np.zeros((1, 62), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
