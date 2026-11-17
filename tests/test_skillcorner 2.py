"""SkillCorner loader: labels and resampling, on small made-up arrays."""

import unittest

import numpy as np

from mtah.loaders.skillcorner import _resample, source_labels
from mtah.match import A, AWAY, D, H, HOME


class SourceLabels(unittest.TestCase):
    def test_dead_outside_phases_and_team_carried_through_gaps(self):
        group = np.array([-1, HOME, -1, -1, AWAY, -1, -1, -1], dtype=np.int8)
        alive = np.array([0, 1, 1, 1, 1, 1, 0, 1], dtype=bool)
        phase = np.array([-1, HOME, HOME, HOME, AWAY, AWAY, -1, AWAY], dtype=np.int8)
        self.assertEqual(source_labels(group, alive, phase).tolist(), [D, H, H, H, A, A, D, A])

    def test_stretch_starting_without_group_takes_phase_team(self):
        group = np.array([-1, -1, HOME], dtype=np.int8)
        alive = np.ones(3, dtype=bool)
        phase = np.array([AWAY, AWAY, HOME], dtype=np.int8)
        self.assertEqual(source_labels(group, alive, phase).tolist(), [A, A, H])


class Resample(unittest.TestCase):
    def test_linear_and_missing_neighbour_gives_nan(self):
        values = np.array([[0.0, 0.0], [10.0, 0.0], [np.nan, 0.0], [30.0, 0.0]], dtype=np.float32)
        out, nearest = _resample(values, 0, 3)
        self.assertEqual(len(out), 7)  # 3 source steps at 10 fps become 7 frames at 25 Hz
        np.testing.assert_allclose(out[:3, 0], [0.0, 4.0, 8.0], atol=1e-5)
        self.assertTrue(np.isnan(out[3:6, 0]).all())  # 1.2, 1.6, 2.0 to 2.4 touch the missing frame
        self.assertEqual(nearest.tolist(), [0, 0, 1, 1, 2, 2, 2])


if __name__ == "__main__":
    unittest.main()
