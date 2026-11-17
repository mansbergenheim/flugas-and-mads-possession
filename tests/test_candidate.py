"""The submitted model: the plain mean of the trees and the TCN, and the guards around the weights file."""

import os
import shutil
import tempfile
import unittest

import numpy as np

from mtah import candidate, network as net
from mtah.features import build_features
from mtah.match import A, D, H
from mtah.model import LearnedModel, TreeConfig
from mtah.provenance import sha256_file
from tests.test_network import FIXTURES, random_model
from tests.test_pipeline import toy_match


def small_trees(n=2):
    models = []
    for seed in range(n):
        model = LearnedModel(trees=TreeConfig(max_iter=5, min_samples_leaf=20, seed=seed), extra=True)
        model.fit([toy_match(seed=seed)])
        models.append(model)
    return models


def small_tcn(names):
    config, mean, std, seeds = random_model("tcn", n_inputs=len(names))
    config["names"] = list(names)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "model.npz")
        net.save(path, config, mean, std, seeds)
        return net.load(path)


class Mix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trees = small_trees()
        cls.model = candidate.Candidate(cls.trees, small_tcn(cls.trees[0].names))
        cls.match = toy_match(seed=7)

    def test_probabilities_are_the_plain_mean_of_the_two_halves(self):
        trees, tcn, frames, half = self.model.parts(self.match)
        prob, frames2, half2 = self.model.probabilities(self.match)
        np.testing.assert_array_equal(prob, (trees + tcn) / 2.0)
        np.testing.assert_array_equal(frames, frames2)
        np.testing.assert_array_equal(half, half2)
        np.testing.assert_allclose(prob.sum(1), 1.0, atol=1e-3)      # the TCN's are rounded to four decimals

    def test_the_trees_half_is_the_mean_of_the_tree_models(self):
        trees, _, _, _ = self.model.parts(self.match)
        alone = [model.probabilities(self.match)[0] for model in self.trees]
        np.testing.assert_allclose(trees, sum(alone) / len(alone), atol=1e-12)

    def test_predict_labels_every_frame_of_each_half(self):
        labels = self.model.predict(self.match)
        self.assertEqual(len(labels), self.match.n_frames)
        for start, stop in self.match.halves:
            self.assertTrue(np.isin(labels[start:stop], (H, A, D)).all())

    def test_refuses_a_network_trained_on_other_inputs(self):
        other = small_tcn(["x"] + list(self.trees[0].names[1:]))
        with self.assertRaises(ValueError):
            candidate.Candidate(self.trees, other)

    def test_refuses_tree_models_with_different_inputs(self):
        plain = LearnedModel(trees=TreeConfig(max_iter=5, min_samples_leaf=20))
        plain.fit([toy_match()])
        with self.assertRaises(ValueError):
            candidate.Candidate([self.trees[0], plain], small_tcn(self.trees[0].names))


class ShippedNetwork(unittest.TestCase):
    def test_the_shipped_file_is_the_pinned_one_and_reads_the_inputs_we_build(self):
        tcn = candidate.load_tcn()
        _, names, _, _ = build_features(toy_match(3000), extra=True)
        self.assertEqual(list(tcn["config"]["names"]), list(names))
        self.assertEqual(tcn["config"]["kind"], "tcn")

    def test_a_changed_file_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "tcn.npz")
            shutil.copy(candidate.TCN_FILE, path)
            with open(path, "ab") as handle:
                handle.write(b"x")
            with self.assertRaises(ValueError):
                candidate.load_tcn(path)
            with self.assertRaises(FileNotFoundError):
                candidate.load_tcn(os.path.join(folder, "missing.npz"))

    def test_the_reference_check_covers_the_shipped_file(self):
        """tests/test_network.py compares numpy with PyTorch on this fixture, so it must be the shipped network."""
        fixture = os.path.join(FIXTURES, "network_model_tcn.npz")
        if not os.path.exists(fixture):
            self.skipTest("no TCN fixture")
        self.assertEqual(sha256_file(fixture), candidate.TCN_SHA256)


if __name__ == "__main__":
    unittest.main()
