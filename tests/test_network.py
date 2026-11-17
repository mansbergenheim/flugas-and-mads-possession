"""The numpy networks: the arithmetic against slow, obvious versions, and the properties prediction relies on.

Whether the arithmetic matches the trained PyTorch network is checked where the training happens
(the last cell of the notebooks in colab/), and here too if a reference file from that check has
been placed in tests/fixtures.
"""

import glob
import os
import tempfile
import unittest

import numpy as np

from mtah import network as net

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def random_model(kind, n_inputs=12, seeds=2, rng=None):
    rng = rng or np.random.default_rng(0)
    half = n_inputs // 2
    plan = {"source": list(range(half, n_inputs)) + list(range(half)),         # the two halves change places
            "sign": [1.0] * n_inputs, "offset": [0.0] * n_inputs}
    mask = [0, 3, half, half + 3]                                             # closed under the swap
    config = {"kind": kind, "names": [f"f{i}" for i in range(n_inputs)], "swap": plan, "mask_columns": mask,
              "clip": 5.0, "temperature": 1.3, "seeds": seeds}
    n_in = n_inputs + len(mask)

    def w(*shape):
        return rng.normal(0, 0.4, shape)

    all_weights = []
    for _ in range(seeds):
        if kind == "tcn":
            config.update(channels=8, dilations=[1, 2, 4])
            weights = {"inp.weight": w(8, n_in, 1), "inp.bias": w(8), "out.weight": w(3, 8, 1), "out.bias": w(3)}
            for b in range(3):
                for c in ("conv1", "conv2"):
                    weights[f"blocks.{b}.{c}.weight"] = w(8, 8, 3)
                    weights[f"blocks.{b}.{c}.bias"] = w(8)
        else:
            config.update(proj=7, hidden=5, layers=2, window=16, hop=8)
            weights = {"inp.weight": w(7, n_in), "inp.bias": w(7), "out.weight": w(3, 10), "out.bias": w(3)}
            for layer, size in ((0, 7), (1, 10)):
                for suffix in (f"l{layer}", f"l{layer}_reverse"):
                    weights[f"lstm.weight_ih_{suffix}"] = w(20, size)
                    weights[f"lstm.weight_hh_{suffix}"] = w(20, 5)
                    weights[f"lstm.bias_ih_{suffix}"] = w(20)
                    weights[f"lstm.bias_hh_{suffix}"] = w(20)
        all_weights.append(weights)
    return config, rng.normal(0, 1, n_inputs), np.abs(rng.normal(1, 0.2, n_inputs)) + 0.1, all_weights


def in_memory(kind, **kwargs):
    config, mean, std, seeds = random_model(kind, **kwargs)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "model.npz")
        net.save(path, config, mean, std, seeds)
        return net.load(path)


def some_inputs(n=90, n_inputs=12, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 2, (n, n_inputs))
    x[rng.random((n, n_inputs)) < 0.1] = np.nan
    half = np.where(np.arange(n) < n // 2, 0, 1).astype(np.int8)
    return x, half


class Arithmetic(unittest.TestCase):
    def test_convolution_matches_the_slow_version(self):
        rng = np.random.default_rng(2)
        h, weight, bias = rng.normal(size=(40, 4)), rng.normal(size=(6, 4, 3)), rng.normal(size=6)
        for dilation in (1, 2, 4, 8):
            slow = np.tile(bias, (40, 1))
            for t in range(40):
                for i in range(3):
                    source = t + (i - 1) * dilation
                    if 0 <= source < 40:
                        slow[t] += weight[:, :, i] @ h[source]
            np.testing.assert_allclose(net._conv(h, weight, bias, dilation), slow, atol=1e-12)

    def test_lstm_matches_the_slow_version(self):
        rng = np.random.default_rng(3)
        hidden, size, steps = 4, 3, 9
        weights = {"lstm.weight_ih_l0": rng.normal(size=(16, size)), "lstm.weight_hh_l0": rng.normal(size=(16, hidden)),
                   "lstm.bias_ih_l0": rng.normal(size=16), "lstm.bias_hh_l0": rng.normal(size=16)}
        x = rng.normal(size=(2, steps, size))
        for backwards in (False, True):
            slow = np.zeros((2, steps, hidden))
            for window in range(2):
                h, c = np.zeros(hidden), np.zeros(hidden)
                for t in (reversed(range(steps)) if backwards else range(steps)):
                    a = (weights["lstm.weight_ih_l0"] @ x[window, t] + weights["lstm.bias_ih_l0"]
                         + weights["lstm.weight_hh_l0"] @ h + weights["lstm.bias_hh_l0"])
                    gate_in, gate_forget = 1 / (1 + np.exp(-a[0:4])), 1 / (1 + np.exp(-a[4:8]))
                    cell, gate_out = np.tanh(a[8:12]), 1 / (1 + np.exp(-a[12:16]))
                    c = gate_forget * c + gate_in * cell
                    h = gate_out * np.tanh(c)
                    slow[window, t] = h
            np.testing.assert_allclose(net._lstm_direction(x, weights, "l0", backwards), slow, atol=1e-12)

    def test_every_moment_is_read_from_a_window_that_holds_it(self):
        for n in (1, 5, 16, 17, 23, 24, 100, 257):
            starts, which = net.window_plan(n, 16, 8)
            self.assertEqual(len(which), n)
            position = np.arange(n)
            self.assertTrue(np.all(starts[which] <= position))
            self.assertTrue(np.all(position < starts[which] + min(n, 16)))
            if n > 16:
                inside = position - starts[which]
                middle = (position >= 4) & (position < n - 4)
                self.assertTrue(np.all(inside[middle] >= 4) and np.all(inside[middle] <= 11),
                                "away from the ends a moment has at least 4 steps on each side")


class Prediction(unittest.TestCase):
    def test_probabilities_are_well_formed(self):
        x, half = some_inputs()
        for kind in ("tcn", "bilstm"):
            prob = net.probabilities(in_memory(kind), x, half, decimals=None)
            self.assertEqual(prob.shape, (len(x), 3))
            self.assertTrue(np.all(np.isfinite(prob)) and np.all(prob >= 0))
            np.testing.assert_allclose(prob.sum(1), 1.0, atol=1e-12)
            rounded = net.probabilities(in_memory(kind), x, half)
            self.assertTrue(np.array_equal(rounded, np.round(prob, net.DECIMALS)))

    def test_swapping_the_teams_swaps_home_and_away(self):
        x, half = some_inputs()
        for kind in ("tcn", "bilstm"):
            model = in_memory(kind)
            prob = net.probabilities(model, x, half, decimals=None)
            swapped = net.probabilities(model, net.swap(x, model["config"]["swap"]), half, decimals=None)
            np.testing.assert_allclose(swapped, prob[:, (net.A, net.H, net.D)], atol=1e-12)

    def test_one_half_does_not_see_the_other(self):
        x, half = some_inputs()
        other = x.copy()
        other[half == 1] = np.random.default_rng(9).normal(0, 2, other[half == 1].shape)
        for kind in ("tcn", "bilstm"):
            model = in_memory(kind)
            a, b = net.probabilities(model, x, half, decimals=None), net.probabilities(model, other, half, decimals=None)
            self.assertTrue(np.array_equal(a[half == 0], b[half == 0]))
            self.assertFalse(np.allclose(a[half == 1], b[half == 1]))

    def test_missing_inputs_become_zero_and_a_flag(self):
        model = in_memory("tcn")
        x, _ = some_inputs()
        z = net.prepare(x, model)
        self.assertEqual(z.shape[1], x.shape[1] + len(model["config"]["mask_columns"]))
        self.assertTrue(np.all(np.isfinite(z)))
        self.assertTrue(np.all(z[:, :x.shape[1]][np.isnan(x)] == 0.0))
        np.testing.assert_array_equal(z[:, x.shape[1]:], np.isnan(x[:, model["config"]["mask_columns"]]))
        self.assertLessEqual(np.abs(z[:, :x.shape[1]]).max(), model["config"]["clip"])

    def test_a_short_half_is_read_in_one_window(self):
        x, _ = some_inputs(n=10)
        prob = net.probabilities(in_memory("bilstm"), x, np.zeros(10, dtype=np.int8), decimals=None)
        self.assertEqual(prob.shape, (10, 3))


class Reference(unittest.TestCase):
    def test_reproduces_the_trained_network(self):
        """Files written by the last cell of a training notebook: inputs, and what PyTorch answered."""
        files = sorted(glob.glob(os.path.join(FIXTURES, "network_reference_*.npz")))
        if not files:
            self.skipTest("no reference file from a training notebook in tests/fixtures yet")
        for path in files:
            model = net.load(path.replace("network_reference_", "network_model_"))
            with np.load(path, allow_pickle=False) as z:
                prob = net.probabilities(model, z["x"], z["half"], decimals=None)
                self.assertLess(np.abs(prob - z["prob"]).max(), 1e-4, os.path.basename(path))


if __name__ == "__main__":
    unittest.main()
