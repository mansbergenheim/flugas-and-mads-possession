"""Two small sequence networks, a TCN and a BiLSTM, run in plain numpy.

The networks are trained elsewhere (colab/), on the open matches only. This module only predicts: it
loads the saved weights and repeats the network's arithmetic, so the submission needs no
deep-learning library. The sums are done in 64-bit and the probabilities rounded to four decimals,
so every computer gives the same numbers.

A network reads the same 198 inputs as the trees (features.build_features(extra=True)), one row
every 0.2 s, and looks a few seconds to either side of each moment. Like the trees it is asked twice,
once with home and away exchanged, and the two answers are averaged.

This file has no imports from the rest of the package at the top, because the training notebooks
carry a copy of it to check that it reproduces the trained network.
"""

import json
from typing import Dict, List, Tuple

import numpy as np

H, A, D = 0, 1, 2
DECIMALS = 4
SEPARATOR = "__"


def save(path: str, config: dict, mean: np.ndarray, std: np.ndarray, seeds: List[Dict[str, np.ndarray]]) -> None:
    """One file: the settings as JSON, the input scaling, and each seed's weights under s<k>__<name>."""
    arrays = {"config": np.array(json.dumps(config)), "mean": np.asarray(mean, dtype=np.float32),
              "std": np.asarray(std, dtype=np.float32)}
    for k, weights in enumerate(seeds):
        for name, value in weights.items():
            arrays[f"s{k}{SEPARATOR}{name}"] = np.asarray(value, dtype=np.float32)
    np.savez_compressed(path, **arrays)


def load(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        config = json.loads(str(z["config"]))
        seeds: List[Dict[str, np.ndarray]] = [{} for _ in range(int(config["seeds"]))]
        for key in z.files:
            if SEPARATOR in key:
                k, name = key.split(SEPARATOR, 1)
                seeds[int(k[1:])][name] = z[key].astype(np.float64)
        model = {"config": config, "mean": z["mean"].astype(np.float64), "std": z["std"].astype(np.float64),
                 "seeds": seeds}
    if any(not weights for weights in seeds):
        raise ValueError(f"{path}: weights are missing for a seed")
    return model


def swap(x: np.ndarray, plan: dict) -> np.ndarray:
    """The inputs as they would be with home and away exchanged (scripts/export_network_pack.py)."""
    return x[:, plan["source"]] * np.asarray(plan["sign"]) + np.asarray(plan["offset"])


def prepare(x: np.ndarray, model: dict) -> np.ndarray:
    """Scale and clip the inputs, put 0 where one is missing, and add a missing flag for each input that can be."""
    config = model["config"]
    x = np.asarray(x, dtype=np.float64)
    z = np.clip((x - model["mean"]) / model["std"], -config["clip"], config["clip"])
    z[np.isnan(x)] = 0.0
    return np.concatenate([z, np.isnan(x[:, config["mask_columns"]]).astype(np.float64)], axis=1)


def window_plan(n: int, window: int, hop: int) -> Tuple[np.ndarray, np.ndarray]:
    """Where the windows over n moments start, and for each moment the window whose middle is nearest."""
    if n <= window:
        return np.zeros(1, dtype=np.int64), np.zeros(n, dtype=np.int64)
    starts = list(range(0, n - window + 1, hop))
    if starts[-1] != n - window:
        starts.append(n - window)
    starts = np.asarray(starts, dtype=np.int64)
    middles = starts + (window - 1) / 2.0
    which = np.searchsorted((middles[:-1] + middles[1:]) / 2.0, np.arange(n), side="left")
    return starts, which.astype(np.int64)


def read_windows(per_window: np.ndarray, starts: np.ndarray, which: np.ndarray) -> np.ndarray:
    """[windows, steps, k] back to [n, k], each moment from the window chosen for it."""
    position = np.arange(len(which))
    return per_window[which, position - starts[which]]


def _relu(x):
    return np.maximum(x, 0.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _conv(h: np.ndarray, weight: np.ndarray, bias: np.ndarray, dilation: int) -> np.ndarray:
    """A same-length convolution over time. h [n, c_in], weight [c_out, c_in, k], zeros beyond the ends."""
    n, k = h.shape[0], weight.shape[2]
    pad = dilation * (k - 1) // 2
    padded = np.pad(h, ((pad, pad), (0, 0)))
    out = np.tile(bias, (n, 1))
    for i in range(k):
        out += padded[i * dilation: i * dilation + n] @ weight[:, :, i].T
    return out


def tcn_logits(z: np.ndarray, weights: Dict[str, np.ndarray], config: dict) -> np.ndarray:
    """One half, [n, inputs] to [n, 3]."""
    h = _relu(_conv(z, weights["inp.weight"], weights["inp.bias"], 1))
    for b, dilation in enumerate(config["dilations"]):
        y = _relu(_conv(h, weights[f"blocks.{b}.conv1.weight"], weights[f"blocks.{b}.conv1.bias"], dilation))
        y = _relu(_conv(y, weights[f"blocks.{b}.conv2.weight"], weights[f"blocks.{b}.conv2.bias"], dilation))
        h = h + y
    return _conv(h, weights["out.weight"], weights["out.bias"], 1)


def _lstm_direction(x: np.ndarray, weights: Dict[str, np.ndarray], suffix: str, backwards: bool) -> np.ndarray:
    """One LSTM layer in one direction. x [windows, steps, inputs] to [windows, steps, hidden]."""
    w_ih, w_hh = weights[f"lstm.weight_ih_{suffix}"], weights[f"lstm.weight_hh_{suffix}"]
    bias = weights[f"lstm.bias_ih_{suffix}"] + weights[f"lstm.bias_hh_{suffix}"]
    windows, steps, hidden = x.shape[0], x.shape[1], w_hh.shape[1]
    h, c = np.zeros((windows, hidden)), np.zeros((windows, hidden))
    out = np.empty((windows, steps, hidden))
    for t in (range(steps - 1, -1, -1) if backwards else range(steps)):
        gates = x[:, t] @ w_ih.T + h @ w_hh.T + bias
        i, f, g, o = (gates[:, k * hidden:(k + 1) * hidden] for k in range(4))
        c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
        h = _sigmoid(o) * np.tanh(c)
        out[:, t] = h
    return out


def bilstm_logits(z: np.ndarray, weights: Dict[str, np.ndarray], config: dict) -> np.ndarray:
    """One half, [n, inputs] to [n, 3], read in overlapping windows so every moment has context on both sides."""
    starts, which = window_plan(len(z), config["window"], config["hop"])
    steps = min(len(z), config["window"])
    h = _relu(z[starts[:, None] + np.arange(steps)] @ weights["inp.weight"].T + weights["inp.bias"])
    for layer in range(config["layers"]):
        h = np.concatenate([_lstm_direction(h, weights, f"l{layer}", False),
                            _lstm_direction(h, weights, f"l{layer}_reverse", True)], axis=2)
    return read_windows(h @ weights["out.weight"].T + weights["out.bias"], starts, which)


LOGITS = {"tcn": tcn_logits, "bilstm": bilstm_logits}


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def one_view(model: dict, x: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Probabilities [n, 3] for the inputs as given, averaged over the seeds. Each half is read on its own."""
    config = model["config"]
    z = prepare(x, model)
    out = np.zeros((len(z), 3))
    for part in np.unique(half):
        rows = np.flatnonzero(half == part)
        if not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
            raise ValueError("a half's rows are not next to each other")
        for weights in model["seeds"]:
            out[rows] += _softmax(LOGITS[config["kind"]](z[rows], weights, config) / config["temperature"])
    return out / len(model["seeds"])


def probabilities(model: dict, x: np.ndarray, half: np.ndarray, decimals: int = DECIMALS) -> np.ndarray:
    """Probabilities of H, A and D at each moment: the mean of the answer as given and with the teams swapped."""
    x = np.asarray(x, dtype=np.float64)
    direct = one_view(model, x, half)
    mirrored = one_view(model, swap(x, model["config"]["swap"]), half)[:, (A, H, D)]
    prob = (direct + mirrored) / 2.0
    return prob if decimals is None else np.round(prob, decimals)


class NetworkModel:
    """A trained network behind the same probabilities(match) call the tree models answer."""

    def __init__(self, path: str, name: str = "network"):
        self.model = load(path)
        self.name = name

    def probabilities(self, match):
        from .features import build_features
        values, names, frames, half = build_features(match, extra=True)
        if names != self.model["config"]["names"]:
            raise ValueError("the inputs differ from the ones the network was trained on")
        return probabilities(self.model, values, half), frames, half
