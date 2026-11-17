"""The submitted model: three tree models and a TCN, with their probabilities averaged in equal shares.

The trees. Three LearnedModels with the extra inputs, each trained on its own random draw of the
damaged training copies and with its own tree seed. They are trained when the submission is built,
which repeats to the last bit on any machine (scripts/repro_check.py).

The TCN. A small network over the same inputs, trained on Google Colab from the same ten open matches
by colab/train_tcn.ipynb. Training on a GPU does not repeat to the last bit, so the trained weights
ship as a file, models/tcn.npz, and its SHA-256 is pinned below. Prediction from that file is plain
numpy (mtah/network.py), rounded so that every machine gets the same numbers.

The mix. At each moment both halves give probabilities for home, away and dead ball, and the
submission uses their plain mean. Nothing about the mix is fitted. It was chosen on open matches only,
by a rule fixed before the networks had a result: see the README.
"""

import os
from typing import List, Optional, Sequence

import numpy as np

from . import harness, network
from .config import DECODER, TRAIN_COPIES, TREE_DRAWS
from .decoder import DecoderConfig, decode_samples, expand
from .match import Match
from .model import LearnedModel, TreeConfig
from .provenance import sha256_file

TCN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "tcn.npz")
TCN_SHA256 = "b451ff1d19ec15ba1b8692912636474b91cf2371631516d59d39c88ceb374478"


def load_tcn(path: str = TCN_FILE, expected: Optional[str] = TCN_SHA256) -> dict:
    """The shipped network. A file that is not the pinned one is refused, so a stale or damaged copy
    cannot build a submission."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"the TCN's weights are missing: {os.path.relpath(path)}")
    if expected is not None and sha256_file(path) != expected:
        raise ValueError(f"{os.path.relpath(path)} is not the TCN this code was tested with (its SHA-256 differs)")
    return network.load(path)


def train_trees(ids: Sequence[str], stats) -> List[LearnedModel]:
    """Tree model k uses the k-th random draw of the training copies and tree seed k, as in every test."""
    models = []
    for draw in range(TREE_DRAWS):
        model = LearnedModel(trees=TreeConfig(seed=draw), extra=True)
        model.fit(harness.training_matches(list(ids), draw, TRAIN_COPIES, stats))
        models.append(model)
    return models


class Candidate:
    """Tree models and a network behind the probabilities(match) and predict(match) calls of LearnedModel."""

    def __init__(self, trees: List[LearnedModel], tcn: dict, decoder: DecoderConfig = DECODER,
                 name: str = "trees + tcn"):
        if not trees:
            raise ValueError("need at least one tree model")
        if any(model.names != trees[0].names for model in trees):
            raise ValueError("the tree models use different inputs")
        if list(tcn["config"]["names"]) != list(trees[0].names):
            raise ValueError("the trees and the TCN were trained on different inputs")
        self.trees = list(trees)
        self.tcn = tcn
        self.decoder = decoder
        self.name = name

    def parts(self, match: Match):
        """(the trees' probabilities, the TCN's, frames, half). The inputs are built once for both."""
        values, swapped, frames, half = self.trees[0].views(match)
        trees = sum(model.probabilities_from(values, swapped) for model in self.trees) / float(len(self.trees))
        return trees, network.probabilities(self.tcn, values, half), frames, half

    def probabilities(self, match: Match):
        trees, tcn, frames, half = self.parts(match)
        return (trees + tcn) / 2.0, frames, half

    def predict(self, match: Match) -> np.ndarray:
        prob, frames, half = self.probabilities(match)
        path = decode_samples(prob, half, self.decoder)
        return expand(path, frames, half, match.halves, match.n_frames)


def build(ids: Sequence[str], stats, tcn_path: str = TCN_FILE) -> Candidate:
    """The model make_submission.py builds: the shipped TCN, and the trees trained now on these open matches."""
    tcn = load_tcn(tcn_path)
    return Candidate(train_trees(ids, stats), tcn)
