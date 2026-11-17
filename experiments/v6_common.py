"""Shared pieces for judging the v6 candidate. Open data only.

The candidate is v5 plus three things, each off by default in the shipped code:
  - extra inputs (features.build_features(extra=True)),
  - three models from different random draws, averaged (model.AveragedModel),
  - a learned smoother over the averaged probabilities (mtah/smoother.py).

The rule, fixed on 10 Nov 2026 before any 10-match result: v6 replaces v5 only if the mean gain on
the 10 open matches is at least 0.015 with at least 8 matches up, no stress row is more than 0.01
below v5, and its lead over the hand-built rules on SkillCorner's working half is not smaller.
"""

import glob
import os
import sys
from typing import Dict, List, Sequence

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from mtah.smoother import Smoother, sample_weights  # noqa: E402

DRAWS = ("v6_d0", "v6_d1", "v6_d2")
MIN_GAIN, MIN_UP, MAX_STRESS_LOSS = 0.015, 8, 0.01


def load(tag: str) -> Dict[str, dict]:
    """Saved held-out probabilities for one run of scripts/train_oof.py, by match."""
    out = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "runs", "oof", tag, "*.npz"))):
        with np.load(path, allow_pickle=False) as z:
            out[os.path.basename(path)[:-4]] = {k: z[k] for k in z.files}
    if not out:
        raise FileNotFoundError(f"no held-out predictions under runs/oof/{tag}")
    return out


def averaged(tags: Sequence[str]) -> Dict[str, dict]:
    """Folds whose probabilities are the mean over several draws on the same test copies."""
    sets = [load(tag) for tag in tags]
    out = {}
    for match_id, first in sets[0].items():
        for other in sets[1:]:
            if match_id not in other:
                raise ValueError(f"{match_id} is missing from one of {list(tags)}")
            if not (np.array_equal(first["labels"], other[match_id]["labels"])
                    and np.array_equal(first["frames"], other[match_id]["frames"])):
                raise ValueError(f"{match_id}: the draws were not scored on the same test copy")
        fold = dict(first)
        fold["prob"] = sum(s[match_id]["prob"] for s in sets) / float(len(sets))
        out[match_id] = fold
    return out


def smoother_set(fold: dict):
    """One match's held-out probabilities as a training set for the smoother."""
    labels = fold["labels"]
    return fold["prob"], fold["half"], labels[fold["frames"]], sample_weights(labels, fold["frames"])


def fit_smoother(folds: Dict[str, dict], ids: Sequence[str]) -> Smoother:
    smoother = Smoother()
    smoother.fit([smoother_set(folds[i]) for i in ids])
    return smoother


def smoothed(folds: Dict[str, dict]) -> Dict[str, dict]:
    """Each match smoothed by a model fitted on the other matches' held-out probabilities.

    The first model's outputs on those other matches came from models that had seen this match, the
    same small leak as the decoder choice has. A clean version would train a first model for every
    pair of matches.
    """
    ids: List[str] = sorted(folds)
    out = {}
    for held in ids:
        smoother = fit_smoother(folds, [i for i in ids if i != held])
        fold = dict(folds[held])
        fold["prob"] = smoother.probabilities(folds[held]["prob"], folds[held]["half"])
        out[held] = fold
    return out
