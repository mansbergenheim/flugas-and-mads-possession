"""Possession models. Both produce per-sample probabilities that the shared decoder turns into spells.

RuleModel is hand-built with no training, the design other teams have published. It is the reference
any learned model has to beat, and the fallback if training goes wrong.

LearnedModel is gradient-boosted trees trained on degraded open matches. Each training match is also
used with home and away exchanged, and predictions average both views, so the model can't favour
a team by position in the file.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import SPELL_WEIGHT
from .decoder import DecoderConfig, decode_samples, expand
from .features import STEP, build_features, swap_teams
from .match import A, D, H, Match, X


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class RuleModel:
    """Nearest-player evidence for team, low player speed and missing ball for dead."""

    def __init__(self, decoder: DecoderConfig = DecoderConfig(temperature=1.0, turnover=10.0,
                                                              to_dead=6.0, from_dead=4.0)):
        self.decoder = decoder
        self.name = "rule"

    def fit(self, matches: List[Match]) -> None:
        pass

    def probabilities(self, match: Match):
        values, names, frames, half = build_features(match)
        col = {n: values[:, i] for i, n in enumerate(names)}
        vote = np.nan_to_num(col["vote_c1"], nan=0.0)
        memory = np.nan_to_num(col["last_vote"], nan=0.0)
        seen = col["seen_c1"]
        team_evidence = np.where(np.nan_to_num(seen) > 0, vote, 0.3 * memory)
        p_home = _sigmoid(team_evidence / 1.5)

        dead_score = 1.2 * (1.2 - np.nan_to_num(col["all_speed_c3"], nan=1.5))
        dead_score += 1.5 * (0.5 - np.nan_to_num(col["seen_c3"], nan=0.0))
        dead_score += 2.0 * np.nan_to_num(col["off_pitch"], nan=0.0)
        p_dead = _sigmoid(dead_score)
        prob = np.column_stack(((1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead))
        return prob, frames, half

    def predict(self, match: Match) -> np.ndarray:
        prob, frames, half = self.probabilities(match)
        path = decode_samples(prob, half, self.decoder)
        return expand(path, frames, half, match.halves, match.n_frames)


@dataclass(frozen=True)
class TreeConfig:
    max_iter: int = 200
    learning_rate: float = 0.07
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 300
    l2_regularization: float = 1.0
    seed: int = 0


def spell_lengths(labels: np.ndarray) -> np.ndarray:
    """Frames in the unbroken run of one label that each frame belongs to."""
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], edges, [len(labels)]))
    sizes = np.diff(bounds)
    return np.repeat(sizes, sizes)


class LearnedModel:
    def __init__(self, decoder: DecoderConfig = DecoderConfig(), trees: TreeConfig = TreeConfig(),
                 name: str = "learned", spell_weight: float = SPELL_WEIGHT, extra: bool = False):
        self.decoder = decoder
        self.trees = trees
        self.name = name
        self.spell_weight = spell_weight
        self.extra = extra  # the v6 candidate's extra inputs; off reproduces v5 exactly
        self.model = None
        self.names: Optional[List[str]] = None

    def _extra(self) -> bool:
        return bool(getattr(self, "extra", False))  # models saved before v6 have no such field

    def _rows(self, match: Match):
        values, names, frames, _ = build_features(match, extra=self._extra())
        labels = match.labels[frames]
        keep = labels != X
        weight = (spell_lengths(match.labels)[frames] / match.fps).clip(1.0, 60.0) ** -self.spell_weight
        return values[keep], labels[keep], names, weight[keep]

    def fit(self, matches: List[Match]) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier

        xs, ys, ws = [], [], []
        for match in matches:
            for view in (match, swap_teams(match)):
                x, y, self.names, w = self._rows(view)
                xs.append(x)
                ys.append(y)
                ws.append(w)
        self.model = HistGradientBoostingClassifier(
            max_iter=self.trees.max_iter, learning_rate=self.trees.learning_rate,
            max_leaf_nodes=self.trees.max_leaf_nodes, min_samples_leaf=self.trees.min_samples_leaf,
            l2_regularization=self.trees.l2_regularization, early_stopping=False,
            random_state=self.trees.seed)
        self.model.fit(np.vstack(xs), np.concatenate(ys), sample_weight=np.concatenate(ws))

    def views(self, match: Match):
        """(values, values with the teams swapped, frames, half): what probabilities_from needs."""
        values, names, frames, half = build_features(match, extra=self._extra())
        if names != self.names:
            raise ValueError("feature names differ from training")
        swapped, _, _, _ = build_features(swap_teams(match), extra=self._extra())
        return values, swapped, frames, half

    def probabilities_from(self, values: np.ndarray, swapped: np.ndarray) -> np.ndarray:
        direct = self.model.predict_proba(values)
        mirrored = self.model.predict_proba(swapped)[:, (A, H, D)]
        return (direct + mirrored) / 2.0

    def probabilities(self, match: Match):
        values, swapped, frames, half = self.views(match)
        return self.probabilities_from(values, swapped), frames, half

    def predict(self, match: Match) -> np.ndarray:
        prob, frames, half = self.probabilities(match)
        path = decode_samples(prob, half, self.decoder)
        return expand(path, frames, half, match.halves, match.n_frames)


class AveragedModel:
    """Several LearnedModels built from different random draws, with their probabilities averaged.

    One model is one draw of the training copies and one tree seed, and draws differ by more than
    most changes we test. The average is steadier than any single draw. Part of the v6 candidate.
    """

    def __init__(self, models: List[LearnedModel], decoder: DecoderConfig = DecoderConfig(),
                 name: str = "averaged"):
        if not models:
            raise ValueError("need at least one model")
        self.models = list(models)
        self.decoder = decoder
        self.name = name

    def probabilities(self, match: Match):
        values, swapped, frames, half = self.models[0].views(match)
        total = None
        for model in self.models:
            if model.names != self.models[0].names:
                raise ValueError("the averaged models use different inputs")
            prob = model.probabilities_from(values, swapped)
            total = prob if total is None else total + prob
        return total / float(len(self.models)), frames, half

    def predict(self, match: Match) -> np.ndarray:
        prob, frames, half = self.probabilities(match)
        path = decode_samples(prob, half, self.decoder)
        return expand(path, frames, half, match.halves, match.n_frames)
