"""Every fixed way of combining the three trained models, against the plain average of the trees and the TCN.

compare_networks.py covered the plain average of every set of members and the two learnt mixes
(weighted, fused). This adds the rules that learn nothing, so that no way of using the three trained
models is left untried:

  average            the mean of the members' probabilities (the candidate is this, for trees + TCN)
  family average     the trees get half, the two networks a quarter each
  geometric          the mean of the log-probabilities, which lets one doubtful member pull the answer down
  the surer one      at each moment, the answer of whichever member is most sure
  vote, moments      each member names a class at each moment, the shares of votes go to the decoder
  vote, labels       each member is decoded alone and each frame gets the label two of the three agree
                     on; where all three differ, the plain average of the three decides
  roles              one source says whether play is live and another which team has the ball: the chance
                     of a dead ball comes from the trees, the TCN or their average, and the rest is split
                     between home and away as another of the three splits it

The five tests and the decoder setting are those of compare_networks.py. The reserve half of
SkillCorner is never loaded. Open data only.

    python3 experiments/combine_rules.py      # about 4 minutes
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import compare_networks as cn  # noqa: E402
from mtah.match import D  # noqa: E402

EPS = 1e-4           # as in the fused format: keeps the log of a zero probability finite
FLOOR = 0.01          # a vote share of nothing is given this much, so the decoder can still overrule it
CANDIDATE = "trees + tcn, average"
OUT = os.path.join("runs", "combine_rules.json")


def average(parts, weights=None):
    weights = weights or [1.0 / len(parts)] * len(parts)
    return sum(w * p for w, p in zip(weights, parts))


def geometric(parts):
    out = np.exp(sum(np.log(p + EPS) for p in parts) / float(len(parts)))
    return out / out.sum(1, keepdims=True)


def surer(parts):
    pick = np.stack([p.max(1) for p in parts]).argmax(0)
    return np.stack(parts)[pick, np.arange(len(pick))]


def vote_moments(parts):
    votes = sum(np.eye(3)[p.argmax(1)] for p in parts) / float(len(parts))
    return votes * (1.0 - 3 * FLOOR) + FLOOR


def vote_labels(fold, parts):
    a, b, c = (cn.predicted(fold, p) for p in parts)
    tie = cn.predicted(fold, average(parts))
    return np.where((a == b) | (a == c), a, np.where(b == c, b, tie))


def roles(dead, team):
    """One source says whether play is live, another which team has the ball while it is.

    dead and team are "trees", "tcn" or "both" (the plain average). The chance of a dead ball comes
    from the first, and what is left is split between home and away as the second splits it.
    """
    def rule(parts):
        source = {"trees": parts[0], "tcn": parts[1], "both": average(parts)}
        d = source[dead][:, D]
        split = source[team][:, :2] / np.maximum(source[team][:, :2].sum(1, keepdims=True), EPS)
        out = np.empty_like(parts[0])
        out[:, :2] = (1.0 - d)[:, None] * split
        out[:, D] = d
        return out
    return rule


TWO, THREE, OTHER = ("trees", "tcn"), ("trees", "tcn", "bilstm"), ("trees", "bilstm")
FORMATS = (
    ("trees", ("trees",), average),
    (CANDIDATE, TWO, average),
    ("trees + tcn, geometric", TWO, geometric),
    ("trees + tcn, the surer one", TWO, surer),
    ("dead ball: trees, team: tcn", TWO, roles("trees", "tcn")),
    ("dead ball: tcn, team: trees", TWO, roles("tcn", "trees")),
    ("dead ball: both, team: tcn", TWO, roles("both", "tcn")),
    ("dead ball: both, team: trees", TWO, roles("both", "trees")),
    ("dead ball: trees, team: both", TWO, roles("trees", "both")),
    ("dead ball: tcn, team: both", TWO, roles("tcn", "both")),
    ("trees + bilstm, average", OTHER, average),
    ("all three, average", THREE, average),
    ("all three, family average", THREE, lambda parts: average(parts, [0.5, 0.25, 0.25])),
    ("all three, geometric", THREE, geometric),
    ("all three, the surer one", THREE, surer),
    ("all three, vote on moments", THREE, vote_moments),
    ("all three, vote on labels", THREE, None),
)


if __name__ == "__main__":
    os.chdir(cn.ROOT)
    record = cn.manifest()
    members = cn.members(False)
    ids = sorted(members["tcn"]["open matches"])
    strict = {i: cn.follow_definition(cn.harness.harmonise(cn.ds.load_open(i)), 4.0) for i in ids}

    scores = {}                                   # scores[test][format] = {match or copy: spellF1}
    for test in cn.TESTS:
        folds = {name: members[name][test] for name in THREE}
        keys = sorted(set.intersection(*(set(f) for f in folds.values())))
        if test in ("open matches", "stricter labels"):
            keys = [k for k in keys if k in ids]
        scores[test] = {name: {} for name, _, _ in FORMATS}
        for key in keys:
            first = folds["trees"][key]
            if any(not np.array_equal(folds[n][key]["frames"], first["frames"]) for n in THREE):
                raise ValueError(f"{key}: the members are not predicted at the same moments")
            truth = strict[key] if test == "stricter labels" else first["labels"]
            for name, who, rule in FORMATS:
                parts = [folds[n][key]["prob"] for n in who]
                labels = vote_labels(first, parts) if rule is None else cn.predicted(first, rule(parts))
                scores[test][name][key] = cn.m.evaluate(truth, labels)["spell_f1"]
        base = scores[test][CANDIDATE]
        print(f"{test}  ({len(base)} {'copies' if test == 'stress table' else 'matches'})")
        print(f"  {'':<34}{'spellF1':>8}{'against the candidate':>23}{'ahead/behind':>14}")
        for name, _, _ in FORMATS:
            gain = np.array([scores[test][name][k] - base[k] for k in base])
            print(f"  {name:<34}{np.mean(list(scores[test][name].values())):>8.3f}{gain.mean():>+23.3f}"
                  f"{(gain > 0).sum():>10}/{(gain < 0).sum():<3}")
        print(flush=True)

    print("against the candidate, test by test")
    print(f"  {'':<34}" + "".join(f"{test:>17}" for test in cn.TESTS) + f"{'tests ahead':>13}")
    summary = {}
    for name, _, _ in FORMATS:
        gains = [float(np.mean([scores[t][name][k] - scores[t][CANDIDATE][k] for k in scores[t][CANDIDATE]]))
                 for t in cn.TESTS]
        summary[name] = dict(zip(cn.TESTS, gains))
        print(f"  {name:<34}" + "".join(f"{g:>+17.3f}" for g in gains) + f"{sum(g > 0 for g in gains):>13}")
    record.update(matches=ids, against_candidate=summary,
                  spell_f1={name: {t: float(np.mean(list(scores[t][name].values()))) for t in cn.TESTS}
                            for name, _, _ in FORMATS})
    cn.save_manifest(OUT, record)
