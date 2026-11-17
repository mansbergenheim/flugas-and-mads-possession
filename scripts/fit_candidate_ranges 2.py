"""Reference values for scripts/check_candidate.py. Open matches only.

The candidate is the plain average of the trees (the extra inputs, three draws averaged) and the TCN.
Everything here is measured on matches the models had not seen:

  output    the eight figures check_real_match.py uses for v5, for the candidate's own answers on the
            10 open matches (runs/oof/v6_d0 to v6_d2 averaged, and runs/oof/net_tcn)
  members   how often the trees and the TCN name a different class at the same moment: on the 10 open
            matches, and on SkillCorner's working half, where the networks are at their weakest
  labels    how often the candidate's final label differs from v5's, on the same two sets
  window    five-minute windows of the 10 open matches, scored the way the public window is scored:
            the candidate's spellF1 minus v5's

Written to mtah/check_candidate.json. The reserve half of SkillCorner is never loaded.

    python3 scripts/fit_candidate_ranges.py      # about 2 minutes
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from check_real_match import output_figures  # noqa: E402
from mtah import harness, metrics as m  # noqa: E402
from mtah.config import DECODER  # noqa: E402
from mtah.decoder import decode_samples, expand  # noqa: E402
from mtah.degrade import DegradeConfig, load_stats  # noqa: E402
from mtah.match import X  # noqa: E402
from v6_common import DRAWS, averaged, load  # noqa: E402

OUT = os.path.join(ROOT, "mtah", "check_candidate.json")
WINDOW = 7500          # frames: five minutes, the length of the public window


def box(values):
    return [float(v) for v in np.nanpercentile(np.asarray(values, dtype=np.float64), (0, 25, 75, 100))]


def halves_of(fold):
    return [tuple(int(v) for v in row) for row in fold["halves"]]


def labels_of(fold, prob):
    return expand(decode_samples(prob, fold["half"], DECODER), fold["frames"], fold["half"], halves_of(fold),
                  len(fold["labels"]))


def inside(fold):
    keep = np.zeros(len(fold["labels"]), dtype=bool)
    for start, stop in halves_of(fold):
        keep[start:stop] = True
    return keep


def pair_figures(v5, trees, tcn):
    """Per match: how often the two members differ at a moment, and how often the final label differs from v5's."""
    members, labels = [], []
    for match_id in sorted(trees):
        fold = trees[match_id]
        for other in (v5[match_id], tcn[match_id]):
            if not np.array_equal(other["frames"], fold["frames"]):
                raise ValueError(f"{match_id}: the models were not predicted at the same moments")
        members.append(float((fold["prob"].argmax(1) != tcn[match_id]["prob"].argmax(1)).mean()))
        candidate = labels_of(fold, (fold["prob"] + tcn[match_id]["prob"]) / 2.0)
        keep = inside(fold)
        labels.append(float((candidate[keep] != labels_of(fold, v5[match_id]["prob"])[keep]).mean()))
    return members, labels


def window_gaps(v5, trees, tcn):
    """The candidate's spellF1 minus v5's in every five-minute window, labels inside the window only."""
    gaps, first = [], []
    for match_id in sorted(trees):
        fold = trees[match_id]
        truth = fold["labels"]
        candidate = labels_of(fold, (fold["prob"] + tcn[match_id]["prob"]) / 2.0)
        old = labels_of(fold, v5[match_id]["prob"])
        mine = []
        for start, stop in m.valid_runs(truth):
            for a in range(start, stop - WINDOW + 1, WINDOW):
                part = np.full_like(truth, X)
                part[a:a + WINDOW] = truth[a:a + WINDOW]
                mine.append(m.evaluate(part, candidate)["spell_f1"] - m.evaluate(part, old)["spell_f1"])
        gaps += mine
        first.append(mine[0])
    return np.array(gaps), np.array(first)


if __name__ == "__main__":
    os.chdir(ROOT)
    stats = load_stats()
    v5, trees, tcn = load("v5"), averaged(DRAWS), load("net_tcn")

    rows = []
    for match_id in sorted(trees):
        fold = trees[match_id]
        test = harness.prepared(match_id, DegradeConfig(), harness.TEST_SEED_OFFSET, stats)
        if not np.array_equal(test.labels, fold["labels"]):
            raise ValueError(f"{match_id}: the rebuilt test copy differs from the saved fold")
        seen = np.isfinite(test.ball[fold["frames"], 0])
        rows.append(output_figures((fold["prob"] + tcn[match_id]["prob"]) / 2.0, fold["frames"], fold["half"], seen,
                                   halves_of(fold), len(fold["labels"])))
        print(f"output measured on {match_id}", flush=True)
    output_ref = {key: box([row[key] for row in rows]) for key in rows[0]}

    members_open, labels_open = pair_figures(v5, trees, tcn)
    members_sc, labels_sc = pair_figures(load("v5_sc"), load("trees_sc"), load("net_tcn_sc"))
    gaps, first = window_gaps(v5, trees, tcn)

    print(f"\nmembers differ at a moment: open matches {min(members_open):.3f} to {max(members_open):.3f}, "
          f"SkillCorner {min(members_sc):.3f} to {max(members_sc):.3f}")
    print(f"final label differs from v5's: open matches {min(labels_open):.3f} to {max(labels_open):.3f}, "
          f"SkillCorner {min(labels_sc):.3f} to {max(labels_sc):.3f}")
    print(f"{len(gaps)} five-minute windows: candidate minus v5 mean {gaps.mean():+.3f}, sd {gaps.std():.3f}, "
          f"below v5 in {np.mean(gaps < 0):.0%}, 1 in 20 below {np.percentile(gaps, 5):+.3f}, "
          f"1 in 100 below {np.percentile(gaps, 1):+.3f}, lowest {gaps.min():+.3f}")
    print(f"first window after kickoff, by match: " + " ".join(f"{g:+.2f}" for g in first))

    with open(OUT, "w") as handle:
        json.dump({"made_from": "open matches only: held-out answers in runs/oof (v5, v6_d0 to v6_d2, net_tcn) and "
                                "SkillCorner's working half (v5_sc, trees_sc, net_tcn_sc)",
                   "output": output_ref,
                   "members": {"open": box(members_open), "skillcorner": box(members_sc)},
                   "labels": {"open": box(labels_open), "skillcorner": box(labels_sc)},
                   "window": {"windows": int(len(gaps)), "mean": float(gaps.mean()), "sd": float(gaps.std()),
                              "below_v5": float(np.mean(gaps < 0)), "p05": float(np.percentile(gaps, 5)),
                              "p01": float(np.percentile(gaps, 1)), "lowest": float(gaps.min())}},
                  handle, indent=1)
    print(f"written {os.path.relpath(OUT)}")
