"""How close are the candidate's wrong calls? Open data only, no audio.

The idea being sized: sound (the referee's whistle, the commentary) adds evidence before the
decoder, like Smart Tagging does, and flips close calls. That only pays if many wrong calls are
close. On the 10 open matches held out, with trees + TCN at v5's decoder setting, this asks:

1. Where the wrong calls are, by kind (wrong team, missed dead ball, dead ball where play went on),
   and how far behind the right answer was in the models' own probabilities before the decoder:
   "decoder overruled" means the models had it right and the decoder's smoothing changed it.
   "near a change" means within 1 s of a true change of label, so a timing error.
2. A ceiling: a nudge towards the true answer given to the decoder, only on samples where the
   models are unsure (their top two answers within a margin), at two strengths. Three kinds of
   nudge: about dead ball only (what whistles and stoppage words could say), about the team only
   (what spoken player names could say), and both. The nudge reads the answer key, so this is the
   most such evidence could ever give, not what a detector would give.

    python3 experiments/close_calls.py      # about a minute
"""

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from mtah import metrics as m  # noqa: E402
from mtah.config import DECODER  # noqa: E402
from mtah.decoder import decode_samples, expand  # noqa: E402
from mtah.match import A, D, H  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402
from v6_common import DRAWS, averaged, load  # noqa: E402

FPS = 25
NEAR = FPS                       # frames from a true change that count as "near a change"
BANDS = ((0.1, "behind by under 0.1"), (0.25, "behind by 0.1 to 0.25"), (0.5, "behind by 0.25 to 0.5"),
         (1.01, "behind by over 0.5"))
MARGINS = (0.2, 0.4, 1.01)       # top two answers closer than this counts as unsure; 1.01 = every sample
BONUSES = (1.0, 2.0)             # added to the log probability, so 1.0 multiplies the odds by about 2.7
OUT = os.path.join("runs", "close_calls.json")


def kind_of(truth, pred):
    team = (truth != D) & (pred != D) & (truth != pred)
    missed_dead = (truth == D) & (pred != D)
    false_dead = (truth != D) & (pred == D)
    return {"wrong team": team, "missed dead ball": missed_dead, "dead where play went on": false_dead}


def near_change(labels, frames):
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    if not len(change):
        return np.zeros(len(frames), dtype=bool)
    k = np.clip(np.searchsorted(change, frames), 1, len(change) - 1)
    return np.minimum(np.abs(frames - change[k]), np.abs(frames - change[k - 1])) <= NEAR


def nudge(prob, truth, what, margin, bonus):
    """Log prior towards the truth on unsure samples. what: 'dead', 'team' or 'both'."""
    top2 = np.sort(prob, axis=1)
    unsure = (top2[:, 2] - top2[:, 1]) < margin
    prior = np.zeros_like(prob)
    if what in ("dead", "both"):
        prior[unsure & (truth == D), D] += bonus
        prior[unsure & (truth != D), D] -= bonus
    if what in ("team", "both"):
        for team in (H, A):
            prior[unsure & (truth == team), team] += bonus
    return prior


if __name__ == "__main__":
    os.chdir(ROOT)
    record = manifest(near=NEAR, margins=list(MARGINS), bonuses=list(BONUSES))
    trees, tcn = averaged(DRAWS), load("net_tcn")
    ids = sorted(trees)
    data = {}
    for i in ids:
        f = trees[i]
        prob = (f["prob"] + tcn[i]["prob"]) / 2.0
        halves = [tuple(int(v) for v in row) for row in f["halves"]]
        path = decode_samples(prob, f["half"], DECODER)
        data[i] = dict(f=f, prob=prob, halves=halves, path=path,
                       full=expand(path, f["frames"], f["half"], halves, len(f["labels"])))

    # 1. Where the wrong calls are, counted on the decoder's samples (every 5th frame).
    rows = {}
    totals = {"samples": 0, "wrong": 0}
    for i in ids:
        f, prob, path = data[i]["f"], data[i]["prob"], data[i]["path"]
        truth = f["labels"][f["frames"]].astype(np.int64)
        keep = truth <= D
        truth, pred, p = truth[keep], path[keep].astype(np.int64), prob[keep]
        near = near_change(f["labels"], f["frames"][keep])
        wrong = truth != pred
        totals["samples"] += int(keep.sum())
        totals["wrong"] += int(wrong.sum())
        behind = p.max(axis=1) - p[np.arange(len(truth)), truth]
        overruled = wrong & (p.argmax(axis=1) == truth)
        for kind, mask in kind_of(truth, pred).items():
            cell = rows.setdefault(kind, {"wrong": 0, "near": 0, "overruled": 0, **{b: 0 for _, b in BANDS}})
            cell["wrong"] += int(mask.sum())
            cell["near"] += int((mask & near).sum())
            cell["overruled"] += int((mask & overruled).sum())
            low = 0.0
            for high, band in BANDS:
                cell[band] += int((mask & ~overruled & (behind >= low) & (behind < high)).sum())
                low = high

    print(f"1. the wrong calls of trees + TCN on the 10 open matches held out "
          f"({totals['wrong'] / totals['samples']:.1%} of samples wrong)")
    cols = ["share of wrong", "near a change", "decoder overruled"] + [b for _, b in BANDS]
    print("  " + f"{'':<26}" + "".join(f"{c:>23}" for c in cols))
    for kind, cell in rows.items():
        n = max(cell["wrong"], 1)
        values = [cell["wrong"] / totals["wrong"], cell["near"] / n, cell["overruled"] / n] + [cell[b] / n for _, b in BANDS]
        print("  " + f"{kind:<26}" + "".join(f"{v:>23.0%}" for v in values))
    everything = {k: sum(c[k] for c in rows.values()) for k in rows["wrong team"]}
    n = everything["wrong"]
    values = [1.0, everything["near"] / n, everything["overruled"] / n] + [everything[b] / n for _, b in BANDS]
    print("  " + f"{'all wrong calls':<26}" + "".join(f"{v:>23.0%}" for v in values))
    record["wrong_calls"] = {"totals": totals, "by_kind": rows}

    # 2. Ceiling of a tie-breaker that knows the answer, given only where the models are unsure.
    base = np.array([m.evaluate(data[i]["f"]["labels"], data[i]["full"])["spell_f1"] for i in ids])
    print(f"\n2. a nudge towards the true answer, only where the models are unsure "
          f"(trees + TCN as predicted: {base.mean():.3f})")
    print(f"  {'':<48}{'unsure samples':>15}{'spellF1':>9}{'gain':>9}{'up':>8}")
    record["ceiling"] = {}
    for what, label in (("dead", "dead ball only"), ("team", "team only"), ("both", "both")):
        for margin in MARGINS:
            for bonus in BONUSES:
                scores, unsure_share = [], []
                for i in ids:
                    f, prob = data[i]["f"], data[i]["prob"]
                    truth = f["labels"][f["frames"]].astype(np.int64)
                    prior = nudge(prob, truth, what, margin, bonus)
                    prior[truth > D] = 0.0
                    top2 = np.sort(prob, axis=1)
                    unsure_share.append(float(((top2[:, 2] - top2[:, 1]) < margin).mean()))
                    path = decode_samples(prob, f["half"], DECODER, prior)
                    full = expand(path, f["frames"], f["half"], data[i]["halves"], len(f["labels"]))
                    scores.append(m.evaluate(f["labels"], full)["spell_f1"])
                scores = np.array(scores)
                gain = scores - base
                where = "every sample" if margin > 1 else f"top two within {margin}"
                name = f"{label}, {where}, strength {bonus:.0f}"
                record["ceiling"][name] = {"spell_f1": float(scores.mean()), "gain": float(gain.mean()),
                                           "up": int((gain > 0).sum()), "unsure_share": float(np.mean(unsure_share))}
                print(f"  {name:<48}{np.mean(unsure_share):>15.0%}{scores.mean():>9.3f}{gain.mean():>+9.3f}"
                      f"{int((gain > 0).sum()):>5}/{len(ids)}", flush=True)
    record.update(matches=ids, base=float(base.mean()))
    save_manifest(OUT, record)
