"""What could spoken player names be worth? Open data only, no audio.

The idea: the commentator names the player on the ball ("Keller... to Duarte"), the name tells the
team, and that team gets a nudge in the decoder before it decides, like Smart Tagging's nudges.
The open matches have no sound, so mentions are simulated from the labels with the faults a real
commentary has:

- rate:  a name only every so many seconds of play, at random moments (Poisson);
- delay: the name is said about a second after the touch (1.0 s on average, 0.5 s spread);
- wrong: a share of mentions give the other team (a misheard name, a name mapped to the wrong
  team, a player talked about who is not on the ball).

Each mention says "this team had the ball just before I spoke" and nudges that team over the
2 seconds before the moment it was said. Mentions fall only in play. On the 10 open matches held
out, with trees + TCN at v5's decoder setting. Every row is averaged over three random draws of
mentions. The strengths are tried on the same ten matches, so the best row flatters itself.

    python3 experiments/name_mentions.py      # a few minutes
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
from mtah.match import A, H  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402
from v6_common import DRAWS, averaged, load  # noqa: E402

FPS = 25
DELAY, SPREAD = 1.0, 0.5         # seconds from the touch to the name being said
WINDOW = 2 * FPS                 # frames before the spoken moment that the nudge covers
RATES = (3.0, 6.0, 10.0)         # seconds of play per mention, on average
WRONG = (0.1, 0.25)
STRENGTHS = (0.5, 1.0, 2.0)
SEEDS = (0, 1, 2)
OUT = os.path.join("runs", "name_mentions.json")


def mentions(labels, rate, wrong, rng):
    """(spoken frame, team said) for simulated name mentions during play."""
    live = np.flatnonzero((labels == H) | (labels == A))
    n = rng.poisson(len(live) / (rate * FPS))
    touched = np.sort(rng.choice(live, size=min(n, len(live)), replace=False))
    spoken = touched + np.round(rng.normal(DELAY, SPREAD, len(touched)) * FPS).astype(int)
    team = labels[touched].astype(np.int64)
    flip = rng.random(len(team)) < wrong
    team[flip] = 1 - team[flip]                  # H = 0, A = 1
    keep = (spoken > 0) & (spoken < len(labels))
    return spoken[keep], team[keep]


def prior_from(frames, spoken, team, strength):
    prior = np.zeros((len(frames), 3))
    for s, t in zip(spoken, team):
        lo, hi = np.searchsorted(frames, (s - WINDOW, s))
        prior[lo:hi, t] += strength
    return prior


if __name__ == "__main__":
    os.chdir(ROOT)
    record = manifest(delay=DELAY, spread=SPREAD, window=WINDOW, rates=list(RATES), wrong=list(WRONG),
                      strengths=list(STRENGTHS), seeds=list(SEEDS))
    trees, tcn = averaged(DRAWS), load("net_tcn")
    ids = sorted(trees)
    data = {}
    for i in ids:
        f = trees[i]
        prob = (f["prob"] + tcn[i]["prob"]) / 2.0
        halves = [tuple(int(v) for v in row) for row in f["halves"]]
        full = expand(decode_samples(prob, f["half"], DECODER), f["frames"], f["half"], halves, len(f["labels"]))
        data[i] = dict(f=f, prob=prob, halves=halves, base=m.evaluate(f["labels"], full)["spell_f1"])
    base = np.array([data[i]["base"] for i in ids])
    print(f"spoken player names as nudges to the decoder (trees + TCN as predicted: {base.mean():.3f})")
    print(f"  {'':<52}{'spellF1':>9}{'gain':>9}{'up':>8}")
    record["rows"] = {}
    for rate in RATES:
        for wrong in WRONG:
            for strength in STRENGTHS:
                per_seed = []
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    scores = []
                    for i in ids:
                        f = data[i]["f"]
                        spoken, team = mentions(f["labels"], rate, wrong, rng)
                        prior = prior_from(f["frames"], spoken, team, strength)
                        path = decode_samples(data[i]["prob"], f["half"], DECODER, prior)
                        full = expand(path, f["frames"], f["half"], data[i]["halves"], len(f["labels"]))
                        scores.append(m.evaluate(f["labels"], full)["spell_f1"])
                    per_seed.append(scores)
                scores = np.mean(per_seed, axis=0)
                gain = scores - base
                name = f"a name every {rate:.0f} s, {wrong:.0%} wrong, strength {strength:g}"
                record["rows"][name] = {"spell_f1": float(scores.mean()), "gain": float(gain.mean()),
                                        "up": int((gain > 0).sum())}
                print(f"  {name:<52}{scores.mean():>9.3f}{gain.mean():>+9.3f}{int((gain > 0).sum()):>5}/{len(ids)}",
                      flush=True)
    record.update(matches=ids, base=float(base.mean()))
    save_manifest(OUT, record)
