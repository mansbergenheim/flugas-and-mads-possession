"""Do reported stoppages from the commentary help, at the strength the pipeline uses? Open data only.

mtah/commentary.py turns a reported stoppage (a foul, offside, ball out... said to have happened)
into a dead-ball nudge of 0.15 per sample from 1 s before to 4 s after the moment it was said. That
strength was set by judgement, not measured. This simulates reports on the ten open matches from
their labels, with the faults real commentary has, and decodes with and without them:

- recall:    the share of real stoppages the commentary reports at all;
- delay:     a report comes about 3 s after play stops (1.5 s spread), never before;
- precision: the share of reports that are real; the rest fall at random moments of live play.

Then the whole commentary as the pipeline would use it: player names (one every 6 s of play, a
quarter pointing at the wrong team, strength 0.5, 1 s delay, as experiments/name_mentions.py) plus
stoppage reports, through mtah/commentary.py's own masks and prior. Trees + TCN as predicted on the
held-out matches, v5's decoder, no Smart Tagging (the open matches have none). Three random draws
per row. The strengths beside 0.15 are shown to see the shape, not to pick one: any change is the
team's decision, and would be made here on open data, never on the competition match.

    python3 experiments/commentary_dead_sim.py      # a few minutes
"""

import os
import sys
from dataclasses import replace

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from mtah import commentary as cm  # noqa: E402
from mtah import metrics as m  # noqa: E402
from mtah.config import DECODER  # noqa: E402
from mtah.decoder import decode_samples, expand  # noqa: E402
from mtah.match import A, D, H  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402
from v6_common import DRAWS, averaged, load  # noqa: E402

FPS = 25
DELAY, SPREAD = 3.0, 1.5
RECALLS = (0.3, 0.6)
PRECISIONS = (0.6, 0.8, 0.95)
DEAD = (0.075, 0.15, 0.3)
SEEDS = (0, 1, 2)
NAME_RATE, NAME_WRONG, NAME_DELAY = 6.0, 0.25, 1.0
OUT = os.path.join("runs", "commentary_dead_sim.json")


def stoppage_reports(labels, recall, precision, rng):
    """Spoken frames of simulated stoppage reports: real ones after a stoppage starts, false ones in play."""
    dead = labels == D
    starts = np.flatnonzero(dead[1:] & ~dead[:-1]) + 1
    real = starts[rng.random(len(starts)) < recall]
    spoken = real + np.round(np.abs(rng.normal(DELAY, SPREAD, len(real))) * FPS).astype(int)
    n_false = int(round(len(real) * (1 - precision) / precision))
    live = np.flatnonzero((labels == H) | (labels == A))
    false = rng.choice(live, size=min(n_false, len(live)), replace=False) if n_false else np.zeros(0, int)
    spoken = np.concatenate([spoken, false])
    return [{"label": "foul", "status": "happened", "basis": "described_live", "frame": int(f)}
            for f in spoken if 0 <= f < len(labels)]


def name_mentions(labels, rng):
    live = np.flatnonzero((labels == H) | (labels == A))
    n = rng.poisson(len(live) / (NAME_RATE * FPS))
    touched = np.sort(rng.choice(live, size=min(n, len(live)), replace=False))
    spoken = touched + np.round(rng.normal(NAME_DELAY, 0.5, len(touched)) * FPS).astype(int)
    team = labels[touched].astype(int)
    flip = rng.random(len(team)) < NAME_WRONG
    team[flip] = 1 - team[flip]
    return [{"side": "home" if t == H else "away", "role": "on_ball", "frame": int(s)}
            for s, t in zip(spoken, team) if 0 <= s < len(labels)]


def score(d, prior):
    f = d["f"]
    path = decode_samples(d["prob"], f["half"], DECODER, prior)
    return m.evaluate(f["labels"], expand(path, f["frames"], f["half"], d["halves"], len(f["labels"])))["spell_f1"]


if __name__ == "__main__":
    os.chdir(ROOT)
    record = manifest(delay=DELAY, spread=SPREAD, recalls=list(RECALLS), precisions=list(PRECISIONS),
                      dead=list(DEAD), seeds=list(SEEDS))
    trees, tcn = averaged(DRAWS), load("net_tcn")
    ids = sorted(trees)
    data = {}
    for i in ids:
        f = trees[i]
        prob = (f["prob"] + tcn[i]["prob"]) / 2.0
        halves = [tuple(int(v) for v in row) for row in f["halves"]]
        d = dict(f=f, prob=prob, halves=halves)
        d["base"] = score(d, None)
        data[i] = d
    base = np.array([data[i]["base"] for i in ids])
    print(f"stoppage reports from the commentary as dead-ball nudges (trees + TCN alone: {base.mean():.3f})")
    print(f"  {'':<58}{'spellF1':>9}{'gain':>9}{'up':>8}")
    record["rows"] = {}

    def row(name, make_prior):
        per_seed = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            per_seed.append([score(data[i], make_prior(data[i], rng)) for i in ids])
        scores = np.mean(per_seed, axis=0)
        gain = scores - base
        record["rows"][name] = {"spell_f1": float(scores.mean()), "gain": float(gain.mean()), "up": int((gain > 0).sum())}
        print(f"  {name:<58}{scores.mean():>9.3f}{gain.mean():>+9.3f}{int((gain > 0).sum()):>5}/{len(ids)}", flush=True)

    for recall in RECALLS:
        for precision in PRECISIONS:
            for dead in DEAD:
                w = replace(cm.CommentaryWeights(), dead=dead)

                def prior(d, rng, recall=recall, precision=precision, w=w):
                    f = d["f"]
                    ev = stoppage_reports(f["labels"], recall, precision, rng)
                    return cm.prior_from(cm.masks([], ev, f["frames"], f["half"], d["halves"], w), w)
                mark = "  <- pipeline" if dead == cm.CommentaryWeights().dead else ""
                row(f"{recall:.0%} of stoppages reported, {precision:.0%} real, dead {dead:g}{mark}", prior)

    print("\n  the whole commentary at the pipeline's settings (names: every 6 s, 25% wrong, strength 0.5):")
    for recall, precision in ((0.3, 0.6), (0.6, 0.8)):
        def both(d, rng, recall=recall, precision=precision):
            f = d["f"]
            names = name_mentions(f["labels"], rng)
            ev = stoppage_reports(f["labels"], recall, precision, rng)
            return cm.prior_from(cm.masks(names, ev, f["frames"], f["half"], d["halves"]))

        def names_only(d, rng):
            f = d["f"]
            return cm.prior_from(cm.masks(name_mentions(f["labels"], rng), [], f["frames"], f["half"], d["halves"]))
        if recall == 0.3:
            row("names only", names_only)
        row(f"names + stoppages ({recall:.0%} reported, {precision:.0%} real)", both)
    record.update(matches=ids, base=float(base.mean()))
    save_manifest(OUT, record)
