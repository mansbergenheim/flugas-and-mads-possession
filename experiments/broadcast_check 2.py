"""Why does the learned model's lead shrink on SkillCorner? Tracking or labels?

On SkillCorner the learned model beat the rules on all 10 working matches, but by 0.085 instead of
the 0.32 it has on our own matches (runs/skillcorner_test.txt). SkillCorner differs in three ways:
a broadcast camera sees about 9.5 players a frame, the tracking is 10 fps rather than 25, and the
labels come from SkillCorner's algorithms. The first two can be put onto our own matches, whose
labels stay as they are:
  - camera: only the 10 players nearest the ball are kept on each frame
  - 10 fps: every frame rebuilt from 10 fps samples, the way SkillCorner's are
SkillCorner can also be given back its guesses for players off camera (22 players, still 10 fps).

One model, trained without the 3 test matches as in clean_tracking_check.py. About 20 minutes.
The competition's tracking has about 20.5 objects at 25 Hz. If camera and 10 fps take most of the
lead away on our own matches, the shrinking is about SkillCorner's tracking. If they don't, it's
about labels or league, and the lead may be smaller on the competition's labels too.

    python3 experiments/broadcast_check.py
"""

import dataclasses
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from mtah import datasets as ds, harness, metrics as m  # noqa: E402
from mtah.config import DECODER, TRAIN_COPIES  # noqa: E402
from mtah.decoder import DecoderConfig  # noqa: E402
from mtah.degrade import DegradeConfig, degrade, fill_ball, load_stats  # noqa: E402
from mtah.loaders.skillcorner import _resample, load_skillcorner  # noqa: E402
from mtah.model import LearnedModel, RuleModel  # noqa: E402

HELD_OUT = ("J03WR9", "J03WOH", "METRICA2")
RULE_DECODER = DecoderConfig(temperature=1.0, turnover=1.5, to_dead=6.0, from_dead=6.0)
SEEN = 10


def camera(match):
    """Keep only the SEEN players nearest the ball, as a broadcast view roughly does."""
    ball = fill_ball(match.ball)
    gap = np.hypot(*(match.players - ball[:, None, :]).transpose(2, 0, 1))
    rank = np.argsort(np.argsort(np.where(np.isfinite(gap), gap, np.inf), axis=1), axis=1)
    players = np.where((rank < SEEN)[..., None], match.players, np.nan).astype(np.float32)
    return dataclasses.replace(match, players=players)


def ten_fps(match):
    """Sample at 10 fps, then rebuild 25 Hz by the same interpolation the SkillCorner loader uses."""
    def rebuild(values):
        out = np.full_like(values, np.nan)
        for a, b in match.halves:
            part = values[a:b]
            n = b - a
            u = np.arange(int(np.ceil((n - 1) / 2.5)) + 2) * 2.5
            i0 = np.minimum(np.floor(u).astype(int), n - 1)
            i1 = np.minimum(i0 + 1, n - 1)
            w = (u - np.floor(u)).reshape((-1,) + (1,) * (values.ndim - 1))
            sampled = (1 - w) * part[i0] + w * part[i1]
            back, _ = _resample(sampled, 0, len(sampled) - 1)
            out[a:b] = back[:n]
        return out
    return dataclasses.replace(match, ball=rebuild(match.ball), players=rebuild(match.players))


CHANGES = {"as tested": lambda x: x, "camera": camera, "10 fps": ten_fps,
           "camera and 10 fps": lambda x: camera(ten_fps(x))}


def scores(model, rules, match):
    blind = harness.without_labels(match)
    out = []
    for predictor in (model, rules):
        r = m.evaluate(match.labels, predictor.predict(blind))
        out.append((r["spell_f1"], r["team_acc"], r["alive_dead_f1"], r["n_pred_spells"] / r["n_gt_spells"]))
    return out


def show(title, rows):
    r = np.array(rows)  # match, model, measure
    print(f"  {title:<20}learned {r[:, 0, 0].mean():.3f}  rules {r[:, 1, 0].mean():.3f}  "
          f"lead {(r[:, 0, 0] - r[:, 1, 0]).mean():+.3f}   team acc {r[:, 0, 1].mean():.3f}/{r[:, 1, 1].mean():.3f}"
          f"  alive/dead F1 {r[:, 0, 2].mean():.3f}/{r[:, 1, 2].mean():.3f}"
          f"  count x{r[:, 0, 3].mean():.2f}/x{r[:, 1, 3].mean():.2f}", flush=True)


if __name__ == "__main__":
    os.chdir(ROOT)
    stats = load_stats()
    model = LearnedModel(decoder=DECODER)
    model.fit(harness.training_matches([i for i in ds.available() if i not in HELD_OUT], 0, TRAIN_COPIES, stats))
    rules = RuleModel(decoder=RULE_DECODER)
    print("our 3 test matches, standard treatment after each change (learned/rules)")
    for name, change in CHANGES.items():
        rows = []
        for held in HELD_OUT:
            clean = change(harness.harmonise(ds.load_open(held)))
            rows.append(scores(model, rules, degrade(clean, DegradeConfig(), harness.TEST_SEED_OFFSET, stats)))
        show(name, rows)

    print("\nSkillCorner working matches, standard treatment (learned/rules)")
    for name, guesses in (("detected only", False), ("with its guesses", True)):
        rows = []
        for match_id in ds.SKILLCORNER_DEV:
            f = os.path.join(ds.OPEN_DIR, "skillcorner", match_id[2:], match_id[2:])
            clean = load_skillcorner(f + "_match.json", f + "_tracking_extrapolated.jsonl",
                                     f + "_phases_of_play.csv", match_id, guesses=guesses)
            rows.append(scores(model, rules, degrade(clean, DegradeConfig(), harness.TEST_SEED_OFFSET, stats)))
        show(name, rows)
