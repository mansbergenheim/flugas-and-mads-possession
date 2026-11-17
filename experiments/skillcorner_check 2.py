"""SkillCorner's open matches: a test on tracking the model has never seen, then a training try.

Only the 10 working matches (datasets.SKILLCORNER_DEV) are used. The 10 reserve matches are kept
for one final check and must not be loaded here.

    python3 experiments/skillcorner_check.py test    # about 15 minutes
    python3 experiments/skillcorner_check.py train   # about 2 hours

test: one model trained on all 10 DFL and Metrica matches, exactly as the submission trains, is
scored on each working SkillCorner match next to the hand-built rules and the organisers' baseline.
Two versions of the input:
  - standard: the same treatment as every other test match (ball put back during stoppages, then
    blackouts, noise and anonymous player slots), so it compares directly with our other numbers;
  - as it comes: SkillCorner's own tracking, detected positions only. Its ball goes missing mostly
    when the ball is dead, which favours anything that has learned that shortcut.
SkillCorner's labels come from its algorithms, so every model's score falls; what matters is the
learned model's lead over the rules.

train: the 10 DFL and Metrica matches are each held out in turn, as for v5, but training adds the
10 working SkillCorner matches, treated like every other training match. Compared with v5 on the
same test copies, each with its decoder picked on the other matches.

Decided before any result was seen:
  - test: the model "carries over" if, on the standard input, it beats the rules on at least 8 of
    10 matches by a mean of at least 0.16, half its lead on our own matches (0.630 against 0.312).
    Otherwise it's a warning to look into. Nothing is changed either way.
  - train: SkillCorner is worth adding to training only if the gain on the DFL and Metrica matches
    is at least +0.01 with at least 7 of 10 up, and the Metrica matches don't fall on average. A
    smaller gain is a tie, and a tie keeps the simpler current model. Either way the decision to
    change the submitted model is the team's, not this script's.
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from mtah import datasets as ds, harness, metrics as m  # noqa: E402
from mtah.baselines import NearestPlayer  # noqa: E402
from mtah.config import DECODER, TRAIN_COPIES  # noqa: E402
from mtah.decoder import DecoderConfig, decode_samples, expand  # noqa: E402
from mtah.degrade import DegradeConfig, load_stats  # noqa: E402
from mtah.model import LearnedModel, RuleModel  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402
from sweep_decoder import GRID, load, score  # noqa: E402

DEV = ds.SKILLCORNER_DEV
# The rules' decoder from runs/sweep_v5_rule.txt, where they scored 0.312 on our own matches.
RULE_DECODER = DecoderConfig(temperature=1.0, turnover=1.5, to_dead=6.0, from_dead=6.0)


def spell(labels, pred):
    return m.evaluate(labels, pred)["spell_f1"]


def run_test(seed):
    stats = load_stats()
    model = LearnedModel(decoder=DECODER)
    model.fit(harness.training_matches(ds.available(), seed, TRAIN_COPIES, stats))
    rules = RuleModel(decoder=RULE_DECODER)
    rules.fit([])
    nearest = NearestPlayer()
    nearest.fit([])
    rows = []
    for match_id in DEV:
        clean = harness.harmonise(ds.load_open(match_id))
        inputs = {"standard": harness.prepared(match_id, DegradeConfig(), seed + harness.TEST_SEED_OFFSET, stats),
                  "as it comes": clean}
        for name, match in inputs.items():
            blind = harness.without_labels(match)
            rows.append((match_id, name, spell(match.labels, model.predict(blind)),
                         spell(match.labels, rules.predict(blind)), spell(match.labels, nearest.predict(blind))))
            print(f"{match_id} {name:<12} learned {rows[-1][2]:.3f}  rules {rows[-1][3]:.3f}  "
                  f"baseline {rows[-1][4]:.3f}", flush=True)
    for name in ("standard", "as it comes"):
        r = np.array([row[2:] for row in rows if row[1] == name])
        lead = r[:, 0] - r[:, 1]
        print(f"\n{name}: learned {r[:, 0].mean():.3f}, rules {r[:, 1].mean():.3f}, "
              f"baseline {r[:, 2].mean():.3f}; lead over rules {lead.mean():+.3f}, "
              f"ahead on {(lead > 0).sum()} of {len(lead)}")
    lead = np.array([row[2] - row[3] for row in rows if row[1] == "standard"])
    verdict = "carries over" if (lead > 0).sum() >= 8 and lead.mean() >= 0.16 else "warning: look into it"
    print(f"verdict (standard input): {verdict}")


def run_train(seed, tag):
    ids = ds.available()
    cfg, stats = DegradeConfig(), load_stats()
    folder = os.path.join("runs", "oof", tag)
    os.makedirs(folder, exist_ok=True)
    save_manifest(os.path.join(folder, "manifest.json"),
                  manifest(held_out=list(ids), seed=seed, extra_training=list(DEV)))
    for held in ids:
        out = os.path.join(folder, f"{held}.npz")
        if os.path.exists(out):
            continue
        started = time.time()
        test = harness.prepared(held, cfg, seed + harness.TEST_SEED_OFFSET, stats)
        model = LearnedModel()
        model.fit(harness.training_matches([i for i in ids if i != held] + list(DEV), seed, TRAIN_COPIES, stats))
        prob, frames, half = model.probabilities(harness.without_labels(test))
        np.savez_compressed(out, prob=prob, frames=frames, half=half, labels=test.labels,
                            halves=np.array(test.halves), source=test.source)
        print(f"{held} done in {time.time() - started:.0f}s", flush=True)

    new, old = load(tag), load("v5")
    configs = [DecoderConfig(*c) for c in itertools.product(*GRID.values())]
    scores = {}
    for name, folds in (("v5", old), ("with SkillCorner", new)):
        grid = np.array([[score(folds[i], c)["spell_f1"] for i in ids] for c in configs])
        scores[name] = np.array([grid[int(np.delete(grid, j, axis=1).mean(1).argmax()), j] for j in range(len(ids))])
    for held in ids:
        if not np.array_equal(old[held]["labels"], new[held]["labels"]):
            raise ValueError(f"{held}: test copy differs from v5's")
    diff = scores["with SkillCorner"] - scores["v5"]
    met = np.array([not i.startswith("J") for i in ids])
    for name, s in scores.items():
        print(f"{name:<18}spellF1 {s.mean():.3f}  DFL {s[~met].mean():.3f}  Metrica {s[met].mean():.3f}")
    print(f"gain {diff.mean():+.3f}, {(diff > 0).sum()} up / {(diff < 0).sum()} down, "
          f"Metrica {diff[met].mean():+.3f}")
    print("   per match: " + " ".join(f"{i}:{d:+.3f}" for i, d in zip(ids, diff)))
    worth = diff.mean() >= 0.01 and (diff > 0).sum() >= 7 and diff[met].mean() >= 0
    print("verdict: " + ("worth adding, for the team to decide" if worth else "not worth adding; keep v5"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=("test", "train"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="v5_skillcorner")
    args = parser.parse_args()
    os.chdir(ROOT)
    run_test(args.seed) if args.part == "test" else run_train(args.seed, args.tag)
