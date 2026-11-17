"""The SkillCorner part of the v6 rule: the lead over the hand-built rules must not be smaller than v5's.

Only the 10 working matches (datasets.SKILLCORNER_DEV). The reserve matches are never loaded here.
Both models are trained on all 10 DFL and Metrica matches and scored on the standard test treatment,
as in experiments/skillcorner_check.py. The candidate (three models, smoother, decoder setting) is
also saved to runs/models/v6_candidate.pkl for dry runs. Open data only.

    python3 experiments/skillcorner_v6.py      # about 45 minutes
"""

import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
from mtah import datasets as ds, harness, metrics as m  # noqa: E402
from mtah.config import DECODER, TRAIN_COPIES  # noqa: E402
from mtah.decoder import DecoderConfig, decode_samples, expand  # noqa: E402
from mtah.degrade import DegradeConfig, load_stats  # noqa: E402
from mtah.model import AveragedModel, LearnedModel, RuleModel, TreeConfig  # noqa: E402
from v6_common import DRAWS, averaged, fit_smoother  # noqa: E402

DEV = ds.SKILLCORNER_DEV
RULE_DECODER = DecoderConfig(temperature=1.0, turnover=1.5, to_dead=6.0, from_dead=6.0)
V5_MODEL = os.path.join("runs", "models", "v5_final_model.pkl")
V6_MODEL = os.path.join("runs", "models", "v6_candidate.pkl")


def spell(labels, pred):
    return m.evaluate(labels, pred)["spell_f1"]


if __name__ == "__main__":
    os.chdir(ROOT)
    with open(os.path.join("runs", "oof", "v6", "summary.json")) as handle:
        summary = json.load(handle)
    candidate = list(summary["results"])[-1]
    decoder = DecoderConfig(**summary["results"][candidate]["best_overall_config"])
    stats = load_stats()
    ids = ds.available()

    started = time.time()
    if os.path.exists(V5_MODEL):
        with open(V5_MODEL, "rb") as handle:
            v5 = pickle.load(handle)
    else:
        v5 = LearnedModel(decoder=DECODER)
        v5.fit(harness.training_matches(ids, 0, TRAIN_COPIES, stats))
    models = []
    for draw in range(len(DRAWS)):
        model = LearnedModel(trees=TreeConfig(seed=draw), extra=True)
        model.fit(harness.training_matches(ids, draw, TRAIN_COPIES, stats))
        models.append(model)
    ensemble = AveragedModel(models, decoder=decoder)
    smoother = fit_smoother(averaged(DRAWS), ids)
    os.makedirs(os.path.dirname(V6_MODEL), exist_ok=True)
    with open(V6_MODEL, "wb") as handle:
        pickle.dump({"ensemble": ensemble, "smoother": smoother, "decoder": decoder, "candidate": candidate}, handle)
    print(f"candidate: {candidate} with {decoder}; models ready in {time.time() - started:.0f}s", flush=True)

    rules = RuleModel(decoder=RULE_DECODER)
    rows = []
    for match_id in DEV:
        match = harness.prepared(match_id, DegradeConfig(), harness.TEST_SEED_OFFSET, stats)
        blind = harness.without_labels(match)
        prob, frames, half = ensemble.probabilities(blind)
        path = decode_samples(smoother.probabilities(prob, half), half, decoder)
        v6_pred = expand(path, frames, half, match.halves, match.n_frames)
        rows.append((spell(match.labels, v5.predict(blind)), spell(match.labels, v6_pred),
                     spell(match.labels, rules.predict(blind))))
        print(f"{match_id}  v5 {rows[-1][0]:.3f}  v6 {rows[-1][1]:.3f}  rules {rows[-1][2]:.3f}", flush=True)
    r = np.array(rows)
    lead5, lead6 = r[:, 0] - r[:, 2], r[:, 1] - r[:, 2]
    print(f"\nv5 {r[:, 0].mean():.3f}, v6 {r[:, 1].mean():.3f}, rules {r[:, 2].mean():.3f}")
    print(f"lead over rules: v5 {lead5.mean():+.3f}, v6 {lead6.mean():+.3f}; v6 ahead of v5 on "
          f"{(r[:, 1] > r[:, 0]).sum()} of {len(r)}")
    passed = lead6.mean() >= lead5.mean()
    print("SkillCorner part of the rule (lead not smaller than v5's): " + ("PASSED" if passed else "NOT passed"))
