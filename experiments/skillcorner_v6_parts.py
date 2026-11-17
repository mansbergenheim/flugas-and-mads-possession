"""Which pieces of the v6 candidate carry over to SkillCorner's working half.

experiments/skillcorner_v6.py gives one number for the whole candidate. This splits it into the
pieces experiments/evaluate_v6.py reports for the 10 open matches, so the two tables can be read
side by side. No first model is trained here: the candidate saved by skillcorner_v6.py is reused.
Two more smoothers are fitted (for v5's own probabilities and for one draw), the same way that
script fits the candidate's. Each piece is decoded with the setting evaluate_v6.py chose for it on
the 10 open matches, and again with v5's shipped setting, so nothing is chosen on SkillCorner.

Only the 10 working matches (datasets.SKILLCORNER_DEV). The reserve matches are never loaded here.
Open data only.

    python3 experiments/skillcorner_v6_parts.py      # about 5 minutes
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
from mtah.config import DECODER  # noqa: E402
from mtah.decoder import DecoderConfig, decode_samples, expand  # noqa: E402
from mtah.degrade import DegradeConfig, load_stats  # noqa: E402
from mtah.model import RuleModel  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402
from skillcorner_v6 import DEV, RULE_DECODER, V5_MODEL, V6_MODEL  # noqa: E402
from v6_common import DRAWS, fit_smoother, load  # noqa: E402

OUT = os.path.join("runs", "skillcorner_v6_parts.json")
SAME_DECODER = ", v5's decoder setting"


if __name__ == "__main__":
    os.chdir(ROOT)
    record = manifest(matches=list(DEV))
    with open(os.path.join("runs", "oof", "v6", "summary.json")) as handle:
        home = json.load(handle)["results"]
    setting = {name: DecoderConfig(**r["best_overall_config"]) for name, r in home.items()}
    with open(V5_MODEL, "rb") as handle:
        v5 = pickle.load(handle)
    with open(V6_MODEL, "rb") as handle:
        saved = pickle.load(handle)
    ensemble, smoother3, candidate = saved["ensemble"], saved["smoother"], saved["candidate"]
    averaged_name = candidate.replace(", smoothed", "")
    if saved["decoder"] != setting[candidate] or averaged_name not in setting:
        raise ValueError("the saved candidate does not match runs/oof/v6/summary.json")
    stats = load_stats()
    ids = ds.available()

    started = time.time()
    smoother5 = fit_smoother(load("v5"), ids)
    smoother1 = fit_smoother(load(DRAWS[0]), ids)
    print(f"two smoothers fitted in {time.time() - started:.0f}s", flush=True)

    rules = RuleModel(decoder=RULE_DECODER)
    pieces = ["v5", "v5, smoothed", "extra inputs", "extra inputs, smoothed", averaged_name, candidate]
    names = pieces + [piece + SAME_DECODER for piece in pieces[1:]]
    rows = {name: [] for name in names}
    rule_rows = []
    for match_id in DEV:
        match = harness.prepared(match_id, DegradeConfig(), harness.TEST_SEED_OFFSET, stats)
        blind = harness.without_labels(match)

        def grade(prob, frames, half, cfg):
            path = decode_samples(prob, half, cfg)
            return m.evaluate(match.labels, expand(path, frames, half, match.halves, match.n_frames))

        p5, f5, h5 = v5.probabilities(blind)
        values, swapped, frames, half = ensemble.models[0].views(blind)
        each = [model.probabilities_from(values, swapped) for model in ensemble.models]
        mean = sum(each) / float(len(each))
        prob = {
            "v5, smoothed": (smoother5.probabilities(p5, h5), f5, h5),
            "extra inputs": (each[0], frames, half),
            "extra inputs, smoothed": (smoother1.probabilities(each[0], half), frames, half),
            averaged_name: (mean, frames, half),
            candidate: (smoother3.probabilities(mean, half), frames, half),
        }
        got = {"v5": grade(p5, f5, h5, DECODER)}
        for piece, (p, f, h) in prob.items():
            got[piece] = grade(p, f, h, setting[piece])
            got[piece + SAME_DECODER] = grade(p, f, h, DECODER)
        for name in names:
            rows[name].append(got[name])
        rule_rows.append(m.evaluate(match.labels, rules.predict(blind))["spell_f1"])
        print(f"{match_id}  v5 {got['v5']['spell_f1']:.3f}  v6 {got[candidate]['spell_f1']:.3f}  "
              f"rules {rule_rows[-1]:.3f}", flush=True)

    base = np.array([r["spell_f1"] for r in rows["v5"]])
    results = {}
    print(f"\nrules {np.mean(rule_rows):.3f}")
    print(f"\n{'':66}spellF1  macroF1  count    gain  up/down   gain at home")
    for name in names:
        s = np.array([r["spell_f1"] for r in rows[name]])
        gain = s - base
        count = float(np.mean([r["n_pred_spells"] / r["n_gt_spells"] for r in rows[name]]))
        macro = float(np.mean([r["macro_f1"] for r in rows[name]]))
        at_home = home[name]["gain"] if name in home else None
        results[name] = {"spell_f1": float(s.mean()), "per_match": dict(zip(DEV, s.tolist())),
                         "gain": float(gain.mean()), "up": int((gain > 0).sum()), "down": int((gain < 0).sum()),
                         "macro_f1": macro, "count_ratio": count, "gain_at_home": at_home}
        print(f"{name:<66}{s.mean():.3f}    {macro:.3f}  x{count:.2f}  {gain.mean():+.3f}  "
              f"{results[name]['up']}/{results[name]['down']}      "
              + ("" if at_home is None else f"{at_home:+.3f}"), flush=True)
    record.update(results=results, rules=float(np.mean(rule_rows)))
    save_manifest(OUT, record)
