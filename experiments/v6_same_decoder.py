"""The 10-match part of the v6 rule again, with no decoder setting re-picked for the candidate.

experiments/evaluate_v6.py lets every piece choose its own decoder setting on the other matches.
Here every piece is decoded with v5's shipped setting, so the only difference from v5 is the model.
That setting was chosen on these 10 matches for v5, so this comparison leans towards v5.

The smoothed held-out probabilities are saved under runs/oof/<name>, where
scripts/sweep_decoder.py --tag <name> can read them. Open data only.

    python3 experiments/v6_same_decoder.py      # about 10 minutes
"""

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
from mtah.config import DECODER  # noqa: E402
from sweep_decoder import score  # noqa: E402
from v6_common import DRAWS, MIN_GAIN, MIN_UP, averaged, load, smoothed  # noqa: E402

SAVE_AS = {"v5, smoothed": "v5_smoothed", "extra inputs, smoothed": "v6_d0_smoothed",
           "extra inputs, 3 draws averaged, smoothed": "v6_avg_smoothed"}


if __name__ == "__main__":
    os.chdir(ROOT)
    base = load("v5")
    ids = sorted(base)
    sets = {"v5": base, "extra inputs": load(DRAWS[0]), "extra inputs, 3 draws averaged": averaged(DRAWS)}
    for name in list(sets):
        smooth_name = name + ", smoothed"
        sets[smooth_name] = smoothed(sets[name])
        folder = os.path.join("runs", "oof", SAVE_AS[smooth_name])
        os.makedirs(folder, exist_ok=True)
        for match_id, fold in sets[smooth_name].items():
            np.savez_compressed(os.path.join(folder, match_id + ".npz"), **fold)
        print(f"smoothed '{name}', saved under {folder}", flush=True)

    dfl = np.array([i.startswith("J") for i in ids])
    reference = np.array([score(base[i], DECODER)["spell_f1"] for i in ids])
    print(f"\nevery piece decoded with {DECODER}")
    print(f"\n{'':52}spellF1    DFL  Metrica  count    gain  up/down")
    order = ["v5", "v5, smoothed", "extra inputs", "extra inputs, smoothed", "extra inputs, 3 draws averaged",
             "extra inputs, 3 draws averaged, smoothed"]
    for name in order:
        rows = [score(sets[name][i], DECODER) for i in ids]
        s = np.array([r["spell_f1"] for r in rows])
        gain = s - reference
        count = float(np.mean([r["n_pred_spells"] / r["n_gt_spells"] for r in rows]))
        up, down = int((gain > 0).sum()), int((gain < 0).sum())
        print(f"{name:<52}{s.mean():.3f}  {s[dfl].mean():.3f}    {s[~dfl].mean():.3f}  x{count:.2f}  "
              f"{gain.mean():+.3f}  {up}/{down}", flush=True)
        if name != "v5":
            print("      per match: " + " ".join(f"{i}:{g:+.3f}" for i, g in zip(ids, gain)))
    passed = gain.mean() >= MIN_GAIN and up >= MIN_UP
    print(f"\n10-match part of the rule for '{order[-1]}' with v5's decoder setting "
          f"(gain at least {MIN_GAIN}, at least {MIN_UP} of {len(ids)} up): " + ("PASSED" if passed else "NOT passed"))
