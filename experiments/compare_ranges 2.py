"""Paired comparison of the wider and narrower training ranges on identical test data.

    python3 experiments/compare_ranges.py
"""

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from mtah import harness  # noqa: E402
from mtah.config import DECODER  # noqa: E402
from sweep_decoder import load, score  # noqa: E402

old, new = load("v5_old_ranges"), load("v5")
ids = sorted(new)
assert all(np.array_equal(old[i]["labels"], new[i]["labels"]) for i in ids)
a = [dict(match=i, **score(old[i], DECODER)) for i in ids]
b = [dict(match=i, **score(new[i], DECODER)) for i in ids]
for name, rows in (("narrow range", a), ("wide range", b)):
    print(f"{name:<14} spellF1 {np.mean([r['spell_f1'] for r in rows]):.3f}  "
          f"macroF1 {np.mean([r['macro_f1'] for r in rows]):.3f}  "
          f"possessions x{np.mean([r['n_pred_spells'] / r['n_gt_spells'] for r in rows]):.2f}")
print(harness.compare(a, b))
print("gain per match:", " ".join(f"{x['match']}:{y['spell_f1'] - x['spell_f1']:+.3f}" for x, y in zip(a, b)))
