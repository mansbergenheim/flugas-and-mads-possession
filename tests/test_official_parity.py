"""Direct comparison with the organisers' possession_metrics.py on randomised predictions.

This imports their code, so it runs only when you ask for it:

    MTAH_OFFICIAL_REPO=../official-repository python3 -m unittest tests.test_official_parity
"""

import os
import sys
import unittest

import numpy as np

from mtah import metrics as m
from mtah.match import X

OFFICIAL = os.environ.get("MTAH_OFFICIAL_REPO")


@unittest.skipUnless(OFFICIAL, "set MTAH_OFFICIAL_REPO to compare with the official scorer")
class Parity(unittest.TestCase):
    def test_random_sequences_agree(self):
        sys.path.insert(0, os.path.join(OFFICIAL, "evaluation"))
        import possession_metrics as pm

        rng = np.random.default_rng(0)
        for trial in range(30):
            lengths = rng.integers(10, 400, 300)
            gt = np.repeat(rng.integers(0, 3, 300), lengths).astype(np.int8)
            gt[:50] = X
            gt[len(gt) // 2:len(gt) // 2 + 100] = X
            pred = gt.copy()
            for _ in range(int(rng.integers(5, 200))):
                a = int(rng.integers(0, len(gt) - 80))
                pred[a:a + int(rng.integers(1, 80))] = rng.integers(0, 4)
            ours = m.evaluate(gt, pred)
            theirs = pm.evaluate(gt, pred)
            self.assertAlmostEqual(ours["spell_f1"], theirs["spell_f1_mean"], places=10)
            self.assertAlmostEqual(ours["macro_f1"], theirs["macro_f1"], places=10)
            self.assertEqual(ours["n_pred_spells"], theirs["n_pred_spells"])


if __name__ == "__main__":
    unittest.main()
