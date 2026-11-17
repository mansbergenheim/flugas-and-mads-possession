"""Pin our metrics to the official scorer's saved output on the four public baselines."""

import csv
import os
import unittest

import numpy as np

from mtah import metrics as m
from mtah import submission as sub
from mtah.match import A, D, H, X

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.join(ROOT, "official-repository")
SCOREBOARD = os.path.join(ROOT, "research", "verification", "baseline_evaluation", "scoreboard.csv")


# A bare clone has neither folder. Nothing outside these tests needs them.
needs_official = unittest.skipUnless(os.path.isdir(REPO) and os.path.exists(SCOREBOARD),
                                     "needs ../official-repository and ../research/verification")


def public_gt():
    return sub.read_submission(os.path.join(REPO, "evaluation", "gt",
                                            "BAR-ATH_possession_gt_public.csv"))


def baseline(name):
    return sub.read_submission(os.path.join(REPO, "evaluation", "results", name + ".csv"))


@needs_official
class OfficialOutput(unittest.TestCase):
    """Every field below came out of the organisers' run_evaluation.py."""

    def test_all_baselines_match_saved_scoreboard(self):
        gt = public_gt()
        with open(SCOREBOARD) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        for row in rows:
            r = m.evaluate(gt, baseline(row["submission"]), bootstrap=1000)
            with self.subTest(row["submission"]):
                self.assertAlmostEqual(r["spell_f1"], float(row["spell_f1_mean"]), places=5)
                self.assertAlmostEqual(r["macro_f1"], float(row["macro_f1"]), places=5)
                self.assertEqual(r["n_pred_spells"], int(row["poss"]))
                self.assertEqual(r["n_gt_spells"], int(row["n_gt_spells"]))
                self.assertAlmostEqual(r["mean_pred_spell_s"], float(row["dur"]), places=4)
                self.assertAlmostEqual(r["alive_dead_f1"], float(row["alive_dead_f1"]), places=5)
                self.assertAlmostEqual(r["team_acc"], float(row["team_acc_both_alive"]), places=5)
                self.assertAlmostEqual(r["transition_f1_12"], float(row["trans_f1_12"]), places=5)
                self.assertAlmostEqual(r["transition_f1_25"], float(row["trans_f1_25"]), places=5)
                self.assertAlmostEqual(r["latency_25"], float(row["mean_latency"]), places=4)
                self.assertAlmostEqual(r["spell_ci"][0], float(row["spell_bootstrap_low"]), places=5)
                self.assertAlmostEqual(r["spell_ci"][1], float(row["spell_bootstrap_high"]), places=5)


class Behaviour(unittest.TestCase):
    def setUp(self):
        self.gt = np.array([X] * 5 + [H] * 100 + [A] * 100 + [D] * 50 + [H] * 100 + [X] * 5,
                           dtype=np.int8)

    def test_perfect_is_one(self):
        r = m.evaluate(self.gt, self.gt)
        self.assertEqual(r["spell_f1"], 1.0)
        self.assertEqual(r["macro_f1"], 1.0)

    def test_constant_guess_is_near_zero(self):
        # Constant H overlaps two real H spells, but each IoU is well under 0.3 on a real match.
        real = np.tile(np.r_[[H] * 50, [A] * 50], 200).astype(np.int8)
        self.assertEqual(m.evaluate(real, np.full_like(real, H))["spell_f1"], 0.0)

    def test_graded_x_is_filled_for_spells_but_charged_for_macro(self):
        pred = self.gt.copy()
        pred[40:60] = X
        r = m.evaluate(self.gt, pred)
        self.assertEqual(r["spell_f1"], 1.0)
        self.assertLess(r["macro_f1"], 1.0)
        self.assertEqual(r["n_abstain_graded"], 20)

    def test_ungraded_frames_are_ignored(self):
        pred = self.gt.copy()
        pred[:5] = A
        pred[-5:] = D
        self.assertEqual(m.evaluate(self.gt, pred)["spell_f1"], 1.0)

    def test_no_spell_crosses_a_graded_gap(self):
        gt = np.array([H] * 50 + [X] * 20 + [H] * 50, dtype=np.int8)
        self.assertEqual(m.evaluate(gt, gt)["n_gt_spells"], 2)

    def test_single_wrong_frame_splits_a_spell(self):
        pred = self.gt.copy()
        pred[50] = A
        r = m.evaluate(self.gt, pred)
        self.assertEqual(r["n_pred_spells"], 5)
        self.assertLess(r["spell_f1"], 1.0)
        self.assertGreater(r["macro_f1"], 0.99)

    def test_wrong_team_never_matches(self):
        pred = np.where(self.gt == H, A, np.where(self.gt == A, H, self.gt)).astype(np.int8)
        self.assertEqual(m.evaluate(self.gt, pred)["spell_f1"], 0.0)


class Submission(unittest.TestCase):
    @needs_official
    def test_force_no_abstain_keeps_spell_and_lifts_macro(self):
        gt = public_gt()
        pred = baseline("baseline_smooth50").copy()
        rng = np.random.default_rng(0)
        idx = rng.integers(11300, 18600, 400)
        pred[idx] = X
        before = m.evaluate(gt, pred)
        after = m.evaluate(gt, sub.force_no_abstain(pred))
        self.assertAlmostEqual(before["spell_f1"], after["spell_f1"], places=12)
        self.assertGreaterEqual(after["macro_f1"], before["macro_f1"])

    def test_check_flags_graded_x_only(self):
        codes = np.full(sub.N_FRAMES, X, dtype=np.int8)
        for a, b in sub.GRADED_BLOCKS:
            codes[a:b] = D
        self.assertEqual(sub.check_submission(codes), [])
        codes[20000] = X
        self.assertEqual(len(sub.check_submission(codes)), 1)

    @needs_official
    def test_round_trip(self):
        import tempfile
        codes = baseline("baseline_smooth50")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.csv")
            sub.write_submission(path, codes)
            self.assertTrue(np.array_equal(sub.read_submission(path), codes))


if __name__ == "__main__":
    unittest.main()
