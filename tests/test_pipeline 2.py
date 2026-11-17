"""Loaders, degrader, label filter and harness, on small synthetic data.

Nothing here reads the gated match. The ATD test writes a tiny fake file in the documented format.
"""

import dataclasses
import os
import tempfile
import unittest

import numpy as np

from mtah import harness, provenance
from mtah import metrics as m
from mtah.baselines import NearestPlayer, majority, nearest_team
from mtah.decoder import DecoderConfig, decode_samples, expand
from mtah.degrade import DegradeConfig, ball_mask, degrade, fill_ball
from mtah.features import attack_sign, build_features, contact_runs, swap_teams
from mtah.labels import controlled
from mtah.loaders import gated
from mtah.loaders.metrica import labels_from_events
from mtah.match import A, AWAY, D, H, HOME, Match, X, load_cache, save_cache
from mtah.model import spell_lengths
from mtah import submission as sub
from mtah.smart_tagging import halves_without_kickoff, tag_prior


def toy_match(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    ball = np.cumsum(rng.normal(0, 0.2, (n, 2)), axis=0).astype(np.float32)
    players = (ball[:, None, :] + rng.normal(0, 8, (1, 22, 2))).astype(np.float32)
    team = np.array([HOME] * 11 + [AWAY] * 11, dtype=np.int8)
    labels = np.repeat(np.array([H, A, D, H, A, H], dtype=np.int8), n // 6)
    return Match("TOY", "metrica", 25, (105.0, 68.0), ball, players, team,
                 [(0, n // 2), (n // 2, n)], labels)


STATS = {"ball_gap_frames": [50, 100, 300], "ball_run_frames": [200, 400]}


class Format(unittest.TestCase):
    def test_validate_catches_bad_shapes(self):
        x = toy_match()
        bad = dataclasses.replace(x, team=x.team[:-1])
        with self.assertRaises(ValueError):
            bad.validate()

    def test_cache_round_trip(self):
        x = toy_match()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.npz")
            save_cache(x, path)
            y = load_cache(path)
        self.assertTrue(np.array_equal(x.labels, y.labels))
        self.assertEqual(x.halves, y.halves)


class EventLabels(unittest.TestCase):
    def ev(self, team, kind, start, end=None, sub=""):
        return {"team": team, "type": kind, "subtype": sub, "half": 0, "start": start,
                "end": start if end is None else end}

    def test_ball_out_is_dead_until_set_piece(self):
        events = [self.ev(H, "SET PIECE", 10), self.ev(H, "PASS", 10, 30),
                  self.ev(H, "BALL OUT", 40, 50), self.ev(A, "SET PIECE", 90),
                  self.ev(A, "PASS", 90, 120)]
        got = labels_from_events(events, 200, [(0, 200)])
        self.assertTrue((got[:10] == X).all())
        self.assertTrue((got[10:50] == H).all())
        self.assertTrue((got[50:90] == D).all())
        self.assertTrue((got[90:120] == A).all())
        self.assertTrue((got[120:] == X).all())

    def test_carry_during_stoppage_does_not_restart_play(self):
        events = [self.ev(H, "PASS", 0, 10), self.ev(H, "BALL OUT", 20, 30),
                  self.ev(A, "CARRY", 40, 60), self.ev(A, "SET PIECE", 80), self.ev(A, "PASS", 80, 90)]
        got = labels_from_events(events, 100, [(0, 100)])
        self.assertTrue((got[30:80] == D).all())

    def test_challenge_keeps_possession(self):
        events = [self.ev(H, "PASS", 0, 10), self.ev(A, "CHALLENGE", 20), self.ev(H, "PASS", 30, 40)]
        got = labels_from_events(events, 50, [(0, 50)])
        self.assertTrue((got[:40] == H).all())

    def test_goal_ends_play(self):
        events = [self.ev(H, "SHOT", 10, 20, "ON TARGET-GOAL"), self.ev(A, "SET PIECE", 60),
                  self.ev(A, "PASS", 60, 70)]
        got = labels_from_events(events, 80, [(0, 80)])
        self.assertTrue((got[20:60] == D).all())


class Control(unittest.TestCase):
    def test_short_flip_folds_back(self):
        labels = np.array([H] * 100 + [A] * 10 + [H] * 100, dtype=np.int8)
        self.assertTrue((controlled(labels, 1.0) == H).all())

    def test_long_spell_survives(self):
        labels = np.array([H] * 100 + [A] * 40 + [H] * 100, dtype=np.int8)
        self.assertTrue(np.array_equal(controlled(labels, 1.0), labels))

    def test_short_spell_next_to_stoppage_survives(self):
        labels = np.array([D] * 50 + [A] * 10 + [H] * 100, dtype=np.int8)
        self.assertTrue(np.array_equal(controlled(labels, 1.0), labels))

    def test_ping_pong_collapses(self):
        labels = np.array([H] * 100 + [A] * 5 + [H] * 5 + [A] * 5 + [H] * 100, dtype=np.int8)
        self.assertTrue((controlled(labels, 1.0) == H).all())

    def test_never_crosses_x(self):
        labels = np.array([H] * 50 + [X] * 10 + [A] * 10 + [H] * 50, dtype=np.int8)
        self.assertTrue(np.array_equal(controlled(labels, 1.0), labels))


class ContactRuns(unittest.TestCase):
    def test_sequence_of_teams_survives_a_gap(self):
        vote = np.array([2, 2, 2, np.nan, np.nan, -2, -2, 2, 2], dtype=float)
        near, speed = np.ones(9), np.zeros(9)
        start, stop, votes, size = contact_runs(vote, near, speed)
        self.assertEqual(start.tolist(), [0, 5, 7])
        self.assertEqual(stop.tolist(), [2, 6, 8])
        self.assertEqual(np.sign(votes).tolist(), [1, -1, 1])
        self.assertEqual(size.tolist(), [3, 2, 2])

    def test_one_stray_sample_does_not_start_a_run(self):
        vote = np.array([2, 2, -2, 2, 2], dtype=float)
        start, stop, votes, size = contact_runs(vote, np.ones(5), np.zeros(5))
        self.assertEqual(len(start), 1)
        self.assertEqual(size.tolist(), [4])

    def test_fast_or_distant_ball_is_not_contact(self):
        vote = np.full(4, 2.0)
        self.assertEqual(len(contact_runs(vote, np.full(4, 9.0), np.zeros(4))[0]), 0)
        self.assertEqual(len(contact_runs(vote, np.ones(4), np.full(4, 20.0))[0]), 0)


class Weights(unittest.TestCase):
    def test_every_frame_gets_its_run_length(self):
        labels = np.array([H, H, H, A, D, D], dtype=np.int8)
        self.assertEqual(spell_lengths(labels).tolist(), [3, 3, 3, 1, 2, 2])


class Degrader(unittest.TestCase):
    def test_deterministic_and_seed_sensitive(self):
        x = toy_match()
        a, b = degrade(x, seed=1, stats=STATS), degrade(x, seed=1, stats=STATS)
        c = degrade(x, seed=2, stats=STATS)
        self.assertTrue(np.array_equal(np.nan_to_num(a.players), np.nan_to_num(b.players)))
        self.assertFalse(np.array_equal(np.isnan(a.ball), np.isnan(c.ball)))

    def test_labels_and_source_untouched(self):
        x = toy_match()
        before = x.ball.copy()
        y = degrade(x, stats=STATS)
        self.assertTrue(np.array_equal(y.labels, x.labels))
        self.assertTrue(np.array_equal(x.ball, before))
        self.assertEqual(y.players.shape[1], 46)

    def test_visibility_is_independent_of_labels(self):
        x = toy_match(60000)
        cfg = DegradeConfig(dead_hide_prob=0.0, gap_keep=1.0)
        y = degrade(x, cfg, stats=STATS)
        seen = np.isfinite(y.ball[:, 0])
        rates = [seen[x.labels == k].mean() for k in (H, A, D)]
        self.assertLess(max(rates) - min(rates), 0.08)

    def test_dead_hide_links_blackouts_to_stoppages(self):
        x = dataclasses.replace(toy_match(60000), labels=np.tile(
            np.r_[[H] * 500, [D] * 500, [A] * 500, [D] * 500], 30).astype(np.int8))
        cfg = DegradeConfig(dead_hide_prob=0.8, gap_keep=0.4)
        y = degrade(x, cfg, stats=STATS)
        seen = np.isfinite(y.ball[:, 0])
        self.assertLess(seen[x.labels == D].mean(), seen[x.labels != D].mean() - 0.2)

    def test_stoppage_blackouts_do_not_mark_the_true_boundary(self):
        labels = np.tile(np.r_[[H] * 1000, [D] * 500], 40).astype(np.int8)
        x = dataclasses.replace(toy_match(60000), labels=labels)
        cfg = DegradeConfig(dead_hide_prob=1.0, gap_keep=0.0)
        seen = np.isfinite(degrade(x, cfg, stats=STATS).ball[:, 0])
        edges = np.flatnonzero(seen[1:] != seen[:-1]) + 1
        true_edges = set(np.flatnonzero(labels[1:] != labels[:-1]) + 1)
        self.assertLess(len(true_edges & set(edges.tolist())), 4)

    def test_dead_ball_shortcut_is_closed(self):
        x = toy_match(60000)
        ball = x.ball.copy()
        ball[x.labels == D] = np.nan
        y = degrade(dataclasses.replace(x, ball=ball), DegradeConfig(dead_hide_prob=0.0), stats=STATS)
        self.assertGreater(np.isfinite(y.ball[x.labels == D, 0]).mean(), 0.3)

    def test_identity_is_not_preserved(self):
        x = toy_match(60000)
        y = degrade(x, stats=STATS)
        slot_presence = np.isfinite(y.players[..., 0]).mean(0)
        self.assertGreater((slot_presence > 0).sum(), 22)

    def test_mask_uses_supplied_lengths(self):
        mask = ball_mask(100000, np.array([100]), np.array([100]), np.random.default_rng(0))
        self.assertAlmostEqual(mask.mean(), 0.5, delta=0.02)

    def test_fill_ball_interpolates(self):
        b = np.array([[0, 0], [np.nan, np.nan], [2, 2]], dtype=np.float32)
        self.assertEqual(fill_ball(b)[1, 0], 1.0)


class Baseline(unittest.TestCase):
    def test_nearest_team(self):
        x = toy_match(100)
        x.players[:, 0] = x.ball
        x.players[:, 12] = x.ball + 50
        self.assertTrue((nearest_team(x) == HOME).all())

    def test_missing_ball_gives_no_answer(self):
        x = toy_match(100)
        x.ball[5] = np.nan
        self.assertEqual(nearest_team(x)[5], -1)

    def test_majority_matches_published_rules(self):
        codes = np.array([0, 0, 1, 0, 0, -1, -1, -1, -1], dtype=np.int8)
        self.assertEqual(majority(codes, 1)[2], H)
        self.assertEqual(majority(codes, 1)[7], D)


class Decoding(unittest.TestCase):
    def test_viterbi_ignores_a_blip_but_keeps_a_real_change(self):
        prob = np.tile([0.8, 0.1, 0.1], (100, 1))
        prob[40:42] = [0.1, 0.8, 0.1]
        prob[70:] = [0.1, 0.8, 0.1]
        path = decode_samples(prob, np.zeros(100, dtype=np.int8), DecoderConfig(turnover=6.0))
        self.assertTrue((path[:65] == H).all())
        self.assertTrue((path[75:] == A).all())

    def test_halves_are_decoded_independently(self):
        prob = np.tile([0.9, 0.05, 0.05], (20, 1))
        prob[10:] = [0.05, 0.9, 0.05]
        half = np.r_[np.zeros(10, dtype=np.int8), np.ones(10, dtype=np.int8)]
        path = decode_samples(prob, half, DecoderConfig(turnover=1000.0))
        self.assertEqual(path[9], H)
        self.assertEqual(path[10], A)

    def test_expand_covers_each_half_exactly(self):
        frames = np.array([2, 7, 12, 17])
        half = np.zeros(4, dtype=np.int8)
        out = expand(np.array([H, H, A, A], dtype=np.int8), frames, half, [(0, 20)], 25, default=X)
        self.assertEqual(out[:20].tolist(), [H] * 10 + [A] * 10)
        self.assertTrue((out[20:] == X).all())


class Features(unittest.TestCase):
    def test_swap_mirrors_team_features(self):
        x = degrade(toy_match(3000), stats=STATS)
        a, names, _, _ = build_features(x)
        b, _, _, _ = build_features(swap_teams(x))
        i = names.index
        self.assertTrue(np.allclose(a[:, i("a_n")], b[:, i("b_n")]))
        ok = np.isfinite(a[:, i("vote")])
        self.assertTrue(np.allclose(a[ok, i("vote")], -b[ok, i("vote")]))

    def test_no_label_input(self):
        x = toy_match(3000)
        a = build_features(x)[0]
        b = build_features(dataclasses.replace(x, labels=None))[0]
        self.assertTrue(np.array_equal(a, b, equal_nan=True))

    def test_home_always_attacks_plus_x(self):
        x = toy_match(3000)
        x.players[:, :11, 0] -= 20
        x.players[:, 11:, 0] += 20
        self.assertEqual(attack_sign(x, 0, 1500), 1.0)
        flipped = dataclasses.replace(x, players=x.players * np.array([-1, 1], dtype=np.float32))
        self.assertEqual(attack_sign(flipped, 0, 1500), -1.0)


class Tags(unittest.TestCase):
    def setUp(self):
        self.frames = np.arange(2, 5000, 5)
        self.half = np.zeros(len(self.frames), dtype=np.int8)

    def tag(self, code, start, stop, team="Barcelona", kind=None):
        tags = {"Team": team, "Half": "1st Half"}
        if kind:
            tags["Type"] = kind
        return {"code": code, "start": start, "stop": stop, "tags": tags}

    def test_attacking_phase_favours_its_team(self):
        p = tag_prior([self.tag("BUILD UP", 500, 1000)], self.frames, self.half, [(0, 5000)])
        inside = (self.frames >= 500) & (self.frames < 1000)
        self.assertTrue((p[inside, H] > p[inside, A]).all())

    def test_defending_phase_favours_the_other_team(self):
        p = tag_prior([self.tag("DEFENDING IN MIDDLE THIRD", 500, 1000)], self.frames, self.half,
                      [(0, 5000)])
        inside = (self.frames >= 500) & (self.frames < 1000)
        self.assertTrue((p[inside, A] > p[inside, H]).all())

    def test_case_and_spelling_of_the_data_are_handled(self):
        p = tag_prior([self.tag("Build Up", 500, 1000, team="Atletic Club")], self.frames, self.half,
                      [(0, 5000)])
        inside = (self.frames >= 500) & (self.frames < 1000)
        self.assertTrue((p[inside, A] > p[inside, H]).all())

    def test_na_team_gives_no_team_evidence(self):
        p = tag_prior([self.tag("SET PIECES", 500, 625, team="N/A", kind="Throw-in")], self.frames,
                      self.half, [(0, 5000)])
        self.assertTrue(np.array_equal(p[:, H], p[:, A]))

    def test_everything_before_the_kickoff_tag_is_dead(self):
        p = tag_prior([self.tag("SET PIECES", 2000, 2125, kind="Kick Off (Start)")], self.frames,
                      self.half, [(0, 5000)])
        self.assertGreater(p[self.frames < 2000, D].min(), 10)
        self.assertLess(p[self.frames > 2200, D].max(), 1)


class OverlappingTags(unittest.TestCase):
    FRAMES = np.arange(2, 1000, 5)
    HALF = np.zeros(len(FRAMES), dtype=np.int8)
    PHASE = {"code": "PROGRESSION", "start": 100, "stop": 400, "tags": {"Team": "Barcelona"}}
    RESTART = {"code": "SET PIECES", "start": 500, "stop": 625, "tags": {"Team": "Barcelona", "Type": "Throw-in"}}

    def prior(self, tags):
        return tag_prior(tags, self.FRAMES, self.HALF, [(0, 1000)])

    def test_repeated_tags_count_once(self):
        for tag in (self.PHASE, self.RESTART):
            self.assertTrue(np.array_equal(self.prior([tag] * 5), self.prior([tag])))

    def test_different_tags_on_one_team_do_not_stack(self):
        other = dict(self.PHASE, code="BALL IN FINAL THIRD")
        self.assertTrue(np.array_equal(self.prior([self.PHASE, other]), self.prior([self.PHASE])))

    def test_opposing_tags_cancel(self):
        rival = dict(self.PHASE, tags={"Team": "Atletic Club"})
        prior = self.prior([self.PHASE, rival])
        self.assertTrue(np.array_equal(prior[:, H], prior[:, A]))


class TrackingComparison(unittest.TestCase):
    @staticmethod
    def module():
        import importlib.util
        path = os.path.join(gated.PROJECT, "scripts", "compare_tracking.py")
        spec = importlib.util.spec_from_file_location("compare_tracking", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_each_position_has_its_word(self):
        position = self.module().position
        self.assertTrue(position(0, 1, 2, 3, 4).startswith("BELOW"))
        self.assertEqual(position(1.5, 1, 2, 3, 4), "inside, low end")
        self.assertEqual(position(2.5, 1, 2, 3, 4), "inside, middle")
        self.assertEqual(position(3.5, 1, 2, 3, 4), "inside, high end")
        self.assertTrue(position(5, 1, 2, 3, 4).startswith("ABOVE"))

    def test_the_report_holds_no_numbers(self):
        module = self.module()
        measured = {key: 123.456 for key in module.NAMES}
        ranges = {key: [1, 2, 3, 4] for key in module.NAMES}
        text = " ".join(verdict for _, verdict in module.report(measured, ranges))
        self.assertFalse(any(c.isdigit() for c in text))

    def test_every_measurement_has_a_range(self):
        import json
        module = self.module()
        with open(module.RANGES) as handle:
            ranges = json.load(handle)["ranges"]
        self.assertEqual(set(module.NAMES), set(ranges))


class SameBitsEverywhere(unittest.TestCase):
    def test_length_is_exact_where_it_can_be_and_keeps_the_type(self):
        from mtah.features import _length
        x = np.array([3.0, np.nan, 0.0], dtype=np.float32)
        y = np.array([4.0, 1.0, 0.0], dtype=np.float32)
        out = _length(x, y)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out[0], 5.0)
        self.assertTrue(np.isnan(out[1]))
        self.assertEqual(out[2], 0.0)

    def test_uniform_draws_stay_in_range_and_repeat(self):
        from mtah.degrade import _uniform, random_config
        a = [_uniform(np.random.default_rng(5), 2.0, 8.0) for _ in range(3)]
        self.assertTrue(all(2.0 <= v < 8.0 for v in a))
        self.assertEqual(len(set(a)), 1)
        self.assertEqual(random_config(np.random.default_rng(9)), random_config(np.random.default_rng(9)))


class OpenFileCheck(unittest.TestCase):
    def test_reports_missing_and_changed_files(self):
        import hashlib
        import json
        from unittest import mock
        from mtah import datasets as ds
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "dfl"))
            for name, body in (("dfl/a.xml", b"a"), ("dfl/b.xml", b"b")):
                with open(os.path.join(root, name), "wb") as handle:
                    handle.write(body)
            listing = os.path.join(root, "sums.json")
            with open(listing, "w") as handle:
                json.dump({"dfl/a.xml": hashlib.sha256(b"a").hexdigest(),
                           "dfl/b.xml": hashlib.sha256(b"not b").hexdigest(),
                           "dfl/c.xml": hashlib.sha256(b"c").hexdigest()}, handle)
            with mock.patch.object(ds, "OPEN_DIR", root), mock.patch.object(ds, "OPEN_FILES", listing):
                self.assertEqual(ds.check_open_files(), ["changed dfl/b.xml", "missing dfl/c.xml"])

    def test_listing_covers_every_training_match(self):
        import json
        from mtah import datasets as ds
        with open(ds.OPEN_FILES) as handle:
            names = set(json.load(handle))
        for match_id in ds.DFL_IDS:
            for kind in ("02_01_matchinformation", "03_02_events_raw", "04_03_positions_raw_observed"):
                self.assertTrue(any(n.startswith(f"dfl/DFL_{kind}_") and n.endswith(f"{match_id}.xml")
                                    for n in names), (match_id, kind))
        self.assertEqual(sum(n.startswith("metrica/") for n in names), 10)


class Manifest(unittest.TestCase):
    def test_records_source_settings_and_environment(self):
        record = provenance.manifest(note="x")
        self.assertEqual(len(record["source_sha256"]), 64)
        self.assertEqual(record["note"], "x")
        self.assertIn("decoder", record["settings"])
        self.assertIn("scikit_learn", record["environment"])

    def test_source_hash_is_stable(self):
        self.assertEqual(provenance.source_hash(), provenance.source_hash())

    def test_it_is_plain_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "m.json")
            provenance.save_manifest(path, provenance.manifest())
            self.assertTrue(os.path.getsize(path) > 100)


class Kickoffs(unittest.TestCase):
    KICK = {"code": "SET PIECES", "stop": 0, "tags": {"Type": "Kick Off (Start)"}}

    def test_a_half_with_no_kickoff_is_named(self):
        tags = [dict(self.KICK, start=95)]
        self.assertEqual(halves_without_kickoff(tags, [(100, 500), (1000, 1500)], 50), [1])

    def test_case_is_ignored(self):
        tag = {"code": "set pieces", "stop": 0, "start": 1100, "tags": {"Type": "KICK OFF (START)"}}
        self.assertEqual(halves_without_kickoff([tag], [(1000, 1500)]), [])

    def test_other_restarts_do_not_count(self):
        tag = {"code": "SET PIECES", "stop": 0, "start": 1100, "tags": {"Type": "Kick Off (After Goal)"}}
        self.assertEqual(halves_without_kickoff([tag], [(1000, 1500)]), [0])


class Submission(unittest.TestCase):
    def good(self):
        codes = np.full(sub.N_FRAMES, X, dtype=np.int8)
        for a, b in sub.GRADED_BLOCKS:
            codes[a:b] = H
        return codes

    def test_good_array_passes(self):
        self.assertEqual(sub.check_submission(self.good()), [])

    def test_invalid_codes_are_caught(self):
        for code in (9, -1, 4):
            bad = self.good()
            bad[20000] = code
            self.assertEqual(sub.check_submission(bad), ["1 invalid label codes"])
            with self.assertRaises(ValueError):
                sub.write_submission(os.devnull, bad)

    def test_written_file_reads_back(self):
        codes = self.good()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.csv")
            sub.write_submission(path, codes)
            self.assertEqual(sub.check_file(path, codes), [])
            changed = codes.copy()
            changed[20000] = A
            self.assertEqual(sub.check_file(path, changed), ["written file differs from the predictions"])

    def test_damaged_file_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.csv")
            with open(path, "w") as f:
                f.write("frame,label\n0,Q\n")
            self.assertEqual(len(sub.check_file(path, self.good())), 1)
            self.assertEqual(len(sub.check_file(os.path.join(tmp, "absent.csv"), self.good())), 1)


class Harness(unittest.TestCase):
    def test_predictor_never_sees_labels(self):
        seen = {}

        class Spy:
            name = "spy"

            def fit(self, matches):
                pass

            def predict(self, match):
                seen["labels"] = match.labels
                return np.full(match.n_frames, D, dtype=np.int8)

        original = harness.prepared
        harness.prepared = lambda *a, **k: toy_match()
        try:
            harness.run_lomo(Spy, ["A", "B"], needs_training=False)
        finally:
            harness.prepared = original
        self.assertIsNone(seen["labels"])

    def test_training_excludes_the_held_out_match_and_uses_another_seed(self):
        calls = []

        class Spy:
            name = "spy"

            def fit(self, matches):
                calls.append(("fit", len(matches)))

            def predict(self, match):
                return np.full(match.n_frames, D, dtype=np.int8)

        def fake(match_id, cfg, seed, stats):
            calls.append((match_id, seed))
            return toy_match()

        original = harness.prepared
        harness.prepared = fake
        try:
            harness.run_lomo(Spy, ["A", "B", "C"], seed=5)
        finally:
            harness.prepared = original
        self.assertEqual(calls[:4], [("B", 5), ("C", 5), ("fit", 2), ("A", 5 + harness.TEST_SEED_OFFSET)])

    def test_compare_calls_small_gaps_noise(self):
        a = [{"match": str(i), "spell_f1": 0.40} for i in range(8)]
        b = [{"match": str(i), "spell_f1": 0.42} for i in range(8)]
        self.assertEqual(harness.compare(a, b)["verdict"], "within noise")
        c = [{"match": str(i), "spell_f1": 0.50} for i in range(8)]
        self.assertEqual(harness.compare(a, c)["verdict"], "real")


class GatedGuards(unittest.TestCase):
    def test_refuses_data_inside_workspace(self):
        inside = os.path.join(gated.PROJECT, "data")
        with self.assertRaises(RuntimeError):
            gated.data_dir(inside)
        with self.assertRaises(RuntimeError):
            gated.data_dir(os.path.join(os.path.dirname(gated.PROJECT), "data"))

    def test_accepts_data_outside_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(gated.data_dir(tmp), os.path.realpath(tmp))

    def test_requires_a_location(self):
        previous = os.environ.pop("MTAH_DATA_DIR", None)
        try:
            with self.assertRaises(RuntimeError):
                gated.data_dir()
        finally:
            if previous is not None:
                os.environ["MTAH_DATA_DIR"] = previous


class FakeAtd(unittest.TestCase):
    """A made-up file in the format DATA.md documents. No real data."""

    META = ('<root><Teams><Team id="T1"/><Team id="T2"/></Teams><Players>'
            '<Player id="a" teamId="T1"/><Player id="b" teamId="T2"/><Player id="c" teamId="T2"/>'
            '</Players></root>')
    RAW = ("1:0.5,0.5,1.0;NaN,NaN,NaN;0.25,0.75,2.0:0.5,0.5,3.0\n"
           "2:0.6,0.5,1.0;0.1,0.1,1.0;NaN,NaN,NaN:NaN,NaN,NaN\n"
           "3:NaN,NaN,NaN;NaN,NaN,NaN;NaN,NaN,NaN:NaN,NaN,NaN\n")

    def test_parse_alignment_and_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta, raw = os.path.join(tmp, "m.xml"), os.path.join(tmp, "r.txt")
            with open(meta, "w") as f:
                f.write(self.META)
            with open(raw, "w") as f:
                f.write(self.RAW)
            x = gated.load_atd_match(meta_path=meta, raw_path=raw, n_frames=3, halves=[(0, 3)],
                                     check=False)
        self.assertEqual(x.team.tolist(), [HOME, AWAY, AWAY])
        self.assertAlmostEqual(float(x.ball[0, 0]), 0.0)
        self.assertTrue(np.isnan(x.ball[1, 0]))
        self.assertTrue(np.isnan(x.players[0, 1, 0]))
        self.assertAlmostEqual(float(x.players[0, 2, 0]), -0.25 * 105.3, places=3)
        self.assertAlmostEqual(float(x.players[0, 2, 1]), -0.25 * 68.0, places=3)
        self.assertTrue(np.isnan(x.ball[2, 0]))

    def parse(self, text, slots=3, frames=3):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.txt")
            with open(path, "w") as f:
                f.write(text)
            return gated.parse_atd(path, slots, frames)

    def refused(self, text, expected, **kwargs):
        with self.assertRaises(gated.BadTracking) as caught:
            self.parse(text, **kwargs)
        self.assertIn(expected, str(caught.exception))
        return str(caught.exception)

    def test_a_good_file_with_a_blank_line_passes(self):
        ball, players = self.parse(self.RAW + "\n")
        self.assertEqual(int(np.isfinite(ball[:, 0]).sum()), 1)

    def test_empty_file_is_refused(self):
        self.refused("", "3 frames missing")

    def test_duplicate_frame_is_refused(self):
        self.refused(self.RAW + self.RAW.splitlines()[0] + "\n", "1 duplicate frames")

    def test_missing_frame_is_refused(self):
        self.refused("".join(self.RAW.splitlines(True)[:2]), "1 frames missing")

    def test_out_of_range_frame_is_refused(self):
        self.refused(self.RAW + "9:NaN,NaN,NaN;NaN,NaN,NaN;NaN,NaN,NaN:NaN,NaN,NaN\n", "1 frames out of range")

    def test_wrong_slot_count_is_refused(self):
        self.refused(self.RAW, "3 lines with the wrong number of slots", slots=4)

    def test_malformed_lines_are_refused_without_quoting_them(self):
        secret = "0.12345,garbage"
        message = self.refused(self.RAW + f"4|{secret}\n", "1 malformed lines")
        self.assertNotIn(secret, message)
        message = self.refused(self.RAW + f"x:{secret}:{secret}\n", "1 malformed lines")
        self.assertNotIn(secret, message)

    def test_a_slot_on_neither_team_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = os.path.join(tmp, "m.xml")
            with open(meta, "w") as f:
                f.write(self.META.replace('teamId="T2"/><Player id="c"', 'teamId="T9"/><Player id="c"'))
            with self.assertRaises(gated.BadTracking):
                gated.track_teams(meta)

    def tracking(self, n=1000):
        rng = np.random.default_rng(0)
        team = np.array([HOME] * 11 + [AWAY] * 11, dtype=np.int8)
        ball = rng.uniform(-30, 30, (n, 2)).astype(np.float32)
        ball[: int(0.4 * n)] = np.nan
        return Match("T", "atd", 25, gated.PITCH, ball,
                     rng.uniform(-30, 30, (n, 22, 2)).astype(np.float32), team, [(0, n)])

    def test_believable_tracking_passes(self):
        gated.check_tracking(self.tracking())

    def test_no_ball_is_refused(self):
        x = self.tracking()
        x.ball[:] = np.nan
        with self.assertRaises(gated.BadTracking):
            gated.check_tracking(x)

    def test_too_few_objects_are_refused(self):
        x = self.tracking()
        x.players[:, 3:] = np.nan
        with self.assertRaises(gated.BadTracking):
            gated.check_tracking(x)

    def test_one_team_missing_is_refused(self):
        x = dataclasses.replace(self.tracking(), team=np.zeros(22, dtype=np.int8))
        with self.assertRaises(gated.BadTracking):
            gated.check_tracking(x)

    def test_wrong_units_are_refused(self):
        x = self.tracking()
        x.players[:] *= 100
        with self.assertRaises(gated.BadTracking):
            gated.check_tracking(x)

    def test_summary_reports_only_aggregates(self):
        import importlib.util
        path = os.path.join(gated.PROJECT, "scripts", "atd_summary.py")
        spec = importlib.util.spec_from_file_location("atd_summary", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.summarise(toy_match())
        self.assertTrue(all(isinstance(v, (int, float)) for v in result.values()))
        self.assertAlmostEqual(result["objects_per_graded_frame"], 22.0)

    def test_smart_tagging_times_become_frames(self):
        xml = ("<file><ALL_INSTANCES><instance><ID>1</ID><start>2.0</start><end>4.0</end>"
               "<code>SET PIECES</code><label><group>Team</group><text>N/A</text></label>"
               "</instance></ALL_INSTANCES></file>")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.xml")
            with open(path, "w") as f:
                f.write(xml)
            got = gated.load_smart_tagging(path)
        self.assertEqual(got[0]["start"], 50)
        self.assertEqual(got[0]["stop"], 100)
        self.assertEqual(got[0]["tags"]["Team"], "N/A")


if __name__ == "__main__":
    unittest.main()
