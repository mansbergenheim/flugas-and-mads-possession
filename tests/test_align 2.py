"""The aligner on small made-up matches, where the true line-up is known. No real or open data."""

import os
import tempfile
import unittest

import numpy as np

from mtah.align import (LIVE, NO_SHIFT, NOT_COVERED, NOT_WIDE, REPLAY, UNSURE, AlignConfig, NoAlignment, align,
                        director_link, fit_distance, on_tactical_pitch, tactical_view)
from mtah.fakecast import CastConfig, fake_broadcast
from mtah.loaders import gated
from mtah.match import A, AWAY, D, H, HOME, Match

SMALL = AlignConfig(anchors=120)      # fewer clips, because a four-minute match offers few different moments


def small_match(seed=0, half=3000):
    """22 players who each wander on their own, so that no two moments look alike. Two halves."""
    rng = np.random.default_rng(seed)
    n = 2 * half
    ball = np.cumsum(rng.normal(0, 0.35, (n, 2)), axis=0)
    ball = (np.clip(ball, -40, 40) * np.array([1.0, 0.6])).astype(np.float32)
    home = np.column_stack((rng.uniform(-45, 45, 22), rng.uniform(-30, 30, 22)))
    wander = np.cumsum(rng.normal(0, 0.18, (n, 22, 2)), axis=0)
    wander -= np.linspace(0, 1, n)[:, None, None] * wander[-1]          # ends where it began
    players = (home[None] + wander + 0.3 * ball[:, None, :]).astype(np.float32)
    team = np.array([HOME] * 11 + [AWAY] * 11, dtype=np.int8)
    block = np.array([H, H, A, D, H, A, A, D, H, A, H, D], dtype=np.int8)
    labels = np.repeat(block, n // len(block))
    return Match(f"SMALL{seed}", "metrica", 25, (105.0, 68.0), ball, players, team, [(0, half), (half, n)], labels)


def tactical_side(match, seed=5):
    """The same match as a second camera's tracking would give it: its own small errors, a few players missing."""
    rng = np.random.default_rng(seed)
    players = match.players + rng.normal(0, 0.15, match.players.shape).astype(np.float32)
    players[rng.random(players.shape[:2]) < 0.05] = np.nan
    return players


def truth_shift(cast):
    return np.where(cast.live >= 0, cast.live - np.arange(len(cast.live)), NO_SHIFT)


class LiningUp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.match = small_match()
        cls.tactical = tactical_side(cls.match)
        cls.cast = fake_broadcast(cls.match, seed=1)
        cls.found = align(cls.tactical, cls.cast.players, cls.cast.valid, cls.match.pitch, cls.match.team, cls.cast.team,
                          SMALL)

    def test_both_halves_are_found_to_the_frame(self):
        truth = truth_shift(self.cast)
        both = (truth != NO_SHIFT) & (self.found.shift != NO_SHIFT)
        self.assertEqual(len(self.found.pieces), 2)
        self.assertGreater(both.sum() / (truth != NO_SHIFT).sum(), 0.99)
        self.assertTrue(np.array_equal(self.found.shift[both], truth[both]))

    def test_way_round_and_teams_are_reported(self):
        self.assertEqual(self.found.flip, (1, 1))
        self.assertTrue(self.found.same_teams)
        self.assertLess(self.found.fit_m, 1.0)

    def test_what_is_called_live_is_live(self):
        called = self.found.state == LIVE
        self.assertGreater((self.cast.state[called] == LIVE).mean(), 0.99)
        self.assertGreater(called[self.cast.state == LIVE].mean(), 0.9)

    def test_frames_without_a_pitch_transform_are_not_wide(self):
        self.assertTrue((self.found.state[~self.cast.valid] == NOT_WIDE).all())
        self.assertTrue((self.found.state[self.cast.valid] != NOT_WIDE).all())

    def test_replays_are_not_taken_for_live(self):
        replay = self.cast.state == REPLAY
        if replay.any():
            self.assertLess((self.found.state[replay] == LIVE).mean(), 0.1)

    def test_tactical_view_puts_the_broadcast_on_the_tactical_timeline(self):
        frame, state = tactical_view(self.found, self.match.n_frames)
        shown = np.flatnonzero(frame >= 0)
        on_air = self.cast.live[frame[shown]] >= 0           # the rest are studio shots, carried on in real time
        self.assertGreater(on_air.mean(), 0.9)
        self.assertTrue(np.array_equal(self.cast.live[frame[shown]][on_air], shown[on_air]))
        covered = np.zeros(self.match.n_frames, dtype=bool)
        covered[self.cast.live[self.cast.live >= 0]] = True
        self.assertTrue((frame[covered] >= 0).all())
        self.assertTrue((state[frame < 0] == NOT_COVERED).all())
        self.assertGreater((state == LIVE).mean(), 0.5)


class OtherBroadcasts(unittest.TestCase):
    def test_a_turned_broadcast_with_the_teams_swapped(self):
        match = small_match(2)
        cast = fake_broadcast(match, seed=1, flip=(-1, -1), swap_teams=True)
        found = align(tactical_side(match), cast.players, cast.valid, match.pitch, match.team, cast.team, SMALL)
        truth = truth_shift(cast)
        both = (truth != NO_SHIFT) & (found.shift != NO_SHIFT)
        self.assertEqual(found.flip, (-1, -1))
        self.assertIs(found.same_teams, False)
        self.assertTrue(np.array_equal(found.shift[both], truth[both]))

    def test_a_pitch_drawn_too_long_and_off_centre_is_put_right(self):
        match = small_match(3)
        cast = fake_broadcast(match, seed=1)
        skewed = cast.players.copy()
        skewed[:, :, 0] = skewed[:, :, 0] * 1.04 + 2.0
        found = align(tactical_side(match), skewed, cast.valid, match.pitch, cfg=SMALL)
        truth = truth_shift(cast)
        both = (truth != NO_SHIFT) & (found.shift != NO_SHIFT)
        self.assertTrue(np.array_equal(found.shift[both], truth[both]))
        back = on_tactical_pitch(found, np.array([[1.04 * 20.0 + 2.0, 5.0]], dtype=np.float32))
        self.assertLess(np.abs(back - np.array([[20.0, 5.0]])).max(), 0.5)

    def test_a_broadcast_with_no_wide_shots_is_refused(self):
        match = small_match(4)
        cast = fake_broadcast(match, seed=1)
        with self.assertRaises(NoAlignment):
            align(tactical_side(match), cast.players, np.zeros(len(cast.valid), dtype=bool), match.pitch, cfg=SMALL)

    def test_a_broadcast_of_another_match_is_refused(self):
        cast = fake_broadcast(small_match(6), seed=1)
        with self.assertRaises(NoAlignment):
            align(tactical_side(small_match(7)), cast.players, cast.valid, (105.0, 68.0), cfg=SMALL)

    def test_same_input_gives_the_same_answer(self):
        match = small_match(8)
        cast = fake_broadcast(match, seed=2)
        again = fake_broadcast(match, seed=2)
        self.assertTrue(np.array_equal(cast.live, again.live))
        one = align(tactical_side(match), cast.players, cast.valid, match.pitch, cfg=SMALL)
        two = align(tactical_side(match), again.players, again.valid, match.pitch, cfg=SMALL)
        self.assertTrue(np.array_equal(one.shift, two.shift))
        self.assertTrue(np.array_equal(one.state, two.state))


class Pieces(unittest.TestCase):
    def test_fit_distance_ignores_missing_players_and_a_stray_one(self):
        tactical = np.zeros((1, 6, 2), dtype=np.float32)
        tactical[0, :, 0] = np.arange(6) * 10.0
        broadcast = np.full((1, 5, 2), np.nan, dtype=np.float32)
        broadcast[0, :4] = 0.0
        broadcast[0, :4, 0] = np.array([0.0, 10.0, 20.0, 35.0]) + 0.5      # the last one is 4.5 m from anyone
        self.assertAlmostEqual(float(fit_distance(broadcast, tactical)[0]), 0.5, places=4)

    def test_fake_broadcast_cuts_away_in_stoppages(self):
        match = small_match(9)
        cast = fake_broadcast(match, seed=3, cfg=CastConfig(stoppage_cut=1.0, live_cut_every_s=1e9))
        covered = cast.live >= 0
        dead = np.zeros(len(cast.live), dtype=bool)
        dead[covered] = match.labels[cast.live[covered]] == D
        away = covered & (cast.state != LIVE)
        self.assertGreater(away.sum(), 0)
        self.assertGreater(dead[away].mean(), 0.7)


class BroadcastFiles(unittest.TestCase):
    def test_homography_file_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "h.csv")
            with open(path, "w") as f:
                f.write("frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n")
                f.write("0,1,0,0,0,1,0,0,0,1,0.25,1\n1,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,0\n2,2,0,0,0,2,0,0,0,1,7.5,1\n")
            valid, error, matrices = gated.load_homography(path, 3)
        self.assertEqual(valid.tolist(), [True, False, True])
        self.assertAlmostEqual(float(error[2]), 7.5)
        self.assertTrue(np.isnan(matrices[1]).all())
        self.assertEqual(matrices[2, 0, 0], 2.0)

    def test_homography_file_with_a_missing_frame_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "h.csv")
            with open(path, "w") as f:
                f.write("frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n0,1,0,0,0,1,0,0,0,1,0.25,1\n")
            with self.assertRaises(gated.BadTracking) as caught:
                gated.load_homography(path, 3)
        self.assertIn("frames missing", str(caught.exception))


class DirectorLink(unittest.TestCase):
    def setUp(self):
        n = 6000
        self.labels = np.full(n, H, dtype=np.int8)
        self.state = np.full(n, LIVE, dtype=np.uint8)
        for s in (1000, 3000, 5000):                       # three 10 s stoppages, each with a 4 s cut away
            self.labels[s:s + 250] = D
            self.state[s + 40:s + 140] = NOT_WIDE

    def test_cuts_inside_stoppages_are_seen(self):
        found = director_link(self.labels, self.state, [(0, 6000)])
        self.assertEqual(found["away_live"], 0.0)
        self.assertGreater(found["away_dead"], 0.3)
        self.assertEqual((found["cuts"], found["cuts_near_dead"], found["stops"]), (3, 1.0, 3))
        self.assertEqual((found["stops_with_cut"], found["back_before_restart"]), (1.0, 1.0))
        self.assertLess(found["chance_near_dead"], 0.3)

    def test_cuts_in_play_score_low_and_unsure_frames_are_left_out(self):
        state = np.roll(self.state, 1500)                   # the same cuts, now in open play
        state[:100] = UNSURE
        found = director_link(self.labels, state, [(0, 6000)])
        self.assertEqual(found["away_dead"], 0.0)
        self.assertEqual(found["cuts_near_dead"], 0.0)
        self.assertEqual(found["stops_with_cut"], 0.0)
        self.assertIsNone(found["back_before_restart"])

    def test_only_frames_inside_the_blocks_count(self):
        found = director_link(self.labels, self.state, [(0, 2000)])
        self.assertEqual((found["cuts"], found["stops"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
