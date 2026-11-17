"""The frozen narrow-band detector and the two-soundtrack whistle check, on made-up sound only."""

import unittest

import numpy as np

from mtah import narrowband as nb
from mtah import whistle_check as wc
from mtah.align import NO_SHIFT
from mtah.fakestadium import Scene, make, second_microphone
from mtah.match import A, D, H

SR = nb.SR
FPS = 25


def tone(seconds, pitch):
    t = np.arange(int(seconds * SR)) / SR
    return np.sin(2 * np.pi * pitch * t)


class NarrowBand(unittest.TestCase):
    def test_spectrum_does_not_depend_on_chunk_size(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((SR * 3 + 777, 2)).astype(np.float32)
        whole = nb.spectrum([x])
        pieces = nb.spectrum([x[i:i + 12345] for i in range(0, len(x), 12345)])
        self.assertEqual(whole.shape, pieces.shape)
        np.testing.assert_allclose(whole, pieces, rtol=1e-4, atol=1e-6)
        self.assertEqual(len(whole), (len(x) - nb.WIN) // nb.HOP + 1)

    def test_learns_the_pitch_and_finds_the_blasts(self):
        rng = np.random.default_rng(1)
        seconds, pitch = 120, 3700.0
        x = 0.3 * rng.standard_normal(SR * seconds)
        stops = np.arange(5.0, seconds - 5, 10.0)
        for t in stops:
            a = int((t + 0.2) * SR)
            x[a:a + int(0.4 * SR)] += 0.6 * tone(0.4, pitch)
        S = nb.spectrum([x[:, None].repeat(2, axis=1)])
        quiet = np.ones(len(S), dtype=bool)
        quiet[nb.frames_at(stops, len(S))] = False
        found_pitch, line = nb.learn_line(S, stops, quiet)
        self.assertLess(abs(found_pitch / pitch - 1), nb.CENTRE)
        self.assertGreater(line, nb.Settings().min_line_db)
        starts = np.array([e["start"] for e in nb.find(nb.contrast(S, found_pitch))])
        self.assertTrue(nb.coincide(stops + 0.2, starts, 0.1).all())
        self.assertEqual(len(starts), len(stops))

    def test_nothing_to_learn(self):
        S = np.ones((100, len(nb.FB)), dtype=np.float32)
        pitch, line = nb.learn_line(S, [], np.ones(100, dtype=bool))
        self.assertTrue(np.isnan(pitch))
        self.assertEqual(line, float("-inf"))

    def test_coincide_and_chance(self):
        self.assertEqual(nb.coincide([1.0, 5.0, 9.0], [1.2, 9.5], 0.3).tolist(), [True, False, False])
        self.assertEqual(nb.coincide([1.0], [], 0.3).tolist(), [False])
        self.assertEqual(nb.coincide([2.0], [1.9], 0.3).tolist(), [True])
        a = np.arange(0, 3000, 37.0)
        self.assertEqual(nb.coincide(a, a).mean(), 1.0)
        self.assertLess(nb.chance_coincide(a, a), 0.2)
        self.assertTrue(nb.same_pitch(4000, 4050))
        self.assertFalse(nb.same_pitch(4000, 4100))
        self.assertFalse(nb.same_pitch(float("nan"), 4000))


class Pieces(unittest.TestCase):
    def test_stoppages_and_quiet(self):
        labels = np.full(3000, H, dtype=np.int64)
        labels[1000:1100] = D          # 4 s dead: a stoppage
        labels[2000:2020] = D          # under 2 s: not one
        labels[1500:1600] = A
        stops = wc.stoppages(labels, ((0, 1500), (1500, 3000)))
        self.assertEqual(stops[0].tolist(), [1000])
        self.assertEqual(stops[1].tolist(), [])
        quiet = wc.quiet_frames(labels, ((0, 3000),))
        self.assertFalse(quiet[1000 - 249:1100 + 249].any())
        self.assertTrue(quiet[500] and quiet[1500 + 100])

    def test_broadcast_track_mapping(self):
        n_frames = 1000
        shift = np.full(n_frames, 300, dtype=np.int32)
        shift[:50] = NO_SHIFT
        S = np.zeros((int((n_frames / FPS + 1) / nb.DT), len(nb.FB)), dtype=np.float32)
        track = wc.Track.broadcast("b", S, 0.5, shift, NO_SHIFT, 2000)
        j = int(round((100 / FPS + 0.5) / nb.DT))             # broadcast frame 100
        self.assertEqual(track.to_tactical[j], 400)
        self.assertAlmostEqual(track.from_tactical[400], 100 / FPS + 0.5)
        self.assertTrue(np.isnan(track.from_tactical[300 + 10]))
        late = track.delayed(0.5)
        self.assertEqual(late.to_tactical[j + 20], 400)
        self.assertAlmostEqual(late.from_tactical[400], 100 / FPS + 1.0)

    def test_merge(self):
        self.assertEqual(wc.merge(np.array([1.0, 5.0]), np.array([1.1, 7.0])).tolist(), [1.0, 5.0, 7.0])

    def test_sound_link_finds_a_lag(self):
        rng = np.random.default_rng(2)
        n = int(300 / nb.DT)
        level = np.exp(rng.standard_normal(n + 200))
        S_a = np.repeat(level[100:100 + n, None], len(nb.FB), axis=1).astype(np.float32)
        lag_frames = 16                                       # 0.4 s
        S_b = np.repeat(level[100 - lag_frames:100 - lag_frames + n, None], len(nb.FB), axis=1).astype(np.float32)
        S_b *= np.exp(0.3 * rng.standard_normal(S_b.shape)).astype(np.float32)
        n_tac = int(290 * FPS)
        a = wc.Track.tactical("t", S_a, 0.0, n_tac)
        b = wc.Track.tactical("b", S_b, 0.0, n_tac)
        found = wc.sound_link(a, b, ((0, n_tac),))
        self.assertGreater(found["best_r"], 0.5)
        self.assertAlmostEqual(found["lag_s"], lag_frames * nb.DT, delta=0.05)


class TwoSoundtracks(unittest.TestCase):
    """A made-up match: two halves, the broadcast on its own timeline, the tactical sound fainter."""

    @classmethod
    def setUpClass(cls):
        seconds = 300.0
        scene = Scene(seconds=seconds, whistle_db=2.0, commentary_db=6.0, whistles_per_minute=4.0, seed=5)
        made = make(scene)                                    # the broadcast's sound, with commentators
        tac_sound = second_microphone(made, scene, whistle_db=0.0, seed=6)
        cls.pitch = made.pitch
        tac_offset, shift, bc_offset = 0.5, 200, 0.2          # frame 0 at 0.5 s; broadcast frame 0 = tactical frame 200
        n_tac = int((seconds - tac_offset) * FPS) - 1
        blocks = ((5 * FPS, 145 * FPS), (155 * FPS, 295 * FPS))
        rng = np.random.default_rng(7)
        labels = np.where((np.arange(n_tac) // (12 * FPS)) % 2 == 0, H, A)
        for t in made.onsets:
            k = int((t - tac_offset + rng.uniform(0.0, 0.4)) * FPS)
            labels[k:k + int(rng.uniform(4, 8) * FPS)] = D
        cut = int((tac_offset + shift / FPS - bc_offset) * SR)
        bc_sound = made.sound[cut:]
        n_bc = int((len(bc_sound) / SR - bc_offset) * FPS)
        shifts = np.full(n_bc, shift, dtype=np.int32)
        tactical = wc.Track.tactical("tactical", nb.spectrum([tac_sound]), tac_offset, n_tac)
        broadcast = wc.Track.broadcast("broadcast", nb.spectrum([bc_sound]), bc_offset, shifts, NO_SHIFT, n_tac)
        cls.out = wc.check(labels, blocks, tactical, broadcast)

    def test_pitch_learned_in_both(self):
        for name in ("tactical", "broadcast"):
            for half in self.out["tracks"][name]["halves"]:
                self.assertLess(abs(half["pitch"] / self.pitch - 1), 0.03, name)      # a pea whistle warbles
            self.assertTrue(self.out["tracks"][name]["on"], name)

    def test_whistles_heard_together(self):
        pair = self.out["pair"]
        self.assertGreater(pair["together"], 0.4)
        self.assertGreater(pair["together"], 10 * pair["together_chance"])
        self.assertGreater(pair["together_from_broadcast"], 0.6)

    def test_at_stoppages_far_above_chance(self):
        for name in ("tactical", "broadcast"):
            got = self.out["tracks"][name]
            self.assertGreaterEqual(got["at_stops"], 0.4, name)
            self.assertGreater(got["at_stops"], 5 * got["at_stops_chance"], name)
        both, either = self.out["pair"]["both"], self.out["pair"]["either"]
        self.assertGreater(both["at_stops"], 5 * both["at_stops_chance"])
        self.assertGreaterEqual(either["at_stops"], max(self.out["tracks"][n]["at_stops"] for n in ("tactical", "broadcast")))
        self.assertLessEqual(both["quiet_per_min"], min(self.out["tracks"][n]["quiet_per_min"]
                                                         for n in ("tactical", "broadcast")))

    def test_safe_whistles(self):
        pair, events = self.out["pair"], self.out["events"]
        self.assertTrue(any(pair["usable_halves"]))
        self.assertLess(pair["chance_pairs"], 0.5)
        self.assertGreaterEqual(pair["safe"]["count"], 1)
        self.assertTrue(set(events["safe"].tolist()) <= set(events["both"].tolist()))
        self.assertEqual(pair["safe"]["would_act"], pair["safe"]["count"])     # each sits at a stoppage start

    def test_independent_microphones_are_not_lag_corrected(self):
        self.assertEqual(self.out["lag_used_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
