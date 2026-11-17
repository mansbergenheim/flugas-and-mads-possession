import json
import os
import tempfile
import unittest
import wave

import numpy as np

from mtah import audio
from mtah.fakestadium import Scene, make
from mtah.whistle_eval import combine, judge_half, read_labels
from mtah.whistle import (EventRules, FeatureStream, combined_events, features, find_events, mixes, near_stops,
                          one_voice, pitch_groups, referee_whistles)

SR = 48000


def tone(seconds, pitch, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return np.sin(2 * np.pi * pitch * t)


def scene_with_tone(sr=SR, pitch=3000.0, at=1.0, length=0.3, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    sound = noise * rng.standard_normal((int(3 * sr), 2))
    a = int(at * sr)
    sound[a:a + int(length * sr), 0] += 0.5 * tone(length, pitch, sr)
    return sound


class Detector(unittest.TestCase):
    def test_finds_a_clean_tone_with_its_pitch_and_start(self):
        events = find_events(features(scene_with_tone(), SR)["mid"], SR)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]["pitch"], 3000.0, delta=30.0)
        self.assertAlmostEqual(events[0]["start"], 1.0, delta=0.04)
        self.assertAlmostEqual(events[0]["duration"], 0.3, delta=0.06)

    def test_works_at_a_lower_sample_rate(self):
        events = find_events(features(scene_with_tone(sr=16000), 16000)["mid"], 16000)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]["pitch"], 3000.0, delta=30.0)

    def test_ignores_noise_and_a_sliding_whistle(self):
        rng = np.random.default_rng(1)
        sound = 0.05 * rng.standard_normal((3 * SR, 2))
        t = np.arange(int(0.6 * SR)) / SR
        slide = np.sin(2 * np.pi * np.cumsum(1500 + 2500 * t / 0.6) / SR)      # 1.5 to 4 kHz in 0.6 s
        sound[SR:SR + len(slide), 0] += 0.5 * slide
        self.assertEqual(find_events(features(sound, SR)["mid"], SR), [])

    def test_ignores_a_voice_like_comb_of_harmonics(self):
        t = np.arange(int(0.5 * SR)) / SR
        voice = sum(np.sin(2 * np.pi * 150 * h * t) for h in range(10, 25))   # 1.5 to 3.6 kHz, 150 Hz apart
        sound = np.zeros((2 * SR, 2))
        sound[SR // 2:SR // 2 + len(voice)] += 0.05 * voice[:, None]
        self.assertEqual(find_events(features(sound, SR)["mid"], SR), [])

    def test_too_short_is_not_an_event(self):
        events = find_events(features(scene_with_tone(length=0.04), SR)["mid"], SR)
        self.assertEqual(events, [])

    def test_chunking_does_not_change_the_features(self):
        sound = scene_with_tone(seed=3)
        whole = features(sound, SR, chunk_s=100.0)
        stream = FeatureStream(SR)
        start = 0
        for size in (1000, 7777, 48000, 12345):
            stream.push(sound[start:start + size])
            start += size
        stream.push(sound[start:])
        pieces = stream.result()
        for mode in ("mid", "side"):
            np.testing.assert_array_equal(whole[mode], pieces[mode])

    def test_side_mix_cancels_what_is_dead_centre(self):
        voice = tone(0.5, 2000.0)
        both = mixes(np.stack([voice, voice], axis=1))
        self.assertLess(np.abs(both["side"]).max(), 1e-12)
        self.assertGreater(np.abs(both["mid"]).max(), 0.9)

    def test_one_channel_has_a_silent_side(self):
        f = features(scene_with_tone()[:, 0], SR)
        self.assertEqual(find_events(f["side"], SR), [])
        self.assertEqual(len(find_events(f["mid"], SR)), 1)

    def test_one_voice_picks_the_common_pitch(self):
        events = [{"pitch": p} for p in (3000, 3010, 2990, 3005, 1800, 4200)]
        pitch, share = one_voice(events)
        self.assertAlmostEqual(pitch, 3000, delta=20)
        self.assertAlmostEqual(share, 4 / 6)
        self.assertEqual(one_voice([]), (None, 0.0))

    def test_both_mixes_together_keep_one_event_per_whistle(self):
        sound = scene_with_tone()
        sound[:, 1] += 0.3 * sound[:, 0]                       # the tone is in both channels, off-centre
        events = combined_events(features(sound, SR), SR)
        self.assertEqual(len(events), 1)
        self.assertIn(events[0]["mix"], ("mid", "side"))

    def test_rules_can_be_made_stricter(self):
        f = features(scene_with_tone(), SR)["mid"]
        self.assertEqual(find_events(f, SR, EventRules(tonality_db=200.0)), [])


class RefereeFilter(unittest.TestCase):
    def events(self):
        referee = [{"start": 10.0 * i, "pitch": 3000.0 + 10 * (i % 3)} for i in range(8)]
        noise = [{"start": 10.0 * i + 5, "pitch": p} for i, p in enumerate((1600, 2100, 2500, 3600, 4400, 5200))]
        jingle = [{"start": 3.0 + 7.0 * i, "pitch": 4800.0} for i in range(10)]     # more frequent than the referee
        return referee, noise, jingle

    def test_groups_by_pitch_biggest_first(self):
        referee, noise, _ = self.events()
        groups = pitch_groups(referee + noise, tolerance=0.03, min_size=3)
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(groups[0]["members"]), list(range(8)))

    def test_keeps_the_referee_and_drops_scattered_false_alarms(self):
        referee, noise, _ = self.events()
        kept, report = referee_whistles(referee + noise)
        self.assertEqual([e["start"] for e in kept], [e["start"] for e in referee])
        self.assertEqual(report["how"], "the biggest group")

    def test_stoppages_pick_the_referee_over_a_more_frequent_jingle(self):
        referee, noise, jingle = self.events()
        events = sorted(referee + noise + jingle, key=lambda e: e["start"])
        self.assertAlmostEqual(referee_whistles(events)[1]["chosen"], 4800.0)      # biggest group alone is fooled
        stops = [e["start"] + 1.0 for e in referee]
        kept, report = referee_whistles(events, stops=stops)
        self.assertAlmostEqual(report["chosen"], 3010.0, delta=15)
        self.assertEqual(len(kept), 8)

    def test_nothing_shared_means_nothing_kept(self):
        kept, report = referee_whistles([{"start": 1.0, "pitch": 2000.0}, {"start": 2.0, "pitch": 4000.0}])
        self.assertEqual(kept, [])
        self.assertIsNone(report["chosen"])

    def test_near_stops_window(self):
        found = near_stops([10.0, 20.0, 30.0], [9.5, 23.5, 32.9])
        self.assertEqual(found.tolist(), [True, False, True])


class FakeStadium(unittest.TestCase):
    def test_same_seed_same_scene(self):
        a, b = make(Scene(seconds=20, seed=4)), make(Scene(seconds=20, seed=4))
        np.testing.assert_array_equal(a.sound, b.sound)
        self.assertEqual(a.onsets, b.onsets)

    def test_whistles_fall_inside_the_scene_and_are_found_when_loud(self):
        made = make(Scene(seconds=40, seed=5, whistle_db=0.0, commentary_db=None, fans=False))
        self.assertTrue(made.onsets)
        self.assertTrue(all(0 < o < 40 for o in made.onsets))
        events = combined_events(features(made.sound, SR), SR)
        found = sum(any(abs(e["start"] - o) < 0.2 for e in events) for o in made.onsets)
        self.assertGreaterEqual(found / len(made.onsets), 0.7)


class Judging(unittest.TestCase):
    def test_labels_are_read_per_half_in_seconds(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "Labels-v2.json")
            with open(path, "w") as handle:
                json.dump({"annotations": [{"gameTime": "2 - 01:00", "label": "Foul", "position": "60500"},
                                           {"gameTime": "1 - 00:10", "label": "Kick-off", "position": "10000"}]},
                          handle)
            labels = read_labels(path)
        self.assertEqual(labels[1], [{"label": "Kick-off", "time": 10.0}])
        self.assertEqual(labels[2], [{"label": "Foul", "time": 60.5}])

    def test_windows_quiet_play_and_chance(self):
        labels = [{"label": "Foul", "time": 100.0}, {"label": "Foul", "time": 300.0},
                  {"label": "Ball out of play", "time": 500.0}]
        events = [{"start": 101.0}, {"start": 700.0}]          # one foul found, one false alarm far away
        total = combine([judge_half(events, labels, 900.0)])
        self.assertEqual(total["labels"]["Foul"], [1, 2])
        self.assertEqual(total["labels"]["Ball out of play"], [0, 1])
        self.assertEqual(total["quiet_events"], 1)
        self.assertAlmostEqual(total["quiet_s"], 900.0 - 3 * 20.0)
        self.assertAlmostEqual(total["median_timing"], 1.0)
        self.assertGreater(total["chance"]["Foul"], 0.0)


class Decoding(unittest.TestCase):
    def test_sound_survives_the_trip_through_flac(self):
        try:
            ffmpeg = audio.find_ffmpeg()
        except RuntimeError:
            self.skipTest("no ffmpeg on this computer")
        sound = scene_with_tone()
        with tempfile.TemporaryDirectory() as folder:
            wav, flac = os.path.join(folder, "a.wav"), os.path.join(folder, "a.flac")
            with wave.open(wav, "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(SR)
                handle.writeframes((np.clip(sound, -1, 1) * 32767).astype("<i2").tobytes())
            audio.to_flac(wav, flac, ffmpeg)
            back = np.concatenate(list(audio.chunks(flac, chunk_s=1.0, ffmpeg=ffmpeg)))
        self.assertEqual(back.shape, sound.shape)
        self.assertLess(np.abs(back - sound).max(), 1e-3)
        self.assertEqual(len(find_events(features(back, SR)["mid"], SR)), 1)


if __name__ == "__main__":
    unittest.main()
