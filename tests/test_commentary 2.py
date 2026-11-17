"""The commentary as decoder evidence: name finding, placing on the timeline, nudges, the gate. Invented data only."""

import os
import sys
import unittest

import numpy as np

from mtah import commentary as cm
from mtah.align import LIVE, REPLAY
from mtah.match import A, D, H
from mtah.names import Player, Roster

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import extract_possession_mentions as epm  # noqa: E402

FPS = 25


def roster():
    return Roster([Player("TOMAS ELLERY", "home"), Player("RAFE OKONJO", "home"), Player("MARCO VELLANI", "home"),
                   Player("HUGO LARRANAGA", "away"), Player("PEIO ELLERI", "away"), Player("IKER ALDAZABAL", "away"),
                   Player("JON ALDAZABAL", "away"), Player("MARCO BASTERRA", "away")])


class Names(unittest.TestCase):
    def test_finds_players_and_their_side(self):
        found = roster().find("Okonjo plays it to Vellani, Larranaga closes in")
        self.assertEqual([(m.said, m.side) for m in found], [("Okonjo", "home"), ("Vellani", "home"), ("Larranaga", "away")])

    def test_misspelling_and_two_word_name(self):
        found = roster().find("Then Okonjio again, and Hugo Larranaga")
        self.assertEqual([(m.player, m.side) for m in found], [("RAFE OKONJO", "home"), ("HUGO LARRANAGA", "away")])
        self.assertEqual(found[1].said, "Hugo Larranaga")

    def test_confusable_names_are_never_used(self):
        r = roster()
        self.assertIn("ELLERY", r.dropped)
        self.assertIn("ELLERI", r.dropped)
        self.assertIn("MARCO", r.dropped)          # a first name in both teams
        self.assertEqual(r.find("Ellery and Elleri and Marco"), [])

    def test_lower_case_words_are_not_names(self):
        self.assertEqual(roster().find("the okonjo pass"), [])

    def test_shared_surname_in_one_team_gives_the_team_only(self):
        found = roster().find("Aldazabal heads it clear")
        self.assertEqual((found[0].player, found[0].side), ("", "away"))


class Extractor(unittest.TestCase):
    def test_time_is_the_place_in_the_line(self):
        seg = {"start": 10.0, "end": 14.0, "text": "abcdefghij"}
        self.assertAlmostEqual(epm.mention_time(seg, 4, 6), 12.0)

    def test_parse_keeps_known_roles_once(self):
        raw = '{"mentions": [{"id": "m1", "role": "on_ball"}, {"id": "m1", "role": "replay"}, {"id": "m2", "role": "dancing"}]}'
        self.assertEqual(epm.parse(raw), {"m1": "on_ball"})
        self.assertEqual(epm.parse("not json"), {})


def timeline(n_frames=3000, halves=((100, 1400), (1600, 2900))):
    frames, half = [], []
    for h, (a, b) in enumerate(halves):
        f = np.arange(a, b, 5)
        frames.append(f)
        half.append(np.full(len(f), h))
    return np.concatenate(frames), np.concatenate(half), halves


class Nudges(unittest.TestCase):
    def test_place_maps_and_drops_replays(self):
        pieces = np.array([[0, 1000, 100]])
        state = np.full(1000, LIVE)
        state[200:300] = REPLAY
        rows = [{"time_s": "4.0"}, {"time_s": "10.0"}, {"time_s": "50.0"}]
        kept, outside, replay = cm.place(rows, pieces, state)
        self.assertEqual([r["frame"] for r in kept], [200])
        self.assertEqual((outside, replay), (1, 1))

    def test_team_window_before_the_name(self):
        frames, half, halves = timeline()
        m = cm.masks([{"side": "home", "role": "on_ball", "frame": 500}], [], frames, half, halves)
        on = frames[m["team"][:, 0]]
        self.assertEqual((on.min(), on.max()), (450, 510))
        self.assertFalse(m["team"][:, 1].any())

    def test_loss_nudges_the_other_team_and_conflicts_cancel(self):
        frames, half, halves = timeline()
        m = cm.masks([{"side": "home", "role": "loses_ball", "frame": 500}], [], frames, half, halves)
        self.assertTrue(m["team"][:, 1].any() and not m["team"][:, 0].any())
        m = cm.masks([{"side": "home", "role": "on_ball", "frame": 500},
                      {"side": "away", "role": "on_ball", "frame": 505}], [], frames, half, halves)
        self.assertFalse(m["team"].all(axis=1).any())

    def test_nothing_from_not_live_replay_or_speculation(self):
        frames, half, halves = timeline()
        m = cm.masks([{"side": "home", "role": r, "frame": 500} for r in ("not_live", "replay", "unclear")],
                     [{"label": "foul", "status": "happened", "basis": "speculation", "frame": 500},
                      {"label": "foul", "status": "replay_or_recap", "basis": "described_live", "frame": 600}],
                     frames, half, halves)
        self.assertFalse(m["team"].any() or m["dead"].any() or m["alive"].any())

    def test_events_dead_and_denials_alive(self):
        frames, half, halves = timeline()
        m = cm.masks([], [{"label": "offside", "status": "happened", "basis": "referee_decision", "frame": 700},
                          {"label": "offside", "status": "did_not_happen", "basis": "referee_decision", "frame": 1000},
                          {"label": "foul", "status": "did_not_happen", "basis": "described_live", "frame": 720}],
                     frames, half, halves)
        self.assertTrue(m["dead"][(frames >= 700) & (frames <= 790)].all())
        self.assertTrue(m["alive"][(frames >= 930) & (frames <= 1020)].all())
        self.assertFalse((m["dead"] & m["alive"]).any())

    def test_windows_stay_in_their_half(self):
        frames, half, halves = timeline()
        m = cm.masks([{"side": "away", "role": "on_ball", "frame": 1610}], [], frames, half, halves)
        self.assertTrue((half[m["team"][:, 1]] == 1).all())
        m = cm.masks([{"side": "away", "role": "on_ball", "frame": 1500}], [], frames, half, halves)
        self.assertFalse(m["team"].any())               # half-time: no samples

    def test_prior_counts_each_kind_once(self):
        frames, half, halves = timeline()
        m = cm.masks([{"side": "home", "role": "on_ball", "frame": 500}] * 3,
                     [{"label": "foul", "status": "happened", "basis": "referee_decision", "frame": 500}] * 2,
                     frames, half, halves)
        prior = cm.prior_from(m)
        w = cm.CommentaryWeights()
        self.assertEqual(prior[:, H].max(), w.team)
        self.assertEqual(prior[:, D].max(), w.dead)
        self.assertEqual(prior[:, A].max(), 0.0)


class Gate(unittest.TestCase):
    def setUp(self):
        n = 1000
        self.prob = np.tile([0.95, 0.03, 0.02], (n, 1))
        self.prob[500:] = [0.03, 0.95, 0.02]
        self.team = np.zeros((n, 2), dtype=bool)

    def verdict(self, right_share):
        team = self.team.copy()
        k = int(right_share * 1000)
        truth = np.r_[np.zeros(500, int), np.ones(500, int)]
        said = truth.copy()
        said[k:] = 1 - said[k:]
        team[np.arange(1000), said] = True
        return cm.gate({"team": team, "dead": np.zeros(1000, bool), "alive": np.zeros(1000, bool)}, self.prob)

    def test_verdicts(self):
        self.assertEqual(self.verdict(0.9)["verdict"], "on")
        self.assertEqual(self.verdict(0.5)["verdict"], "noise")
        self.assertEqual(self.verdict(0.1)["verdict"], "swapped")
        few = {"team": self.team.copy(), "dead": np.zeros(1000, bool), "alive": np.zeros(1000, bool)}
        few["team"][:50, 0] = True
        self.assertEqual(cm.gate(few, self.prob)["verdict"], "too little")



class SureCues(unittest.TestCase):
    def test_keeps_model_answers_and_action_words_only(self):
        rows = [{"role": "on_ball", "said": "Okonjo", "line": "Okonjo again"},
                {"role": "not_live", "said": "Vellani", "line": "Vellani passes it wide"},
                {"role": "not_live", "said": "Larranaga", "line": "Larranaga has six goals this season"},
                {"role": "replay", "said": "Okonjo", "line": "watch Okonjo shoots there"},
                {"role": "fouled", "said": "Okonjo", "line": "Okonjo goes down"},
                {"role": "unclear", "said": "Vellani", "line": "a tackle on Vellani"}]
        kept, counts = cm.sure_cues(rows)
        self.assertEqual([(r["said"], r["role"], r["cue"]) for r in kept],
                         [("Okonjo", "on_ball", "model"), ("Vellani", "on_ball", "words")])
        self.assertEqual(counts, {"model": 1, "words": 1, "dropped": 4})

    def test_action_word_must_be_near_the_name(self):
        far = "Vellani was named in the squad on Tuesday by the coach, who later said the team" + " x" * 20 + " passes"
        self.assertFalse(cm.near_action(far, "Vellani"))
        self.assertTrue(cm.near_action("a lovely header from Vellani", "Vellani"))
        self.assertFalse(cm.near_action("Vellani is passionate", "Vellani"))   # whole words only


def write_sides(folder, first, second, first_numbers, second_numbers, home_team, away_team, home_numbers, away_numbers):
    import json
    players = "".join(f'<Player id="p{t}{n}" teamId="{t}"><Name/><ShirtNumber>{n}</ShirtNumber></Player>'
                      for t, nums in (("t1", first_numbers), ("t2", second_numbers)) for n in nums)
    xml = (f'<Root><Teams><Team id="t1"><Name>{first}</Name></Team><Team id="t2"><Name>{second}</Name></Team></Teams>'
           f'<Players>{players}</Players></Root>')
    meta, lineups = os.path.join(folder, "meta.xml"), os.path.join(folder, "lineups.json")
    with open(meta, "w") as f:
        f.write(xml)
    side = lambda team, nums: {"team": team, "how": "test", "others": [],
                               "starting": [{"name": f"P{n}", "number": n} for n in nums]}
    with open(lineups, "w") as f:
        json.dump({"home": side(home_team, home_numbers), "away": side(away_team, away_numbers)}, f)
    return lineups, meta


class Sides(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.dir = tempfile.mkdtemp()

    def test_agrees_swapped_unclear(self):
        a, b = [1, 4, 19, 27, 33, 40], [1, 5, 8, 21, 24, 31]
        args = (self.dir, "Riverton FC", "Atletico Northport", a, b)
        self.assertEqual(cm.side_check(*write_sides(*args, "RIVERTON", "NORTHPORT", a, b))["verdict"], "agrees")
        self.assertEqual(cm.side_check(*write_sides(*args, "NORTHPORT", "RIVERTON", b, a))["verdict"], "swapped")
        # names say one thing, numbers the other: unclear
        self.assertEqual(cm.side_check(*write_sides(*args, "RIVERTON", "NORTHPORT", b, a))["verdict"], "unclear")
        # no team text: numbers alone decide, and only by a clear margin
        self.assertEqual(cm.side_check(*write_sides(*args, "", "", a, b))["verdict"], "agrees")
        self.assertEqual(cm.side_check(*write_sides(*args, "", "", [1], [1]))["verdict"], "unclear")

    def test_team_gate_uses_the_small_minimum_only_when_sides_agree(self):
        n = 100
        prob = np.tile([0.95, 0.03, 0.02], (n, 1))
        team = np.zeros((n, 2), dtype=bool)
        team[:30, 0] = True                                 # 30 sure moments, all agreeing
        m = {"team": team, "dead": np.zeros(n, bool), "alive": np.zeros(n, bool)}
        a, b = [1, 4, 19, 27, 33, 40], [1, 5, 8, 21, 24, 31]
        args = (self.dir, "Riverton FC", "Atletico Northport", a, b)
        self.assertEqual(cm.team_gate(m, prob, *write_sides(*args, "RIVERTON", "NORTHPORT", a, b))["verdict"], "on")
        self.assertEqual(cm.team_gate(m, prob, *write_sides(*args, "RIVERTON", "NORTHPORT", b, a))["verdict"], "too little")
        self.assertEqual(cm.team_gate(m, prob, *write_sides(*args, "NORTHPORT", "RIVERTON", b, a))["verdict"], "swapped")



class CommittedCues(unittest.TestCase):
    """The cue files must give exactly what the full chain gives."""

    def test_cues_give_the_same_masks_as_the_chain(self):
        import csv as _csv
        import tempfile
        folder = tempfile.mkdtemp()
        frames, half, halves = timeline()
        np.savez(os.path.join(folder, "align.npz"), pieces=np.array([[0, 4000, 80]]),
                 broadcast_state=np.where(np.arange(4000) // 100 == 7, REPLAY, LIVE))
        mentions = [  # broadcast seconds; tactical frame = 25 * s + 80
            {"time_s": "16.0", "segment": "s1", "side": "home", "player": "X", "said": "Okonjo", "role": "on_ball", "line": "Okonjo"},
            {"time_s": "20.0", "segment": "s2", "side": "away", "player": "X", "said": "Larranaga", "role": "not_live",
             "line": "Larranaga shoots"},
            {"time_s": "29.0", "segment": "s3", "side": "home", "player": "X", "said": "Vellani", "role": "on_ball", "line": "Vellani"},
            {"time_s": "40.0", "segment": "s4", "side": "home", "player": "X", "said": "Vellani", "role": "not_live", "line": "Vellani scored"},
            {"time_s": "64.0", "segment": "s5", "side": "away", "player": "X", "said": "Elleri", "role": "loses_ball", "line": "Elleri"}]
        events = [{"time_s": "33.0", "label": "foul", "status": "happened", "basis": "described_live", "segment": "s6",
                   "quote": "secret words", "player": "X", "team": "", "p_event": "0.8", "quote_ok": "1"},
                  {"time_s": "45.0", "label": "offside", "status": "did_not_happen", "basis": "referee_decision",
                   "segment": "s7", "quote": "", "player": "", "team": "", "p_event": "0.5", "quote_ok": "1"}]
        for name, rows in (("m.csv", mentions), ("e.csv", events)):
            with open(os.path.join(folder, name), "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        m, e, a = (os.path.join(folder, n) for n in ("m.csv", "e.csv", "align.npz"))
        chain, info = cm.load(m, e, a, frames, half, halves)
        poss, stop, placed = cm.placed_cues(m, e, a)
        cm.write_cues(os.path.join(folder, "cues"), poss, stop)
        with open(os.path.join(folder, "cues", cm.CUE_MANIFEST), "w") as f:
            f.write('{"side_check": {"verdict": "agrees"}, "counts": {}}')
        p2, s2, manifest = cm.read_cues(os.path.join(folder, "cues"))
        again = cm.masks(p2, s2, frames, half, halves)
        for key in ("team", "dead", "alive"):
            np.testing.assert_array_equal(chain[key], again[key])
        self.assertTrue(chain["team"].any() and chain["dead"].any() and chain["alive"].any())
        self.assertEqual(info["cues"], placed["cues"])
        self.assertEqual(info["mentions"]["replay"], 1)          # the 29 s cue falls in the replay
        text = open(os.path.join(folder, "cues", cm.STOPPAGE_CUES)).read() + open(
            os.path.join(folder, "cues", cm.POSSESSION_CUES)).read()
        for private in ("secret", "Okonjo", "Larranaga", "Vellani", "Elleri"):
            self.assertNotIn(private, text)                      # no text and no names in the cue files

    def test_gate_from_cues_follows_the_recorded_side_check(self):
        n = 100
        prob = np.tile([0.95, 0.03, 0.02], (n, 1))
        team = np.zeros((n, 2), dtype=bool)
        team[:30, 0] = True
        m = {"team": team, "dead": np.zeros(n, bool), "alive": np.zeros(n, bool)}
        verdict = lambda s: cm.gate_from_cues(m, prob, {"side_check": {"verdict": s}})["verdict"]
        self.assertEqual((verdict("agrees"), verdict("unclear"), verdict("swapped")), ("on", "too little", "swapped"))


if __name__ == "__main__":
    unittest.main()
