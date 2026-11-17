"""The label-free checks in scripts/check_real_match.py, on made-up tracks."""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import check_real_match as c  # noqa: E402
from mtah.match import A, D, H  # noqa: E402

Q = 1e-5 * 105.3


def wandering(n, seed=0, step=0.4):
    rng = np.random.default_rng(seed)
    xy = np.cumsum(rng.normal(0, step, (n, 2)), axis=0)
    return xy[:, 0].copy(), xy[:, 1].copy()


def bridge(x, y, a, b, decimals=5):
    """Replace frames a..b with a straight, evenly spaced line stored at the file's precision."""
    t = np.linspace(0, 1, b - a + 1)
    x[a:b + 1] = np.round((x[a] + t * (x[b] - x[a])) / 105.3, decimals) * 105.3
    y[a:b + 1] = np.round((y[a] + t * (y[b] - y[a])) / 68.0, decimals) * 68.0


class StraightLines(unittest.TestCase):
    def test_a_wandering_ball_has_none(self):
        x, y = wandering(5000)
        first, last, moving = c.straight_runs(x, y, Q, Q)
        self.assertEqual(len(first), 0)
        self.assertGreater(moving.mean(), 0.9)

    def test_a_bridged_stretch_is_found_where_it_is(self):
        x, y = wandering(5000)
        x[1050], y[1050] = x[1000] + 30.0, y[1000] + 10.0
        bridge(x, y, 1000, 1050)
        first, last, _ = c.straight_runs(x, y, Q, 1e-5 * 68.0)
        self.assertEqual(len(first), 1)
        self.assertLessEqual(abs(int(first[0]) - 1000), 1)
        self.assertLessEqual(abs(int(last[0]) - 1050), 1)

    def test_a_ball_lying_still_does_not_count(self):
        x, y = np.full(500, 3.0), np.full(500, -2.0)
        self.assertEqual(len(c.straight_runs(x, y, Q, Q)[0]), 0)

    def test_a_short_line_does_not_count(self):
        x, y = wandering(2000)
        x[510], y[510] = x[500] + 6.0, y[500] + 2.0
        bridge(x, y, 500, 510)
        self.assertEqual(len(c.straight_runs(x, y, Q, 1e-5 * 68.0)[0]), 0)

    def test_figures_follow_the_share_of_bridged_ball(self):
        x, y = wandering(20000, seed=1)
        for a in range(200, 19000, 400):
            x[a + 100], y[a + 100] = x[a] + 40.0, y[a] + 15.0
            bridge(x, y, a, a + 100)
        f = c.ball_figures(x, y, Q, 1e-5 * 68.0)
        self.assertGreater(f["share"], 0.2)
        self.assertLess(f["share"], 0.3)
        self.assertGreaterEqual(f["longest_s"], 4.0)
        self.assertEqual(c.ball_figures(*wandering(20000, seed=1), Q, Q)["share"], 0.0)

    def test_lines_next_to_long_holes_are_counted(self):
        x, y = wandering(6000, seed=2)
        x[1040], y[1040] = x[1000] + 25.0, y[1000] + 5.0
        bridge(x, y, 1000, 1040)
        x[1041:1400], y[1041:1400] = np.nan, np.nan
        f = c.ball_figures(x, y, Q, 1e-5 * 68.0, blocks=[(0, 6000)])
        self.assertEqual(f["line_before_hole"], 1.0)
        self.assertEqual(f["line_after_hole"], 0.0)

    def test_words(self):
        self.assertEqual(c.share_words(0.0), "almost none of the moving ball")
        self.assertEqual(c.share_words(0.5), "about half of the moving ball")
        self.assertEqual(c.share_words(0.9), "most of the moving ball")
        self.assertEqual(c.length_words(4.5), "4 to 6 s")
        self.assertEqual(c.part_words(0.6), "most")


class BallText(unittest.TestCase):
    def test_reads_the_ball_as_written(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "atd.txt")
            with open(path, "w") as handle:
                handle.write("1:0.1,0.2,0.0;NaN,NaN,NaN:0.50000,0.25000,1.2\n2:0.1,0.2,0.0:NaN,NaN,NaN\n"
                             "3:0.1,0.2,0.0:0.50010,0.25020,1.2\n")
            x, y, decimals = c.read_ball_text(path, 3)
        self.assertEqual(decimals, 5)
        self.assertEqual(x[0], 0.5)
        self.assertTrue(np.isnan(x[1]))
        self.assertEqual(y[2], 0.2502)


class OutputFigures(unittest.TestCase):
    def test_counts_possessions_and_confidence(self):
        n = 300
        frames = np.arange(2, n * 5, 5)
        half = np.zeros(n, dtype=np.int8)
        prob = np.tile([0.9, 0.05, 0.05], (n, 1))
        prob[100:200] = [0.05, 0.9, 0.05]
        prob[250:] = [0.02, 0.02, 0.96]
        seen = np.ones(n, dtype=bool)
        seen[150:] = False
        f = c.output_figures(prob, frames, half, seen, [(0, n * 5)], n * 5)
        self.assertAlmostEqual(f["dead_share"], 50 / 300, places=2)
        self.assertAlmostEqual(f["possessions_per_live_minute"], 3 / (250 * 5 / 25 / 60), places=1)  # H, A, H
        self.assertEqual(f["torn"], 0.0)
        self.assertGreater(f["sure_ball_hidden"], f["sure_ball_seen"])

    def test_input_figures_shape(self):
        values = np.array([[0.0, np.nan], [1.0, 2.0], [2.0, 4.0], [3.0, np.nan]])
        f = c.input_figures(values)
        self.assertEqual(f.shape, (2, len(c.STATS)))
        self.assertEqual(f[1, 0], 0.5)
        self.assertEqual(f[0, 1], 1.5)


class InputDrift(unittest.TestCase):
    BOX = [1.0, 1.5, 2.5, 3.0]

    def test_inside_slightly_and_far(self):
        self.assertEqual(c.how_far(2.0, self.BOX, 0.1), "")
        self.assertEqual(c.how_far(3.1, self.BOX, 0.1), "")
        self.assertEqual(c.how_far(3.5, self.BOX, 0.1), "above")
        self.assertEqual(c.how_far(5.5, self.BOX, 0.1), "FAR ABOVE")
        self.assertEqual(c.how_far(-1.5, self.BOX, 0.1), "FAR BELOW")

    def test_a_range_with_no_width_uses_the_floor(self):
        same = [0.5, 0.5, 0.5, 0.5]
        self.assertEqual(c.how_far(0.52, same, 0.125), "above")
        self.assertEqual(c.how_far(0.7, same, 0.125), "FAR ABOVE")
        self.assertEqual(c.spread_floor({"q25": [0, 1, 1, 2], "q75": [4, 5, 5, 6]}, "q50"), 1.0)
        self.assertEqual(c.spread_floor({}, "missing"), 0.03)


CAMERA = np.array([1822.663, -511.2005, 44.48168, 1.698323, 169.2971, 255.7303, -0.01907188, -0.5269586, 1.0])


def to_pitch(u, v):
    """Pixels back to 0-1 pitch units through CAMERA, rounded as the tracking file is."""
    h = CAMERA.reshape(3, 3)
    xyw = np.linalg.solve(h, np.vstack((u, v, np.ones_like(u))))
    return np.round(xyw[0] / xyw[2], 5), np.round(xyw[1] / xyw[2], 5)


class FollowUp(unittest.TestCase):
    def test_graded_words(self):
        box = [1.0, 1.5, 2.5, 3.0]
        self.assertEqual(c.graded(2.0, box), "inside, middle")
        self.assertEqual(c.graded(1.2, box), "inside, low end")
        self.assertEqual(c.graded(3.3, box), "just above the open matches")
        self.assertEqual(c.graded(4.0, box), "clearly above the open matches")
        self.assertEqual(c.graded(-2.0, box), "FAR BELOW the open matches")
        self.assertEqual(c.graded(float("nan"), box), "not measurable")

    def test_every_ball_in_view_sample_is_in_one_situation(self):
        rng = np.random.default_rng(0)
        n = 4000
        dist_a, dist_b = rng.uniform(0, 12, n), rng.uniform(0, 12, n)
        col = {"dist_a": dist_a, "dist_b": dist_b, "near": np.fmin(dist_a, dist_b),
               "vote": np.clip(dist_b - dist_a, -6, 6), "ball_speed_s": rng.uniform(0, 20, n)}
        for key in col:
            col[key] = col[key].copy()
            col[key][::7] = np.nan            # ball hidden
        masks = c.situation_masks(col)
        total = sum(m.astype(int) for m in masks.values())
        self.assertTrue(np.array_equal(total, np.isfinite(col["near"]).astype(int)))
        self.assertTrue(all(m.any() for m in masks.values()))

    def test_doubt_figures(self):
        n = 100
        col = {"dist_a": np.full(n, 0.5), "dist_b": np.full(n, 9.0), "near": np.full(n, 0.5),
               "vote": np.full(n, 6.0), "ball_speed_s": np.full(n, 2.0)}
        prob = np.tile([0.9, 0.05, 0.05], (n, 1))
        prob[:50] = [0.3, 0.3, 0.4]
        f = c.why_figures(prob, col, np.ones(n, dtype=bool))
        self.assertEqual(f["share|at one player's feet"], 1.0)
        self.assertAlmostEqual(f["stoppage_doubt|view"], 0.5)
        self.assertAlmostEqual(f["team_clear|view"], (50 * 0 + 50 * (0.85 / 0.95)) / 100, places=6)
        self.assertTrue(np.isnan(f["sure|two players at the ball"]))

    def test_a_line_drawn_in_the_picture_is_found_there(self):
        n = 400
        rng = np.random.default_rng(3)
        u = 300 + np.cumsum(rng.normal(2.0, 1.5, n))
        v = 400 + np.cumsum(rng.normal(0.0, 1.5, n))
        u[100:161] = np.linspace(u[100], u[100] + 420.0, 61)       # 7 px a frame, into the depth of the pitch
        v[100:161] = np.linspace(v[100], v[100] - 150.0, 61)
        x, y = to_pitch(u, v)
        matrix = np.tile(CAMERA, (n, 1))
        f = c.picture_figures(matrix, x, y)
        self.assertGreater(f["share"], 0.1)
        self.assertGreaterEqual(f["longest_s"], 2.0)
        wandering_only = c.picture_figures(matrix, *to_pitch(300 + np.cumsum(rng.normal(2.0, 1.5, n)),
                                                             400 + np.cumsum(rng.normal(0.0, 1.5, n))))
        self.assertEqual(wandering_only["share"], 0.0)

    def test_homography_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "h.csv")
            with open(path, "w") as handle:
                handle.write("frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n")
                handle.write("0,1,0,0,0,1,0,0,0,1,0.2,1\n1,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,0\n")
            matrix = c.read_homography(path, 3)
        self.assertEqual(matrix[0].tolist(), [1, 0, 0, 0, 1, 0, 0, 0, 1])
        self.assertTrue(np.isnan(matrix[1]).all() and np.isnan(matrix[2]).all())

    def test_turns_at_a_player(self):
        from mtah.match import Match
        n = 600
        t = np.arange(n)
        x = 10.0 * np.abs(((t / 50.0) % 2) - 1)            # back and forth between x=0 and x=10
        ball = np.column_stack((x, np.zeros(n))).astype(np.float32)
        players = np.zeros((n, 2, 2), dtype=np.float32)
        players[:, 0] = (0.0, 0.5)                          # a player at one end only
        players[:, 1] = (30.0, 30.0)
        match = Match("toy", "toy", 25, (105.0, 68.0), ball, players, np.array([0, 1], dtype=np.int8), [(0, n)])
        f = c.turn_figures(match)
        self.assertGreater(f["turns_per_minute"], 0)
        self.assertGreater(f["turns_at_player"], 0.3)
        self.assertLess(f["turns_at_player"], 0.7)


class OneCamera(unittest.TestCase):
    def flights(self, lift):
        """Forty long balls along the pitch; lift pushes the middle of each toward one touchline."""
        from mtah.match import Match
        rows = []
        for k in range(40):
            t = np.linspace(0, 1, 40)
            x = -30 + 60 * t if k % 2 == 0 else 30 - 60 * t
            y = (k % 5) * 4.0 - 8.0 + lift * 4 * t * (1 - t)
            rows.append(np.column_stack((x, y)))
            rows.append(np.full((20, 2), np.nan))
        ball = np.vstack(rows).astype(np.float32)
        players = np.zeros((len(ball), 2, 2), dtype=np.float32)
        return Match("toy", "toy", 25, (105.0, 68.0), ball, players, np.array([0, 1], dtype=np.int8),
                     [(0, len(ball))])

    def test_straight_flights_do_not_bend(self):
        f = c.flight_bends(self.flights(0.0))
        self.assertEqual(f["flights"], 40)
        self.assertEqual(f["bent_share"], 0.0)

    def test_lifted_flights_bend_one_way(self):
        f = c.flight_bends(self.flights(6.0))
        self.assertEqual(f["bent_share"], 1.0)
        self.assertEqual(f["bend_one_way"], 1.0)
        self.assertEqual(c.one_way_words(f["bend_one_way"]), "nearly always the same way")

    def test_one_way(self):
        self.assertEqual(c.one_way(np.array([True, False] * 20)), 0.0)
        self.assertEqual(c.one_way(np.array([True] * 30)), 1.0)
        self.assertTrue(np.isnan(c.one_way(np.array([True] * 5))))
        self.assertEqual(c.one_way_words(float("nan")), "too few to tell")
        self.assertEqual(c.one_way_words(0.1), "both ways about equally")


if __name__ == "__main__":
    unittest.main()
