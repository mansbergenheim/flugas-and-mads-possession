"""Three label-free checks of the real match. Run this yourself, not through an AI tool.

Prints words only, never a measurement, so the result can be repeated to anyone. Writes nothing.

    MTAH_DATA_DIR=~/mtah-data python3 scripts/check_real_match.py

1. Ball. How much of the visible ball is a perfectly straight, evenly spaced line? A ball followed by
   cameras never is: it slows, curves and jitters. A line means the provider filled a stretch in. The
   public baseline shows almost no holes between half a second and five seconds, so short losses are
   probably bridged, and Metrica's own sample games are straight lines about half the time.
2. Inputs. For every input the model reads, does the real match fall inside what the practice copies
   cover? And does the real match stand out from the practice copies more than they do from each other?
3. Output. Is the model as sure of itself on the real match as on open matches it had not seen, with
   the ball in view and without it?

About 10 minutes. `--only ball`, `--only inputs` or `--only output` runs one of them. The reference
values come from open matches only and are in mtah/check_ranges.json (scripts/fit_check_ranges.py).

A follow-up, run only when asked for, looks at why the model is less sure than in practice when the
ball is in view. About 3 minutes, words only like the rest:

    MTAH_DATA_DIR=~/mtah-data python3 scripts/check_real_match.py --only followup

4a lists every input that is outside the practice copies and which way. 4b splits the ball-in-view
time by what is happening at the ball and says, for each, how much of it there is and how sure the
model is, against the 10 open matches. A match that is simply more contested shows up as more time in
the hard situations. A ball that misleads shows up as doubt in the easy ones. 4c asks whether the ball
turns where the players are, and whether fast balls bend toward one touchline, which is what a ball in
the air looks like when one camera's view is mapped onto the pitch. 4e asks whether anyone is tracked
back near each goal while that team is up the pitch, which is how a tracked goalkeeper shows. 4d repeats the straight-line test in the camera picture, where a fill
made before mapping onto the pitch would show. Its references are in mtah/check_followup.json
(scripts/fit_check_followup.py).
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from compare_tracking import position  # noqa: E402
from mtah import datasets as ds, harness  # noqa: E402
from mtah.config import DECODER, TRAIN_COPIES  # noqa: E402
from mtah.decoder import decode_samples, expand  # noqa: E402
from mtah.degrade import load_stats  # noqa: E402
from mtah.features import build_features  # noqa: E402
from mtah.loaders import gated  # noqa: E402
from mtah.match import D, Match  # noqa: E402
from mtah.model import LearnedModel  # noqa: E402

RANGES = os.path.join(os.path.dirname(__file__), "..", "mtah", "check_ranges.json")
SAVED_MODEL = os.path.join(os.path.dirname(__file__), "..", "runs", "models", "v5_final_model.pkl")
MIN_RUN = 15           # frames: 0.6 s
MIN_STEP_M = 0.08      # per frame, 2 m/s: a ball lying still is straight for free
LONG_HOLE = 125        # frames: 5 s
NEAR_PLAYER_M = 2.0
PRACTICE = ("J03WPY", "J03WMX", "METRICA1", "METRICA3")   # practice copies for the stand-out test
STATS = ("missing", "mean", "q05", "q25", "q50", "q75", "q95")
ASPECT = {"missing": "how often it is missing", "mean": "average", "q05": "lowest values", "q25": "lower values",
          "q50": "typical value", "q75": "higher values", "q95": "highest values"}


def runs_of(flag: np.ndarray):
    edges = np.diff(np.concatenate(([0], flag.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


# ---------------------------------------------------------------- 1. ball

def straight_runs(x: np.ndarray, y: np.ndarray, qx: float, qy: float, slack: float = 0.0,
                  min_step: float = MIN_STEP_M):
    """(first frame, last frame) of every run of at least MIN_RUN frames where the moving ball is on
    an exactly straight, evenly spaced line, and the mask of frames where the ball moves.

    A line stored at a precision of q can be off by q/2 per value, so its second difference stays
    within 2q. slack allows for positions that went through float32 on the way.
    """
    with np.errstate(invalid="ignore"):
        bend_x = np.abs(x[2:] - 2 * x[1:-1] + x[:-2])
        bend_y = np.abs(y[2:] - 2 * y[1:-1] + y[:-2])
        dx, dy = x[1:] - x[:-1], y[1:] - y[:-1]
        step = np.sqrt(dx * dx + dy * dy)
        moving = np.isfinite(step) & (step >= min_step)
        ok = (bend_x <= 2 * qx + slack) & (bend_y <= 2 * qy + slack) & moving[1:] & moving[:-1]
    first, stop = runs_of(ok)
    last = stop + 1
    keep = (last - first + 1) >= MIN_RUN
    return first[keep], last[keep], moving


def ball_figures(x, y, qx, qy, players=None, slack=0.0, fps=25, blocks=None) -> dict:
    """Figures about straight lines in one ball track. x, y in metres, NaN where there is no ball.
    blocks: (first, stop) frames of each half, so the breaks around the halves don't count as holes."""
    first, last, moving = straight_runs(x, y, qx, qy, slack)
    length = last - first + 1
    out = {"share": float(length.sum() / max(int(moving.sum()), 1)),
           "longest_s": float(length.max() / fps) if len(length) else 0.0,
           "long_per_hour": float((length >= 2 * fps).sum() / max(np.isfinite(x).sum() / fps / 3600, 1e-9))}
    seen = np.isfinite(x)
    ends, starts = set(last[length >= fps].tolist()), set(first[length >= fps].tolist())
    before, after = [], []
    for lo, hi in (blocks or [(0, len(x))]):
        hole_start, hole_stop = runs_of(~seen[lo:hi])
        inner = (hole_start > 0) & (hole_stop < hi - lo) & ((hole_stop - hole_start) >= LONG_HOLE)
        before += [any(lo + s - k in ends for k in (1, 2, 3)) for s in hole_start[inner].tolist()]
        after += [any(lo + e + k in starts for k in (0, 1, 2)) for e in hole_stop[inner].tolist()]
    out["line_before_hole"] = float(np.mean(before)) if before else 0.0
    out["line_after_hole"] = float(np.mean(after)) if after else 0.0
    if players is not None and len(first):
        def near(frames):
            gap = players[frames] - np.column_stack((x[frames], y[frames]))[:, None, :]
            size = np.sqrt((gap ** 2).sum(-1))
            return np.where(np.isfinite(size), size, np.inf).min(1) <= NEAR_PLAYER_M
        out["both_ends_at_player"] = float((near(first) & near(last)).mean())
    else:
        out["both_ends_at_player"] = 0.0
    return out


def read_ball_text(raw_path: str, n_frames: int):
    """The ball column exactly as written, in 0-1 pitch units, and the number of decimals used."""
    x = np.full(n_frames, np.nan)
    y = np.full(n_frames, np.nan)
    decimals = 0
    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3 or parts[2].startswith("N"):
                continue
            frame = int(parts[0]) - 1
            fields = parts[2].split(",")
            if 0 <= frame < n_frames:
                x[frame], y[frame] = float(fields[0]), float(fields[1])
                for text in fields[:2]:
                    decimals = max(decimals, len(text.partition(".")[2]))
    return x, y, decimals


def share_words(v: float) -> str:
    for limit, words in ((0.03, "almost none of the moving ball"), (0.15, "a small part of the moving ball"),
                         (0.35, "a large part of the moving ball"), (0.65, "about half of the moving ball")):
        if v < limit:
            return words
    return "most of the moving ball"


def length_words(seconds: float) -> str:
    for limit, words in ((1, "under 1 s"), (2, "1 to 2 s"), (4, "2 to 4 s"), (6, "4 to 6 s")):
        if seconds < limit:
            return words
    return "over 6 s"


def part_words(v: float) -> str:
    return "few" if v < 0.2 else "some" if v < 0.5 else "most"


def check_ball(ranges: dict) -> None:
    match = gated.load_atd_match()
    x, y, decimals = read_ball_text(gated.asset("atd_raw"), match.n_frames)
    quantum = 10.0 ** -decimals
    xm, ym = (x - 0.5) * gated.PITCH[0], (0.5 - y) * gated.PITCH[1]
    graded = np.zeros(match.n_frames, dtype=bool)
    for start, stop in match.halves:
        graded[start:stop] = True
    xm[~graded], ym[~graded] = np.nan, np.nan
    f = ball_figures(xm, ym, quantum * gated.PITCH[0], quantum * gated.PITCH[1], match.players,
                     blocks=match.halves)
    ref = ranges["ball"]
    low, high = ref["metrica_share"]
    if f["share"] < 0.03:
        against = "like a ball followed by cameras (the DFL matches)"
    elif f["share"] < low:
        against = "more than a ball followed by cameras, less than Metrica's sample games"
    elif f["share"] <= high:
        against = "like Metrica's sample games"
    else:
        against = "more than Metrica's sample games"
    rate = f["long_per_hour"]
    print("1. BALL")
    print(f"  On exactly straight, evenly spaced lines   {share_words(f['share'])}")
    print(f"  Compared with open matches                 {against}")
    print(f"  Longest such line                          {length_words(f['longest_s'])}")
    print(f"  Lines of 2 s or more                       {'none' if rate == 0 else 'rare' if rate < 20 else 'common' if rate < 150 else 'very common'}")
    print(f"  Long blackouts with a line right before    {part_words(f['line_before_hole'])}")
    print(f"  Long blackouts with a line right after     {part_words(f['line_after_hole'])}")
    print(f"  Lines that start and end at a player       {part_words(f['both_ends_at_player'])}")
    print(f"  File precision fine enough for this test   {'yes' if decimals >= 4 else 'NO: treat the lines above as unreliable'}")


# ---------------------------------------------------------------- 2. inputs

def input_figures(values: np.ndarray) -> np.ndarray:
    """[inputs, len(STATS)]: share missing, mean and five percentiles of each input."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        q = np.nanpercentile(values, (5, 25, 50, 75, 95), axis=0)
        mean = np.nanmean(values, axis=0)
    return np.column_stack((np.isnan(values).mean(0), mean, q.T))


def graded_rows(match: Match):
    """Inputs on the graded samples where anybody is tracked at all."""
    values, names, frames, half = build_features(match, extra=True)
    tracked = np.isfinite(match.players[frames][..., 0]).any(1)
    return values[tracked], names


def how_far(value: float, box, floor: float) -> str:
    """'' inside the practice copies' range, else below/above, in capitals when far outside.

    Far means beyond the range by more than its own width, where the width is never taken smaller
    than `floor` (a quarter of the input's usual spread). A new match is often a little outside on
    many inputs at once, because inputs come in families and 30 copies of 10 matches are few.
    """
    low, high = box[0], box[3]
    scale = max(high - low, floor, 1e-6)
    if value < low - scale:
        return "FAR BELOW"
    if value > high + scale:
        return "FAR ABOVE"
    if value < low - 0.1 * scale:
        return "below"
    return "above" if value > high + 0.1 * scale else ""


def spread_floor(boxes: dict, stat: str) -> float:
    """The smallest width worth calling a range: 3 points for 'missing', else a quarter of the
    input's usual spread between its lower and upper values."""
    if stat == "missing":
        return 0.03
    upper, lower = boxes["q75"], boxes["q25"]
    return 0.25 * abs((upper[1] + upper[2]) / 2 - (lower[1] + lower[2]) / 2)


def stand_out(rows: np.ndarray, others, seed: int = 0) -> float:
    """How well a classifier tells `rows` from the arrays in `others`: 0.5 can't, 1.0 always can. The
    first half of every source trains and the second half tests, so neighbouring samples don't sit on
    both sides."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    a_train, a_test = rows[:len(rows) // 2], rows[len(rows) // 2:]
    b_train = np.vstack([o[:len(o) // 2] for o in others])
    b_test = np.vstack([o[len(o) // 2:] for o in others])
    model = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=15, min_samples_leaf=100,
                                           learning_rate=0.1, early_stopping=False, random_state=seed)
    model.fit(np.vstack((a_train, b_train)), np.r_[np.ones(len(a_train)), np.zeros(len(b_train))])
    score = model.predict_proba(np.vstack((a_test, b_test)))[:, 1]
    return float(roc_auc_score(np.r_[np.ones(len(a_test)), np.zeros(len(b_test))], score))


def thin(values: np.ndarray, rows: int = 12000) -> np.ndarray:
    """Evenly spaced rows, kept in time order."""
    return values[np.linspace(0, len(values) - 1, min(rows, len(values))).astype(int)]


def check_inputs(ranges: dict) -> None:
    match = gated.load_atd_match()
    values, names = graded_rows(match)
    ref = ranges["inputs"]
    if names != ref["names"]:
        sys.exit("the inputs differ from mtah/check_ranges.json: run scripts/fit_check_ranges.py again")
    figures = input_figures(values)
    print("\n2. INPUTS")
    far = slight = 0
    for i, name in enumerate(names):
        notes, near = [], False
        for j, stat in enumerate(STATS):
            if not np.isfinite(figures[i, j]):
                if stat == "mean":
                    notes.append("never present")
                continue
            verdict = how_far(figures[i, j], ref["ranges"][name][stat], spread_floor(ref["ranges"][name], stat))
            if verdict.isupper():
                notes.append(f"{ASPECT[stat]} {verdict}")
            near = near or bool(verdict)
        if notes:
            far += 1
            print(f"  {name:<22} {'; '.join(notes)}")
        elif near:
            slight += 1
    print(f"  {far} of {len(names)} inputs are far outside what the practice copies cover (listed above)")
    print(f"  {slight} more are slightly outside, which is normal for a match the model has not seen")

    stats = load_stats()
    copies = [thin(build_features(copy, extra=True)[0])
              for copy in harness.training_matches(PRACTICE, 0, 1, stats)]
    among = [stand_out(copies[i], [c for j, c in enumerate(copies) if j != i]) for i in range(len(copies))]
    real = stand_out(thin(values), copies)
    if real <= max(among):
        words = "no more than one practice copy stands out from the others"
    elif real - max(among) < 0.03:
        words = "slightly more than any practice copy stands out from the others"
    else:
        words = "clearly more than any practice copy stands out from the others"
    print(f"  The real match stands out                  {words}")


# ---------------------------------------------------------------- 3. output

def output_figures(prob, frames, half, seen, halves, n_frames, play=None) -> dict:
    """Unlabelled figures about the model's own output. play: (first, stop) frame per half to count."""
    play = play or halves
    path = decode_samples(prob, half, DECODER)
    labels = expand(path, frames, half, halves, n_frames)
    keep = np.zeros(len(frames), dtype=bool)
    spells = alive = 0
    for start, stop in play:
        keep |= (frames >= start) & (frames < stop)
        window = labels[start:stop]
        begins = np.r_[True, window[1:] != window[:-1]]
        spells += int((begins & (window != D)).sum())
        alive += int((window != D).sum())
    top = prob.max(1)
    hidden = keep & ~seen
    return {"sure": float(top[keep].mean()),
            "sure_ball_seen": float(top[keep & seen].mean()) if (keep & seen).any() else float("nan"),
            "sure_ball_hidden": float(top[hidden].mean()) if hidden.any() else float("nan"),
            "torn": float((top[keep] < 0.6).mean()),
            "torn_ball_hidden": float((top[hidden] < 0.6).mean()) if hidden.any() else float("nan"),
            "dead_share": float((path[keep] == D).mean()),
            "possessions_per_live_minute": spells / max(alive / 25 / 60, 1e-9),
            "decoder_overrules": float((path[keep] != prob[keep].argmax(1)).mean())}


OUTPUT_NAMES = {"sure": "How sure the model is",
                "sure_ball_seen": "... with the ball in view",
                "sure_ball_hidden": "... with the ball hidden",
                "torn": "Share of moments it is torn",
                "torn_ball_hidden": "... with the ball hidden",
                "dead_share": "Share it calls dead ball",
                "possessions_per_live_minute": "Possessions per minute of play",
                "decoder_overrules": "How often the decoder overrules it"}


def kickoff_blocks(match: Match):
    """Graded halves without the stretch before each half's kickoff tag, if the tags can be read."""
    try:
        tags = gated.load_smart_tagging()
    except (OSError, ValueError):
        return None
    blocks = []
    for start, stop in match.halves:
        kicks = [t["start"] for t in tags if (t["code"] or "").strip().lower() == "set pieces"
                 and (t["tags"].get("Type") or "").strip().lower() == "kick off (start)" and start <= t["start"] < stop]
        blocks.append((min(kicks) if kicks else start, stop))
    return blocks


def check_output(ranges: dict, model_path: str) -> None:
    match = gated.load_atd_match()
    if os.path.exists(model_path):
        with open(model_path, "rb") as handle:
            model = pickle.load(handle)
        source = "the saved v5 model"
    else:
        ids = ds.available()
        model = LearnedModel(decoder=DECODER)
        model.fit(harness.training_matches(ids, 0, TRAIN_COPIES, load_stats()))
        source = "a v5 model trained just now"
    prob, frames, half = model.probabilities(match)
    seen = np.isfinite(match.ball[frames, 0])
    play = kickoff_blocks(match)
    figures = output_figures(prob, frames, half, seen, match.halves, match.n_frames, play)
    print(f"\n3. OUTPUT  ({source}"
          + ("" if play else "; tags unreadable, so the wait before the second-half kickoff is included") + ")")
    for key, label in OUTPUT_NAMES.items():
        verdict = position(figures[key], *ranges["output"][key]) if np.isfinite(figures[key]) else "not measurable"
        print(f"  {label:<42} {verdict.replace('trained on', 'seen on open matches')}")


# ---------------------------------------------------------------- 4. follow-up

FOLLOWUP = os.path.join(os.path.dirname(__file__), "..", "mtah", "check_followup.json")
FAST_BALL = 9.0        # m/s, the speed above which features.py treats the ball as travelling
PICTURE_TOL = 0.06     # pixels: what rounding the pitch position to the file's precision can add
SITUATIONS = ("at one player's feet", "two players at the ball", "ball travelling fast",
              "ball loose in open space", "in between")


def graded(value: float, box) -> str:
    """Where a value sits among the open matches, and how far outside when it is outside."""
    if not np.isfinite(value):
        return "not measurable"
    low, lower_quarter, upper_quarter, high = box
    width = max(high - low, 1e-9)
    for side, gap in (("below", low - value), ("above", value - high)):
        if gap > 0:
            if gap > width:
                return f"FAR {side.upper()} the open matches"
            return f"{'just' if gap < 0.25 * width else 'clearly'} {side} the open matches"
    if value < lower_quarter:
        return "inside, low end"
    return "inside, high end" if value > upper_quarter else "inside, middle"


def situation_masks(col: dict) -> dict:
    """Ball-in-view samples split by what is happening at the ball. Each is in exactly one group."""
    near, vote, speed = col["near"], col["vote"], col["ball_speed_s"]
    with np.errstate(invalid="ignore"):
        view = np.isfinite(near)
        fast = view & (speed >= FAST_BALL)
        duel = view & ~fast & (col["dist_a"] < 2.5) & (col["dist_b"] < 2.5)
        feet = view & ~fast & ~duel & (near < 1.5) & (np.abs(vote) >= 3.0)
        loose = view & ~fast & ~duel & ~feet & (near >= 3.0)
    other = view & ~(fast | duel | feet | loose)
    return dict(zip(SITUATIONS, (feet, duel, fast, loose, other)))


def why_figures(prob: np.ndarray, col: dict, keep: np.ndarray) -> dict:
    """Unlabelled figures on where the model's doubt sits. keep marks the samples to count."""
    top, p_dead = prob.max(1), prob[:, D]
    with np.errstate(invalid="ignore", divide="ignore"):
        side = np.abs(prob[:, 0] - prob[:, 1]) / (prob[:, 0] + prob[:, 1])
    in_view = np.isfinite(col["near"])
    view, dark = keep & in_view, keep & ~in_view

    def mean(values, mask):
        return float(values[mask].mean()) if mask.any() else float("nan")

    out = {}
    for name, mask in (("view", view), ("dark", dark)):
        out[f"stoppage_doubt|{name}"] = mean(((p_dead > 0.2) & (p_dead < 0.8)).astype(np.float64), mask)
        out[f"team_clear|{name}"] = mean(side, mask & (p_dead < 0.5))
    for name, mask in situation_masks(col).items():
        mask = mask & keep
        out[f"share|{name}"] = float(mask.sum() / max(int(view.sum()), 1))
        out[f"sure|{name}"] = mean(top, mask)
        out[f"team_clear|{name}"] = mean(side, mask & (p_dead < 0.5))
    return out


def turn_figures(match: Match, blocks=None, fps: int = 25) -> dict:
    """Does the ball turn where the players are? A sharp turn is a change of direction of more than
    60 degrees between the 3 frames before and the 3 frames after, with the ball moving both times."""
    ball = match.ball.astype(np.float64)
    turns = at_player = seen_frames = 0
    far, across, in_box = [], [], []
    for lo, hi in (blocks or match.halves):
        b = ball[lo:hi]
        before, after = b[3:-3] - b[:-6], b[6:] - b[3:-3]
        with np.errstate(invalid="ignore", divide="ignore"):
            size_before = np.sqrt((before * before).sum(1))
            size_after = np.sqrt((after * after).sum(1))
            cosine = (before[:, 0] * after[:, 0] + before[:, 1] * after[:, 1]) / (size_before * size_after)
            turn = (size_before >= 3 * MIN_STEP_M) & (size_after >= 3 * MIN_STEP_M) & (cosine < 0.5)
        index = np.flatnonzero(turn) + 3 + lo
        if len(index):
            gap = match.players[index].astype(np.float64) - ball[index][:, None, :]
            dist = np.where(np.isfinite(gap[..., 0]), np.sqrt((gap * gap).sum(-1)), np.inf)
            nearest = dist.argmin(1)
            reach = dist[np.arange(len(index)), nearest]
            at_player += int((reach <= NEAR_PLAYER_M).sum())
            away = (reach > NEAR_PLAYER_M) & np.isfinite(reach)
            spot = np.abs(ball[index][away])
            in_box.extend(((spot[:, 0] > match.pitch[0] / 2 - 16.5) & (spot[:, 1] < 20.16)).tolist())
            far.extend(reach[away].tolist())
            across.extend((-gap[np.arange(len(index)), nearest, 1][away]).tolist())   # ball minus player
        turns += len(index)
        seen_frames += int(np.isfinite(b[:, 0]).sum())
    across = np.array(across)
    return {"turns_at_player": at_player / turns if turns else float("nan"),
            "turns_per_minute": turns / max(seen_frames / fps / 60, 1e-9),
            "away_distance_m": float(np.median(far)) if far else float("nan"),
            "away_one_way": one_way(across > 0),
            "away_in_box": float(np.mean(in_box)) if in_box else float("nan")}


def keeper_figures(col: dict, pitch_length: float, keep: np.ndarray) -> dict:
    """Is anyone tracked back near each goal while that team is up the pitch?

    With a goalkeeper in the tracking the answer is nearly always yes. The inputs are turned so the
    home team defends the goal at minus half the pitch length and the away team the other one.
    """
    half = pitch_length / 2.0
    with np.errstate(invalid="ignore"):
        home_up = keep & (col["a_cx"] > -half + 35.0)
        away_up = keep & (col["b_cx"] < half - 35.0)
        home_back = col["a_minx"] < -half + 30.0
        away_back = col["b_maxx"] > half - 30.0
    return {"home": float(home_back[home_up].mean()) if home_up.any() else float("nan"),
            "away": float(away_back[away_up].mean()) if away_up.any() else float("nan")}


def one_way(flags: np.ndarray, least: int = 20) -> float:
    """0 when a yes/no splits evenly, 1 when it always falls the same way. NaN on too few cases."""
    flags = np.asarray(flags, dtype=bool)
    return float(abs(2.0 * flags.mean() - 1.0)) if len(flags) >= least else float("nan")


def one_way_words(value: float) -> str:
    if not np.isfinite(value):
        return "too few to tell"
    return "both ways about equally" if value < 0.3 else "leaning one way" if value < 0.6 else "nearly always the same way"


def flight_bends(match: Match, blocks=None, fps: int = 25) -> dict:
    """Do fast balls travel in straight lines?

    A ball in the air, seen by one camera and mapped onto the pitch, appears pushed away from the
    camera, the more so the higher it is. So the path of a lofted ball bends toward the far touchline
    and comes back. Cameras all round the pitch give the true spot, and the path stays straight.

    A flight is at least 0.8 s with the ball seen throughout, faster than 12 m/s and covering 8 m. Its
    bend is how far the middle of the path sits from the straight line between its ends.
    """
    ball = match.ball.astype(np.float64)
    bends, across = [], []
    for lo, hi in (blocks or match.halves):
        b = ball[lo:hi]
        with np.errstate(invalid="ignore"):
            step = b[5:] - b[:-5]
            fast = np.sqrt((step * step).sum(1)) * (fps / 5.0) >= 12.0
        first, stop = runs_of(fast)
        for start, end in zip(first.tolist(), (stop + 5).tolist()):
            if end - start < 20 or not np.isfinite(b[start:end, 0]).all():
                continue
            chord = b[end - 1] - b[start]
            length = float(np.sqrt((chord * chord).sum()))
            if length < 8.0:
                continue
            middle = b[(start + end) // 2] - (b[start] + b[end - 1]) / 2.0
            normal = np.array([-chord[1], chord[0]]) / length
            off = float(middle[0] * normal[0] + middle[1] * normal[1])
            bends.append(abs(off))
            across.append(off * normal[1])
    bends, across = np.array(bends), np.array(across)
    bent = bends > 1.5
    return {"flights": int(len(bends)),
            "bent_share": float(bent.mean()) if len(bends) else float("nan"),
            "bend_one_way": one_way(across[bent] > 0) if len(bends) else float("nan")}


def read_homography(path: str, n_frames: int) -> np.ndarray:
    """[n_frames, 9] matrices in row order, NaN where the file has none or marks it unusable."""
    out = np.full((n_frames, 9), np.nan)
    with open(path) as handle:
        header = [c.strip().lower() for c in handle.readline().split(",")]
        cells = [header.index(f"h{r}{c}") for r in range(3) for c in range(3)]
        frame_col, valid_col = header.index("frame"), header.index("valid")
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) != len(header):
                continue
            frame = int(parts[frame_col])
            if 0 <= frame < n_frames and float(parts[valid_col]) == 1:
                out[frame] = [float(parts[c]) for c in cells]
    return out


def picture_points(matrix: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Pitch positions in 0-1 units to pixels, frame by frame, with that frame's own matrix."""
    with np.errstate(invalid="ignore", divide="ignore"):
        w = matrix[:, 6] * x + matrix[:, 7] * y + matrix[:, 8]
        return ((matrix[:, 0] * x + matrix[:, 1] * y + matrix[:, 2]) / w,
                (matrix[:, 3] * x + matrix[:, 4] * y + matrix[:, 5]) / w)


def picture_figures(matrix: np.ndarray, x: np.ndarray, y: np.ndarray, fps: int = 25) -> dict:
    """Straight, evenly spaced lines of the ball in the camera picture. x, y in 0-1 pitch units."""
    u, v = picture_points(matrix, x, y)
    first, last, moving = straight_runs(u, v, PICTURE_TOL / 2, PICTURE_TOL / 2, min_step=1.0)
    length = last - first + 1
    return {"share": float(length.sum() / max(int(moving.sum()), 1)),
            "longest_s": float(length.max() / fps) if len(length) else 0.0}


def load_model(model_path: str):
    if os.path.exists(model_path):
        with open(model_path, "rb") as handle:
            return pickle.load(handle), "the saved v5 model"
    model = LearnedModel(decoder=DECODER)
    model.fit(harness.training_matches(ds.available(), 0, TRAIN_COPIES, load_stats()))
    return model, "a v5 model trained just now"


def check_followup(ranges: dict, model_path: str) -> None:
    with open(FOLLOWUP) as handle:
        ref = json.load(handle)
    match = gated.load_atd_match()

    values, names = graded_rows(match)
    if names != ranges["inputs"]["names"]:
        sys.exit("the inputs differ from mtah/check_ranges.json: run scripts/fit_check_ranges.py again")
    figures, boxes = input_figures(values), ranges["inputs"]["ranges"]
    print("4a. INPUTS OUTSIDE THE PRACTICE COPIES, AND WHICH WAY  (capitals: far outside)")
    for i, name in enumerate(names):
        notes = []
        for j, stat in enumerate(STATS):
            if np.isfinite(figures[i, j]):
                verdict = how_far(figures[i, j], boxes[name][stat], spread_floor(boxes[name], stat))
                if verdict:
                    notes.append(f"{ASPECT[stat]} {verdict}")
        if notes:
            print(f"  {name:<22} {'; '.join(notes)}")

    model, source = load_model(model_path)
    plain, plain_names, frames, half = build_features(match)
    col = {name: plain[:, i] for i, name in enumerate(plain_names)}
    prob, _, _ = model.probabilities(match)
    play = kickoff_blocks(match) or match.halves
    keep = np.zeros(len(frames), dtype=bool)
    for start, stop in play:
        keep |= (frames >= start) & (frames < stop)
    f = why_figures(prob, col, keep)
    out = output_figures(prob, frames, half, np.isfinite(match.ball[frames, 0]), match.halves, match.n_frames, play)
    print(f"\n4b. WHERE THE DOUBT SITS  ({source}, against the 10 open matches)")
    for key, label in (("sure_ball_seen", "How sure, ball in view"), ("sure_ball_hidden", "How sure, ball hidden"),
                       ("decoder_overrules", "How often the decoder overrules")):
        print(f"  {label:<46} {graded(out[key], ranges['output'][key])}")
    for key, label in (("stoppage_doubt|view", "Unsure whether play is on, ball in view"),
                       ("stoppage_doubt|dark", "Unsure whether play is on, ball hidden"),
                       ("team_clear|view", "How clear which team has it, ball in view"),
                       ("team_clear|dark", "How clear which team has it, ball hidden")):
        print(f"  {label:<46} {graded(f[key], ref['why'][key])}")
    print("  With the ball in view, by what is happening at the ball:")
    for name in SITUATIONS:
        print(f"    {name}")
        print(f"      {'share of the time':<42} {graded(f[f'share|{name}'], ref['why'][f'share|{name}'])}")
        print(f"      {'how sure the model is':<42} {graded(f[f'sure|{name}'], ref['why'][f'sure|{name}'])}")
        print(f"      {'how clear which team has it':<42} "
              f"{graded(f[f'team_clear|{name}'], ref['why'][f'team_clear|{name}'])}")

    turns = turn_figures(match, play)
    print("\n4c. DOES THE BALL TURN WHERE THE PLAYERS ARE  (against the 10 open matches as tested)")
    print(f"  {'Sharp turns of the ball that happen at a player':<50} "
          f"{graded(turns['turns_at_player'], ref['turns']['turns_at_player'])}")
    print(f"  {'Sharp turns per minute of visible ball':<50} "
          f"{graded(turns['turns_per_minute'], ref['turns']['turns_per_minute'])}")
    reach = turns["away_distance_m"]
    print(f"  {'The other turns: typical distance to a player':<50} "
          + ("not measurable" if not np.isfinite(reach) else "2 to 4 m" if reach < 4 else "4 to 8 m" if reach < 8
             else "over 8 m"))
    print(f"  {'The other turns: ball to which side of the player':<50} {one_way_words(turns['away_one_way'])}")
    print(f"  {'The other turns: share inside a penalty area':<50} "
          f"{graded(turns['away_in_box'], ref['turns']['away_in_box'])}")
    bends = flight_bends(match, play)
    print(f"  {'Fast balls whose path bends by over 1.5 m':<50} "
          f"{graded(bends['bent_share'], ref['bends']['bent_share'])}")
    print(f"  {'Those bends go':<50} {one_way_words(bends['bend_one_way'])}")

    keepers = keeper_figures(col, match.pitch[0], keep)
    print("\n4e. IS A GOALKEEPER TRACKED  (against the 10 open matches as tested)")
    for side in ("home", "away"):
        label = f"The {side} team has someone back near its goal"
        print(f"  {label:<50} {graded(keepers[side], ref['keepers'])}")

    print("\n4d. LINES DRAWN IN THE CAMERA PICTURE")
    try:
        matrix = read_homography(gated.asset("homography"), match.n_frames)
    except (OSError, ValueError):
        print("  the homography file could not be read: skipped")
        return
    x, y, decimals = read_ball_text(gated.asset("atd_raw"), match.n_frames)
    graded_frames = np.zeros(match.n_frames, dtype=bool)
    for start, stop in match.halves:
        graded_frames[start:stop] = True
    x[~graded_frames], y[~graded_frames] = np.nan, np.nan
    p = picture_figures(matrix, x, y)
    print(f"  {'On exactly straight, evenly spaced lines':<50} {share_words(p['share'])}")
    print(f"  {'Longest such line':<50} {length_words(p['longest_s'])}")
    print(f"  {'File precision fine enough for this test':<50} "
          f"{'yes' if decimals >= 5 else 'NO: treat these two lines as unreliable'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=("ball", "inputs", "output", "followup"))
    parser.add_argument("--model", default=SAVED_MODEL, help="saved v5 model for the output check")
    args = parser.parse_args()
    with open(RANGES) as handle:
        ranges = json.load(handle)
    if args.only in (None, "ball"):
        check_ball(ranges)
    if args.only in (None, "inputs"):
        check_inputs(ranges)
    if args.only in (None, "output"):
        check_output(ranges, args.model)
    if args.only == "followup":
        check_followup(ranges, args.model)
    print("\nWords only: safe to repeat to anyone. Nothing was written.")
