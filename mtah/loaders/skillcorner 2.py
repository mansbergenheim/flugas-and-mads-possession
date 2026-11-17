"""SkillCorner open data: broadcast tracking at 10 fps, resampled to 25 Hz like everything else.

Tracking: only positions SkillCorner marks as detected are kept. The others are its own guesses for
players off camera, and real tracking has no guesses. The ball is kept only when detected too.

Labels come from SkillCorner's algorithms, not from people:
  - Alive while inside a phase of play, which SkillCorner defines only while the ball is in play.
    Everything else in a half is dead.
  - The team is the per-frame possession group. Frames with no group while alive (ball in flight,
    camera on a replay) keep the last team, as there is always a team in possession while the ball
    is alive. A stretch that starts with no group takes the team of its phase of play.
"""

import csv
import json
from typing import Dict, List, Tuple

import numpy as np

from ..match import A, AWAY, D, H, HOME, Match

SOURCE_FPS = 10
FPS = 25
MAX_PLAYERS = 48


def _read_tracking(path: str, last_frame: int, column: Dict[int, int], guesses: bool = False):
    """Arrays indexed by SkillCorner frame. column maps player id to a column, filled as seen."""
    n = last_frame + 1
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    players = np.full((n, MAX_PLAYERS, 2), np.nan, dtype=np.float32)
    group = np.full(n, -1, dtype=np.int8)
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            t = row["frame"]
            if t >= n:
                continue
            b = row.get("ball_data") or {}
            if b.get("is_detected") and b.get("x") is not None:
                ball[t] = (b["x"], b["y"])
            g = (row.get("possession") or {}).get("group")
            group[t] = {"home team": HOME, "away team": AWAY}.get(g, -1)
            for p in row.get("player_data") or ():
                if (guesses or p.get("is_detected")) and p.get("x") is not None:
                    col = column.setdefault(p["player_id"], len(column))
                    players[t, col] = (p["x"], p["y"])
    return ball, players, group


def _alive_team(phases_path: str, n: int, team_of_club: Dict[int, int]):
    alive = np.zeros(n, dtype=bool)
    phase_team = np.full(n, -1, dtype=np.int8)
    with open(phases_path) as handle:
        for row in csv.DictReader(handle):
            a, b = int(row["frame_start"]), min(int(row["frame_end"]), n)
            alive[a:b] = True
            phase_team[a:b] = team_of_club.get(int(row["team_in_possession_id"]), -1)
    return alive, phase_team


def source_labels(group: np.ndarray, alive: np.ndarray, phase_team: np.ndarray) -> np.ndarray:
    """H/A/D at the source frame rate. See the module docstring."""
    labels = np.full(len(group), D, dtype=np.int8)
    current = -1
    for t in range(len(group)):
        if not alive[t]:
            current = -1
            continue
        if group[t] >= 0:
            current = int(group[t])
        elif current < 0:
            current = int(phase_team[t])
        if current >= 0:
            labels[t] = H if current == HOME else A
    return labels


def _resample(values: np.ndarray, start: int, stop: int) -> Tuple[np.ndarray, np.ndarray]:
    """Linear in time between neighbouring source frames; NaN if either neighbour is missing."""
    k = np.arange(int((stop - start) * FPS / SOURCE_FPS))
    s = start + k * SOURCE_FPS / FPS
    i0 = np.floor(s).astype(int)
    w = (s - i0).reshape((-1,) + (1,) * (values.ndim - 1))
    i1 = np.minimum(i0 + 1, len(values) - 1)
    return ((1 - w) * values[i0] + w * values[i1]).astype(np.float32), np.rint(s).astype(int)


def load_skillcorner(match_json: str, tracking_path: str, phases_path: str, match_id: str,
                     guesses: bool = False) -> Match:
    """guesses=True keeps SkillCorner's estimates for players off camera. Only for diagnosis."""
    with open(match_json) as handle:
        meta = json.load(handle)
    home_club = meta["home_team"]["id"]
    team_of_club = {home_club: HOME, meta["away_team"]["id"]: AWAY}
    team_of_player = {p["id"]: team_of_club[p["team_id"]] for p in meta["players"]}
    periods = sorted((p["period"], p["start_frame"], p["end_frame"]) for p in meta["match_periods"]
                     if p["period"] in (1, 2))
    last = max(stop for _, _, stop in periods)

    column: Dict[int, int] = {}
    ball, players, group = _read_tracking(tracking_path, last, column, guesses)
    alive, phase_team = _alive_team(phases_path, last + 1, team_of_club)
    labels = source_labels(group, alive, phase_team)
    used = len(column)
    if used > MAX_PLAYERS:
        raise ValueError(f"{match_id}: {used} players, more than {MAX_PLAYERS}")
    players = players[:, :used]
    team = np.array([team_of_player[pid] for pid, _ in sorted(column.items(), key=lambda kv: kv[1])],
                    dtype=np.int8)

    parts_ball: List[np.ndarray] = []
    parts_players: List[np.ndarray] = []
    parts_labels: List[np.ndarray] = []
    halves = []
    offset = 0
    for _, start, stop in periods:
        b, nearest = _resample(ball, start, stop)
        p, _ = _resample(players, start, stop)
        parts_ball.append(b)
        parts_players.append(p)
        parts_labels.append(labels[nearest])
        halves.append((offset, offset + len(b)))
        offset += len(b)
    return Match(match_id=match_id, source="skillcorner", fps=FPS,
                 pitch=(float(meta["pitch_length"]), float(meta["pitch_width"])),
                 ball=np.concatenate(parts_ball), players=np.concatenate(parts_players), team=team,
                 halves=halves, labels=np.concatenate(parts_labels))
