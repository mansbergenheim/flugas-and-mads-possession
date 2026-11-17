"""Loaders for the gated competition match. Written from DATA.md, never run on the real files by Claude.

The T&Cs forbid showing this data, or extracts of it, to a hosted LLM. So:
  - keep it outside this project and point MTAH_DATA_DIR at it,
  - run anything that uses this module in your own terminal,
  - do not paste values from it into any AI tool.

data_dir() enforces the first rule. Nothing here prints data values, including in errors: BadTracking
reports counts and names only, so a failure is safe to paste anywhere.
"""

import collections
import csv
import math
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..match import AWAY, HOME, Match

GAME = "BAR-ATH"
N_FRAMES = 199319
N_BROADCAST_FRAMES = 164488          # DATA.md: the broadcast is on its own, shorter timeline
FPS = 25
PITCH = (105.3, 68.0)
PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Published in DATA.md and games.py: graded halves as video frames, half-open.
HALVES = [(11185, 90652), (110808, 190198)]
# Wide bands around what DATA.md publishes (ball on 59% of graded frames, 20.5 objects). They catch a
# file that parsed but is the wrong file or the wrong layout, not ordinary variation.
BALL_SHARE_RANGE = (0.30, 0.90)
OBJECTS_RANGE = (15.0, 26.0)
MIN_OBJECTS_PER_TEAM = 6.0
MIN_SHARE_NEAR_PITCH = 0.99


class BadTracking(ValueError):
    """The file isn't what DATA.md describes. Stop rather than predict from damaged input."""


def data_dir(explicit: Optional[str] = None) -> str:
    path = explicit or os.environ.get("MTAH_DATA_DIR")
    if not path:
        raise RuntimeError("Set MTAH_DATA_DIR to the folder holding tactical/ and broadcast/.")
    path = os.path.realpath(path)
    project = os.path.realpath(PROJECT)
    workspace = os.path.dirname(project)
    if path == workspace or path.startswith(workspace + os.sep):
        raise RuntimeError(
            "The gated match data must live outside this project folder, so that AI tools "
            "working here cannot read it. Move it (for example to ~/mtah-data).")
    return path


def asset(name: str, root: Optional[str] = None, source: str = "tactical") -> str:
    suffix = {"video": ".mp4", "atd_meta": "_atd.xml", "atd_raw": "_atd.txt",
              "homography": "_homography.csv", "smart_tagging": "_smart_tagging.xml"}[name]
    return os.path.join(data_dir(root), source, f"{GAME}_{source}{suffix}")


def track_teams(meta_path: str) -> np.ndarray:
    """HOME or AWAY per track slot. First team in the file is home, as in the organisers' baseline."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"expected 2 teams, found {len(team_ids)}")
    slots = [p.get("teamId") for p in root.iter("Player")]
    if not slots:
        raise ValueError("no Player entries in metadata")
    unknown = sum(t not in team_ids for t in slots)
    if unknown:
        raise BadTracking(f"{unknown} track slots belong to neither team")
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def _point(group: str) -> Optional[Tuple[float, float]]:
    """(x, y) from one `x,y,speed` group, None if missing. Raises ValueError if it isn't either."""
    if group.startswith("N"):
        return None
    x, y = (float(v) for v in group.split(",")[:2])
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError
    return x, y


def parse_atd(raw_path: str, n_slots: int, n_frames: int = N_FRAMES):
    """(ball, players) in 0-1 pitch units, NaN where missing, indexed by video frame.

    Raw line: `<frame_id>:<slot>;<slot>;...:<ball>`, each group `x,y,speed`.
    video_frame = frame_id - 1.

    The file must hold every frame exactly once, as DATA.md says. Anything else raises BadTracking
    after the whole file is read, with a count of each problem. Blank lines are ignored.
    """
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    read = np.zeros(n_frames, dtype=bool)
    problems = collections.Counter()
    with open(raw_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.strip().split(":")
            try:
                frame_id, slots, ball_group = parts
                frame = int(frame_id) - 1
                points = [_point(g) for g in slots.split(";")]
                ball_point = _point(ball_group)
            except ValueError:
                problems["malformed lines"] += 1
                continue
            if not 0 <= frame < n_frames:
                problems["frames out of range"] += 1
            elif read[frame]:
                problems["duplicate frames"] += 1
            elif len(points) != n_slots:
                problems["lines with the wrong number of slots"] += 1
            else:
                read[frame] = True
                for slot, point in enumerate(points):
                    if point:
                        players[frame, slot] = point
                if ball_point:
                    ball[frame] = ball_point
    missing = int(n_frames - read.sum())
    if missing:
        problems["frames missing"] = missing
    if problems:
        found = ", ".join(f"{count} {name}" for name, count in sorted(problems.items()))
        raise BadTracking(f"tracking file doesn't match DATA.md: {found}")
    return ball, players


def normalised_to_metres(xy: np.ndarray, pitch=PITCH) -> np.ndarray:
    out = np.empty_like(xy)
    out[..., 0] = (xy[..., 0] - 0.5) * pitch[0]
    out[..., 1] = (0.5 - xy[..., 1]) * pitch[1]
    return out


def check_tracking(match: Match) -> None:
    """Refuse tracking that parsed cleanly but can't be this match. Names problems, no measurements."""
    graded = np.concatenate([np.arange(a, b) for a, b in match.halves])
    ball, players = match.ball[graded], match.players[graded]
    seen = np.isfinite(players[..., 0])
    limit = np.array(match.pitch, dtype=np.float32)  # the pitch, plus half its size on every side
    points = np.concatenate((players[seen], ball[np.isfinite(ball[:, 0])]))
    near = (np.abs(points) <= limit).all(axis=1).mean() if len(points) else 0.0
    problems = []
    if not BALL_SHARE_RANGE[0] <= np.isfinite(ball[:, 0]).mean() <= BALL_SHARE_RANGE[1]:
        problems.append("share of frames with a ball is far from DATA.md")
    if not OBJECTS_RANGE[0] <= seen.sum(1).mean() <= OBJECTS_RANGE[1]:
        problems.append("objects per frame are far from DATA.md")
    if min(seen[:, match.team == side].sum(1).mean() for side in (HOME, AWAY)) < MIN_OBJECTS_PER_TEAM:
        problems.append("one team has too few objects")
    if near < MIN_SHARE_NEAR_PITCH:
        problems.append("positions fall outside the pitch: check the coordinate units")
    if problems:
        raise BadTracking("; ".join(problems))


def load_atd_match(root: Optional[str] = None, meta_path: Optional[str] = None,
                   raw_path: Optional[str] = None, n_frames: int = N_FRAMES,
                   halves: Optional[list] = None, check: bool = True) -> Match:
    """The competition match in the shared format, with no labels.

    Explicit paths and check=False exist so the tests can feed a tiny synthetic file.
    """
    meta_path = meta_path or asset("atd_meta", root)
    raw_path = raw_path or asset("atd_raw", root)
    team = track_teams(meta_path)
    ball, players = parse_atd(raw_path, len(team), n_frames)
    match = Match(GAME, "atd", FPS, PITCH, normalised_to_metres(ball), normalised_to_metres(players),
                  team, halves or HALVES)
    if check:
        check_tracking(match)
    return match


def load_homography(path: str, n_frames: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(valid [n], reprojection error [n], matrices [n, 3, 3]) of one recording, by video frame.

    Row: frame, nine matrix values in row-major order, reprojection_error, valid. Where valid is 0 no
    pitch transform exists and the matrix is NaN. Every frame must be there exactly once.
    """
    valid = np.zeros(n_frames, dtype=bool)
    error = np.full(n_frames, np.nan, dtype=np.float32)
    matrices = np.full((n_frames, 3, 3), np.nan, dtype=np.float64)
    read = np.zeros(n_frames, dtype=bool)
    problems = collections.Counter()

    def number(text: str) -> float:
        return float(text) if text.strip() else float("nan")

    with open(path, newline="") as handle:
        rows = csv.reader(handle)
        column = {name.strip(): k for k, name in enumerate(next(rows, []))}
        needed = ["frame", "reprojection_error", "valid"] + [f"h{r}{c}" for r in range(3) for c in range(3)]
        missing = [name for name in needed if name not in column]
        if missing:
            raise BadTracking(f"homography file has no column named {', '.join(missing)}")
        for row in rows:
            if not row:
                continue
            try:
                frame = int(float(row[column["frame"]]))
                values = [number(row[column[f"h{r}{c}"]]) for r in range(3) for c in range(3)]
                flag = number(row[column["valid"]])
                miss = number(row[column["reprojection_error"]])
            except (ValueError, IndexError):
                problems["malformed lines"] += 1
                continue
            if not 0 <= frame < n_frames:
                problems["frames out of range"] += 1
            elif read[frame]:
                problems["duplicate frames"] += 1
            else:
                read[frame] = True
                valid[frame] = flag == 1.0 and all(math.isfinite(v) for v in values)
                error[frame] = miss
                matrices[frame] = np.array(values).reshape(3, 3)
    if n_frames - int(read.sum()):
        problems["frames missing"] = n_frames - int(read.sum())
    if problems:
        found = ", ".join(f"{count} {name}" for name, count in sorted(problems.items()))
        raise BadTracking(f"homography file doesn't match DATA.md: {found}")
    return valid, error, matrices


def load_broadcast(root: Optional[str] = None, n_frames: int = N_BROADCAST_FRAMES) -> Dict[str, np.ndarray]:
    """The broadcast's tracking on its own timeline: players and ball in metres, team per slot, and
    for each frame whether a pitch transform exists and how far it misses by."""
    team = track_teams(asset("atd_meta", root, "broadcast"))
    ball, players = parse_atd(asset("atd_raw", root, "broadcast"), len(team), n_frames)
    valid, error, _ = load_homography(asset("homography", root, "broadcast"), n_frames)
    return {"players": normalised_to_metres(players), "ball": normalised_to_metres(ball), "team": team,
            "valid": valid, "error": error}


def load_smart_tagging(path: Optional[str] = None, root: Optional[str] = None) -> List[Dict]:
    """[{code, start, stop, tags}] as video frames, stop exclusive. Case kept as in the file."""
    path = path or asset("smart_tagging", root)
    out = []
    for instance in ET.parse(path).iter("instance"):
        out.append({
            "code": instance.findtext("code"),
            "start": int(round(float(instance.findtext("start")) * FPS)),
            "stop": int(round(float(instance.findtext("end")) * FPS)),
            "tags": {label.findtext("group"): label.findtext("text")
                     for label in instance.findall("label")},
        })
    if not out:
        raise ValueError("no tags in the Smart Tagging file")
    return out
