"""DFL open data (IDSSE, Bassek et al. 2025, CC BY 4.0): 25 fps TRACAB tracking with labels.

The ball's FrameSet carries BallPossession (1 home, 2 away) and BallStatus (1 alive, 0 dead) on
every frame. Label meaning relative to the competition is examined in scripts/inspect_labels.py.
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import numpy as np

from ..match import A, AWAY, D, H, HOME, Match, X

SECTIONS = ("firstHalf", "secondHalf")


def _match_info(path: str) -> Tuple[str, str, Tuple[float, float]]:
    root = ET.parse(path).getroot()
    general = next(root.iter("General"))
    env = next(root.iter("Environment"))
    return (general.get("HomeTeamId"), general.get("GuestTeamId"),
            (float(env.get("PitchX")), float(env.get("PitchY"))))


def _read_framesets(path: str) -> List[dict]:
    """Stream the file. One dict per FrameSet with numpy columns; XML is released as we go."""
    out = []
    n, x, y, poss, status = [], [], [], [], []
    context = ET.iterparse(path, events=("start", "end"))
    _, root = next(context)
    for event, elem in context:
        if event == "start":
            if elem.tag == "FrameSet":
                n, x, y, poss, status = [], [], [], [], []
            continue
        if elem.tag == "Frame":
            attrs = elem.attrib
            n.append(int(attrs["N"]))
            x.append(float(attrs["X"]))
            y.append(float(attrs["Y"]))
            poss.append(int(attrs.get("BallPossession", 0)))
            status.append(int(attrs.get("BallStatus", -1)))
            elem.clear()
        elif elem.tag == "FrameSet":
            out.append({"section": elem.get("GameSection"), "team": elem.get("TeamId"),
                        "person": elem.get("PersonId"),
                        "n": np.array(n, dtype=np.int64),
                        "xy": np.column_stack((x, y)).astype(np.float32),
                        "poss": np.array(poss, dtype=np.int8),
                        "status": np.array(status, dtype=np.int8)})
            elem.clear()
            root.clear()
    return out


def load_dfl(info_path: str, positions_path: str, match_id: str) -> Match:
    home_id, away_id, pitch = _match_info(info_path)
    sets = _read_framesets(positions_path)

    ball_sets: Dict[str, dict] = {s["section"]: s for s in sets if s["team"] == "BALL"}
    sections = [s for s in SECTIONS if s in ball_sets]
    if len(sections) != 2:
        raise ValueError(f"{match_id}: expected two halves, found {sorted(ball_sets)}")

    spans, offsets, total = {}, {}, 0
    for section in sections:
        n = ball_sets[section]["n"]
        spans[section] = (int(n.min()), int(n.max()) + 1)
        offsets[section] = total
        total += spans[section][1] - spans[section][0]

    ball = np.full((total, 2), np.nan, dtype=np.float32)
    labels = np.full(total, X, dtype=np.int8)
    for section in sections:
        s = ball_sets[section]
        idx = s["n"] - spans[section][0] + offsets[section]
        ball[idx] = s["xy"]
        code = np.where(s["poss"] == 1, H, np.where(s["poss"] == 2, A, X)).astype(np.int8)
        labels[idx] = np.where(s["status"] == 1, code, D)

    people = sorted({(s["person"], s["team"]) for s in sets if s["team"] in (home_id, away_id)})
    slot = {person: i for i, (person, _) in enumerate(people)}
    players = np.full((total, len(people), 2), np.nan, dtype=np.float32)
    for s in sets:
        if s["person"] in slot and s["section"] in spans:
            lo, hi = spans[s["section"]]
            keep = (s["n"] >= lo) & (s["n"] < hi)
            players[s["n"][keep] - lo + offsets[s["section"]], slot[s["person"]]] = s["xy"][keep]
    team = np.array([HOME if t == home_id else AWAY for _, t in people], dtype=np.int8)

    halves = [(offsets[s], offsets[s] + spans[s][1] - spans[s][0]) for s in sections]
    return Match(match_id, "dfl", 25, pitch, ball, players, team, halves, labels)
