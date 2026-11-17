"""Metrica Sports sample games: anonymised tracking with hand-collected events, 25 fps.

These have no per-frame possession labels, so labels_from_events derives them. The rules follow
Metrica's own events-definitions.pdf, which matches the competition's wording closely: possession
passes on a RECOVERY or an opponent's action, an opponent's touch without significant impact does
not end it, and the ball is dead from going out or a whistled foul until the ball is put back in play.
"""

import csv
import json
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import numpy as np

from ..match import A, AWAY, D, H, HOME, Match, X

PITCH = (105.0, 68.0)
FPS = 25

# Event types that say this team has the ball from the event's first frame.
OWNING = {"SET PIECE", "PASS", "RECOVERY", "SHOT", "CARRY"}


def _to_metres(xy: np.ndarray) -> np.ndarray:
    out = np.empty_like(xy, dtype=np.float32)
    out[..., 0] = (xy[..., 0] - 0.5) * PITCH[0]
    out[..., 1] = (0.5 - xy[..., 1]) * PITCH[1]
    return out


def labels_from_events(events: List[Dict], n_frames: int, halves: List[Tuple[int, int]]) -> np.ndarray:
    """Per-frame H/A/D/X from event rows {team, type, subtype, start, end, half}.

    Frames are 0-based indices into the match. Frames before a half's first event and after its
    last are X, because no event says who had the ball there.
    """
    labels = np.full(n_frames, X, dtype=np.int8)
    for half, (lo, hi) in enumerate(halves):
        rows = sorted((e for e in events if e["half"] == half), key=lambda e: e["start"])
        if not rows:
            continue
        state, cursor = None, None
        for e in rows:
            kind, sub = e["type"], e["subtype"]
            start = int(np.clip(e["start"], lo, hi))
            end = int(np.clip(max(e["end"], e["start"]), lo, hi))
            if kind == "CARRY" and state == D:
                continue  # taggers log the restarter walking the ball back; play has not resumed
            if kind in OWNING and e["team"] in (H, A):
                new = e["team"]
                if cursor is not None and start > cursor:
                    labels[cursor:start] = state
                if new != state:
                    state, cursor = new, start
                elif cursor is None:
                    cursor = start
                if cursor < start:
                    cursor = start
                ends_play = kind == "SHOT" and ("GOAL" in sub or "OUT" in sub)
                if ends_play:
                    labels[cursor:end] = state
                    state, cursor = D, end
            elif kind == "BALL OUT" or (kind == "BALL LOST" and "OFFSIDE" in sub):
                if cursor is not None:
                    labels[cursor:end] = state
                state, cursor = D, end
            elif kind == "FAULT RECEIVED":
                if cursor is not None:
                    labels[cursor:start] = state
                state, cursor = D, start
        last = max(int(np.clip(max(e["end"], e["start"]), lo, hi)) for e in rows)
        if cursor is not None and last > cursor:
            labels[cursor:last] = state
    return labels


def _team_code(name: str) -> int:
    return H if name.strip().lower() in ("home", "team a") else A


def load_csv_game(home_csv: str, away_csv: str, events_csv: str, match_id: str) -> Match:
    def table(path):
        with open(path) as handle:
            rows = csv.reader(handle)
            next(rows)
            next(rows)
            header = next(rows)
            data = np.array([[float(v) if v else np.nan for v in row] for row in rows],
                            dtype=np.float64)
        return header, data

    h_head, home = table(home_csv)
    a_head, away = table(away_csv)
    n = min(len(home), len(away))
    home, away = home[:n], away[:n]
    if not np.array_equal(home[:, 1], away[:, 1]):
        raise ValueError(f"{match_id}: home and away frame columns differ")

    def xy(data):
        block = data[:, 3:]
        return block.reshape(len(data), -1, 2)

    home_xy, away_xy = xy(home), xy(away)
    ball = _to_metres(home_xy[:, -1, :])
    players = _to_metres(np.concatenate((home_xy[:, :-1, :], away_xy[:, :-1, :]), axis=1))
    team = np.array([HOME] * (home_xy.shape[1] - 1) + [AWAY] * (away_xy.shape[1] - 1), dtype=np.int8)

    period = home[:, 0].astype(int)
    halves = [(int(np.flatnonzero(period == p)[0]), int(np.flatnonzero(period == p)[-1]) + 1)
              for p in (1, 2) if (period == p).any()]
    first_frame = int(home[0, 1])

    events = []
    with open(events_csv) as handle:
        for row in csv.DictReader(handle):
            p = int(row["Period"])
            events.append({"team": _team_code(row["Team"]), "type": row["Type"],
                           "subtype": row["Subtype"] or "", "half": p - 1,
                           "start": int(row["Start Frame"]) - first_frame,
                           "end": int(row["End Frame"]) - first_frame})
    labels = labels_from_events(events, n, halves)
    return Match(match_id, "metrica", FPS, PITCH, ball, players, team, halves, labels)


def _parse_epts_meta(path: str):
    root = ET.parse(path).getroot()
    teams = [t.get("id") for t in root.iter("Team")]
    team_of_player = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    player_of_channel = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    params = {}
    for p in root.iter("ProviderParameter"):
        params.setdefault(p.findtext("Name"), p.findtext("Value"))
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        ids = []
        for ref in spec.iter("PlayerChannelRef"):
            channel = ref.get("playerChannelId")
            if channel.endswith("_x"):
                ids.append(player_of_channel[channel])
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), ids))
    return teams, team_of_player, specs, params


def load_epts_game(meta_path: str, raw_path: str, events_json: str, match_id: str) -> Match:
    teams, team_of_player, specs, params = _parse_epts_meta(meta_path)
    people = sorted(team_of_player)
    slot = {p: i for i, p in enumerate(people)}
    team = np.array([HOME if team_of_player[p] == teams[0] else AWAY for p in people], dtype=np.int8)

    end = max(s[1] for s in specs)
    players = np.full((end, len(people), 2), np.nan, dtype=np.float64)
    ball = np.full((end, 2), np.nan, dtype=np.float64)
    specs = sorted(specs)
    current = 0
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0])
            while current + 1 < len(specs) and frame > specs[current][1]:
                current += 1
            for pid, group in zip(specs[current][2], parts[1].split(";")):
                if group[0] != "N":
                    x, y = group.split(",")[:2]
                    players[frame - 1, slot[pid]] = (float(x), float(y))
            if parts[2][0] != "N":
                bx, by = parts[2].split(",")[:2]
                ball[frame - 1] = (float(bx), float(by))

    def span(a, b):
        return int(params[a]) - 1, int(params[b])

    halves = [span("first_half_start", "first_half_end"), span("second_half_start", "second_half_end")]

    with open(events_json) as handle:
        rows = json.load(handle)["data"]
    events = []
    for e in rows:
        sub = e.get("subtypes")
        if isinstance(sub, dict):
            sub = sub.get("name", "")
        elif isinstance(sub, list):
            sub = "-".join(s.get("name", "") for s in sub)
        team_id = (e.get("team") or {}).get("id")
        events.append({"team": H if team_id == teams[0] else A, "type": e["type"]["name"],
                       "subtype": sub or "", "half": int(e["period"]) - 1,
                       "start": int(e["start"]["frame"]) - 1,
                       "end": int((e.get("end") or e["start"])["frame"] or e["start"]["frame"]) - 1})
    labels = labels_from_events(events, end, halves)
    return Match(match_id, "metrica", FPS, PITCH, _to_metres(ball), _to_metres(players), team,
                 halves, labels)
