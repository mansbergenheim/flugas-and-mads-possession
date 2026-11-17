"""Use Metrica's Smart Tagging as outside evidence for the decoder.

Smart Tagging exists only for the competition match, so a model trained on other matches can't take
it as a feature. It enters as a small per-sample prior instead. Per-sample means it adds up over a
phase: 0.1 across a 10 s phase is 5 nats, more than a turnover costs. So weights are kept small, and
overlapping tags don't add up: they come from one automatic inference, so they share their errors,
and each kind of evidence counts once on any sample.

These weights can't be checked against real labels. They were chosen on open matches with simulated
tags that are deliberately mediocre; see simulate_tags and scripts/sweep_tags.py.

Per the organisers' notes: tags are automatic and imperfect, phases are not possessions, every
throw-in has Team N/A, and a set-piece tag is a window of a few seconds around the restart that opens
about 3.5 s before it. Team tags in the data read Barcelona and Atletic Club.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .match import A, D, H

ATTACKING = {"build up", "progression", "creating chances", "attacking transition", "long ball",
             "ball in final third", "ball in the box", "players in the box"}
DEFENDING = {"defending in attacking third", "defending in middle third",
             "defending in defensive third", "defensive transition"}


@dataclass(frozen=True)
class TagWeights:
    phase: float = 0.1            # per sample, towards the phase's team
    alive: float = 0.05           # per sample, against dead ball during a phase
    restart_dead: float = 0.3     # per sample, dead ball in the first part of a set-piece window
    restart_team: float = 0.2     # per sample, restarting team after the window
    restart_tail_frames: int = 50
    before_kickoff: float = 30.0  # per sample, dead ball before a half's kickoff tag


def team_code(tags: Dict[str, str]) -> Optional[int]:
    name = (tags.get("Team") or "").strip().lower()
    if "barcelona" in name:
        return H
    if "atletic" in name or "athletic" in name:
        return A
    return None


def halves_without_kickoff(tags: List[Dict], halves: Sequence[Tuple[int, int]],
                           early_frames: int = 250) -> List[int]:
    """Halves with no `Kick Off (Start)` tag. The window opens early because tags start before the
    restart. Without this tag the dead frames before a kickoff get no help, and nothing else warns."""
    missing = []
    for h, (lo, hi) in enumerate(halves):
        found = any((t["code"] or "").strip().lower() == "set pieces" and
                    (t["tags"].get("Type") or "").strip().lower() == "kick off (start)" and
                    lo - early_frames <= t["start"] < hi for t in tags)
        if not found:
            missing.append(h)
    return missing


def tag_prior(tags: List[Dict], frames: np.ndarray, half: np.ndarray,
              halves: Sequence[Tuple[int, int]], weights: TagWeights = TagWeights()) -> np.ndarray:
    """[n, 3] log prior for the decoder, at the feature samples."""
    n = len(frames)
    phase, restart = np.zeros((n, 3), dtype=bool), np.zeros((n, 3), dtype=bool)
    alive, restart_dead, kickoff = (np.zeros(n, dtype=bool) for _ in range(3))
    for tag in tags:
        code = (tag["code"] or "").strip().lower()
        start, stop = tag["start"], tag["stop"]
        team = team_code(tag["tags"])
        inside = (frames >= start) & (frames < stop)
        if code in ATTACKING or code in DEFENDING:
            alive |= inside
            if team is not None:
                phase[inside, team if code in ATTACKING else (A if team == H else H)] = True
        elif code == "set pieces":
            kind = (tag["tags"].get("Type") or "").strip().lower()
            middle = start + int(0.6 * (stop - start))
            restart_dead |= (frames >= start) & (frames < middle)
            if team is not None:
                restart[(frames >= middle) & (frames < stop + weights.restart_tail_frames), team] = True
            if kind == "kick off (start)":
                for h, (lo, hi) in enumerate(halves):
                    if lo <= start < hi:
                        kickoff |= (half == h) & (frames < start)
    prior = weights.phase * phase + weights.restart_team * restart
    prior[:, D] += (weights.restart_dead * restart_dead + weights.before_kickoff * kickoff
                    - weights.alive * alive)
    return prior


def simulate_tags(labels: np.ndarray, rng, recall: float = 0.7, team_correct: float = 0.8,
                  restart_recall: float = 0.75, false_rate: float = 0.15, fps: int = 25) -> List[Dict]:
    """Made-up tags for open matches. Not real Smart Tagging, only a stand-in to size the weights.

    Spells become phases with sloppy edges and sometimes the wrong team, restarts become windows
    with team missing about half the time, and random false phases are added.
    """
    from . import metrics as m
    from .match import X
    runs = m.valid_runs(labels)
    names = {H: "Barcelona", A: "Atletic Club"}
    other = {H: A, A: H}
    out = []
    segs = m.segments(labels, runs)
    for i, (a, b, k) in enumerate(segs):
        if k in (H, A):
            if (b - a) / fps >= 3 and rng.random() < recall:
                team = k if rng.random() < team_correct else other[k]
                start = a + int(rng.normal(1.0, 2.0) * fps)
                stop = b + int(rng.normal(-1.0, 2.0) * fps)
                if stop > start:
                    attacking = rng.random() < 0.6
                    out.append({"code": "PROGRESSION" if attacking else "DEFENDING IN MIDDLE THIRD",
                                "start": start, "stop": stop,
                                "tags": {"Team": names[team if attacking else other[team]]}})
            if i and segs[i - 1][2] == D and rng.random() < restart_recall:
                start = a - int(rng.uniform(2.5, 4.5) * fps)
                team = names[k] if rng.random() < 0.5 else "N/A"
                out.append({"code": "SET PIECES", "start": start, "stop": start + 5 * fps,
                            "tags": {"Team": team, "Type": "Throw-in"}})
    n_false = int(false_rate * len(out))
    lo, hi = runs[0][0], runs[-1][1]
    for _ in range(n_false):
        start = int(rng.integers(lo, hi))
        out.append({"code": "PROGRESSION", "start": start, "stop": start + int(rng.uniform(5, 20) * fps),
                    "tags": {"Team": names[H if rng.random() < 0.5 else A]}})
    return out
