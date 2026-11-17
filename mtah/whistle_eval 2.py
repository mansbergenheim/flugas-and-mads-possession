"""Judge whistle events against SoccerNet's timed labels, which mark events, not whistles.

SoccerNet has no whistle labels. A foul is timed at the offence, an offside when the flag goes up,
a kick-off at the first pass; the referee's whistle normally follows a foul or offside within a few
seconds and comes just before a kick-off. A ball going out usually gets no whistle. So:

  found     share of fouls, offsides and kick-offs with a detected whistle in their window
  by chance the same share if the detections fell at random, at the rate seen in quiet play
  quiet     detections per minute in play far from any label: mostly false alarms
  ball out  share of ball-outs with a detection in the foul window: should stay low
  timing    median gap from the label to the detection, for the fouls found
  precision share of the detections that lie within 5 s of a labelled stoppage of any kind (foul,
            offside, ball out, throw-in, free kick, corner, goal kick, goal, card, substitution,
            kick-off), against the same share for detections slid 30 s to 5 min ("by chance")
  unexplained  detections with no such label within 5 s, per hour: the false alarms. SoccerNet does
            not label every whistle, so this counts some real whistles too; it errs on the safe side.

Not every foul gets a whistle (the referee may play advantage), so "found" cannot reach 100% even
for a perfect detector; the comparisons between settings are what count.
"""

import json
from typing import Dict, List, Sequence

import numpy as np

WINDOWS = {"Foul": (-1.0, 4.0), "Offside": (-1.0, 4.0), "Kick-off": (-4.0, 1.0)}
CONTROL = {"Ball out of play": (-1.0, 4.0)}
QUIET_MARGIN = 10.0          # seconds from any label before play counts as quiet
EXPLAIN_S = 5.0              # a detection this close to a labelled stoppage is explained by it
NOT_STOPPAGES = ("Shots on target", "Shots off target")
SLIDES = tuple(float(s) for s in np.concatenate([-np.arange(30, 330, 30), np.arange(30, 330, 30)]))


def explained(starts: np.ndarray, stop_times: np.ndarray) -> int:
    """How many detection starts have a labelled stoppage within EXPLAIN_S seconds."""
    if len(starts) == 0 or len(stop_times) == 0:
        return 0
    t = np.sort(stop_times)
    lo = np.searchsorted(t, starts - EXPLAIN_S, side="left")
    hi = np.searchsorted(t, starts + EXPLAIN_S, side="right")
    return int((hi > lo).sum())


def read_labels(path: str) -> Dict[int, List[dict]]:
    """{half: [{"label", "time"}]} with time in seconds from the start of that half's video."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    out: Dict[int, List[dict]] = {1: [], 2: []}
    for a in data.get("annotations", []):
        half = int(str(a["gameTime"]).split("-")[0].strip())
        if half in out:
            out[half].append({"label": a["label"], "time": int(a["position"]) / 1000.0})
    for half in out:
        out[half].sort(key=lambda x: x["time"])
    return out


def judge_half(events: Sequence[dict], labels: Sequence[dict], seconds: float) -> dict:
    """Counts for one half; add halves up with `combine`."""
    starts = np.array(sorted(e["start"] for e in events))
    times = np.array([x["time"] for x in labels])
    far = np.ones(len(starts), dtype=bool)
    if len(times) and len(starts):
        far = np.abs(starts[:, None] - times[None, :]).min(axis=1) > QUIET_MARGIN
    quiet_s = 0.0
    edges = np.concatenate(([0.0], times, [seconds]))
    for a, b in zip(edges[:-1], edges[1:]):
        lo, hi = (a + QUIET_MARGIN if a > 0 else 0.0), (b - QUIET_MARGIN if b < seconds else seconds)
        quiet_s += max(0.0, hi - lo)
    stops = np.array([x["time"] for x in labels if x["label"] not in NOT_STOPPAGES])
    by_chance = float(np.mean([explained((starts + d) % seconds, stops) for d in SLIDES])) if seconds > 0 else 0.0
    out = {"events": len(starts), "quiet_events": int(far.sum()), "quiet_s": quiet_s, "labels": {}, "timing": [],
           "explained": explained(starts, stops), "explained_chance": by_chance, "seconds": float(seconds)}
    for label in labels:
        window = WINDOWS.get(label["label"]) or CONTROL.get(label["label"])
        if window is None:
            continue
        lo, hi = label["time"] + window[0], label["time"] + window[1]
        inside = starts[(starts >= lo) & (starts <= hi)]
        cell = out["labels"].setdefault(label["label"], [0, 0])
        cell[0] += int(len(inside) > 0)
        cell[1] += 1
        if label["label"] == "Foul" and len(inside):
            out["timing"].append(float(inside[0] - label["time"]))
    return out


def combine(parts: Sequence[dict]) -> dict:
    total = {"events": 0, "quiet_events": 0, "quiet_s": 0.0, "labels": {}, "timing": [],
             "explained": 0, "explained_chance": 0.0, "seconds": 0.0}
    for p in parts:
        for key in ("events", "quiet_events", "quiet_s", "explained", "explained_chance", "seconds"):
            total[key] += p.get(key, 0)
        total["timing"] += p["timing"]
        for label, (hit, n) in p["labels"].items():
            cell = total["labels"].setdefault(label, [0, 0])
            cell[0] += hit
            cell[1] += n
    rate = total["quiet_events"] / total["quiet_s"] if total["quiet_s"] else 0.0
    total["quiet_per_min"] = 60.0 * rate
    total["share"] = {k: (h / n if n else None) for k, (h, n) in total["labels"].items()}
    width = {k: w[1] - w[0] for k, w in {**WINDOWS, **CONTROL}.items()}
    total["chance"] = {k: float(1 - np.exp(-rate * width[k])) for k in total["labels"]}
    total["median_timing"] = float(np.median(total["timing"])) if total["timing"] else None
    total["precision"] = total["explained"] / total["events"] if total["events"] else None
    total["precision_chance"] = total["explained_chance"] / total["events"] if total["events"] else None
    hours = total["seconds"] / 3600
    total["unexplained_per_hour"] = (total["events"] - total["explained"]) / hours if hours else None
    return total
