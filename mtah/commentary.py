"""Use the commentary as outside evidence for the decoder, like Smart Tagging.

Two streams, both read from the broadcast commentary on this computer by a local model:

- who has the ball (scripts/extract_possession_mentions.py): a line-up player named on the ball or
  winning it nudges his team; one losing it nudges the other team. Fouls in this stream count as
  dead ball.
- what happened (scripts/extract_commentary_events.py): a foul, offside, ball out, set piece, goal,
  substitution, card... reported as happened nudges dead ball; a foul or offside reported as NOT
  happened ("no offside, play on") nudges live play a little.

Neither stream decides a team from a team name: only from line-up names.

Times are seconds into the broadcast. They go to the tactical timeline through the broadcast
alignment (scripts/align_broadcast.py), and anything said while the broadcast showed a replay is
dropped, as are the model's own "replay", "uncertain", "not live" and "unclear" answers.

Every nudge is per sample (5 a second) over a short window, like Smart Tagging's, and each kind of
evidence counts once at any sample however many reports cover it: overlapping reports share their
errors. Where the two teams are both nudged at one sample, neither is. The team strength, 0.5 per
sample over the 2 s before the name is said, is the one experiments/name_mentions.py found helpful
on all ten open matches even with a quarter of the names pointing at the wrong team. None of these
numbers were chosen on the competition match.

The gate. Before the commentary may be used, it must agree with the model where the model is sure:
at samples where the trees + TCN give one team at least SURE, the commentary's team must match at
least GATE of the time over at least MIN_SAMPLES samples. Near a half it is noise; far under a half,
something is swapped. Both switch the commentary off. The gate uses no labels.

Sure cues only (decided 17 Nov, after the real run). The local model called almost every mention
"not live", so the who-has-the-ball stream is cut down to the mentions where there is evidence the
player is on the ball: the model said so, or a ball-action word ("passes", "shoots", "header"...)
stands right next to the name (sure_cues). That leaves too few for the gate's MIN_SAMPLES, so the
gate's main job, catching a home/away mix-up, is done first by side_check: the line-up graphics'
home team must be the tracking file's first team (the one the labels call H), by team name and shirt
numbers. Only when it agrees does the gate judge over as few as SMALL_MIN samples.
"""

import csv
import os
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .align import REPLAY
from .match import A, D, H
from .names import norm

FPS = 25
DEAD_LABELS = {"goal", "foul", "card", "penalty", "corner", "free_kick", "throw_in", "goal_kick", "offside",
               "ball_out", "substitution", "injury", "var", "kickoff", "half_end"}
DENIABLE = {"foul", "offside", "penalty", "ball_out"}
SURE, GATE, MIN_SAMPLES = 0.9, 0.70, 200
NUMBER_MARGIN = 3
SMALL_MIN = 25                      # sure samples the gate needs once side_check agrees (about two cues)
TEAM_ROLES = ("on_ball", "wins_ball", "loses_ball")
# Words that put the named player on the ball when they stand right next to the name. Words where
# the player may be on either side of the duel (tackle, foul, challenge) are left out on purpose.
ACTION = re.compile(r"\b(pass|passes|passed|plays it|play it|crosses|cross|shoots|shot|strikes|header|heads|"
                    r"dribbl\w*|carries|brings it|takes it|on the ball|with the ball|picks out|finds|clears|"
                    r"cleared|wins it|intercept\w*|lays it|knocks it|chips|fires|curls|turns|goes past|"
                    r"skips past|runs at)\b")
ACTION_BEFORE, ACTION_AFTER = 25, 35  # characters of the line looked at before and after the name


@dataclass(frozen=True)
class CommentaryWeights:
    team: float = 0.5               # per sample, towards the team named on the ball
    team_before_s: float = 2.0      # ...over this long before the name is said
    team_after_s: float = 0.4       # ...and this long after (subtitle timing is loose)
    lost_s: float = 1.0             # a loss nudges the other team this long either side
    dead: float = 0.15              # per sample, dead ball around a reported stoppage
    dead_before_s: float = 1.0
    dead_after_s: float = 4.0
    alive: float = 0.05             # per sample, against dead ball where a foul/offside is denied
    alive_before_s: float = 3.0
    alive_after_s: float = 1.0


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def near_action(line: str, said: str) -> bool:
    k = line.lower().find(said.lower())
    return k >= 0 and bool(ACTION.search(line[max(0, k - ACTION_BEFORE):k + len(said) + ACTION_AFTER].lower()))


def sure_cues(rows: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """Only the mentions with evidence of who has the ball: the model's own on/wins/loses-ball answers,
    and any other mention (not a replay) with a ball-action word next to the name, read as on the ball."""
    kept, counts = [], {"model": 0, "words": 0, "dropped": 0}
    for r in rows:
        if r.get("role") in TEAM_ROLES:
            kept.append(dict(r, cue="model"))
            counts["model"] += 1
        elif r.get("role") != "replay" and near_action(r.get("line", ""), r.get("said", "")):
            kept.append(dict(r, role="on_ball", cue="words"))
            counts["words"] += 1
        else:
            counts["dropped"] += 1
    return kept, counts


def _team_alike(a: str, b: str) -> float:
    """How much two team names share: the share of the shorter name's words (4+ letters) found in the other."""
    wa = {w for w in norm(a).split() if len(w) >= 4}
    wb = {w for w in norm(b).split() if len(w) >= 4}
    return len(wa & wb) / min(len(wa), len(wb)) if wa and wb else 0.0


def side_check(lineups_path: str, meta_path: str) -> Dict:
    """Is the line-up graphics' home team the tracking file's first team (H)? By team name and shirt numbers.
    "agrees", "swapped" or "unclear". Prints of the result hold no names."""
    with open(lineups_path, encoding="utf-8") as f:
        lineups = json.load(f)
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    names = [(t.findtext("Name") or "") for t in teams]
    numbers = {t.get("id"): set() for t in teams}
    for p in root.iter("Player"):
        n = (p.findtext("ShirtNumber") or "").strip()
        if p.get("teamId") in numbers and n.isdigit():
            numbers[p.get("teamId")].add(int(n))
    shirt = [numbers[t.get("id")] for t in teams]
    votes = {}
    straight = _team_alike(lineups["home"].get("team", ""), names[0]) + _team_alike(lineups["away"].get("team", ""), names[1])
    crossed = _team_alike(lineups["home"].get("team", ""), names[1]) + _team_alike(lineups["away"].get("team", ""), names[0])
    votes["names"] = "agrees" if straight > crossed else "swapped" if crossed > straight else "unclear"

    def nums(side):
        return {int(p["number"]) for g in ("starting", "others") for p in lineups[side].get(g, [])
                if str(p.get("number", "")).strip().isdigit()}
    home, away = nums("home"), nums("away")
    straight = len(home & shirt[0]) + len(away & shirt[1])
    crossed = len(home & shirt[1]) + len(away & shirt[0])
    # Shirt numbers repeat across squads (1 to 11 in both), so they vote only by a clear margin.
    votes["numbers"] = ("agrees" if straight - crossed >= NUMBER_MARGIN else
                        "swapped" if crossed - straight >= NUMBER_MARGIN else "unclear")
    decided = {v for v in votes.values() if v != "unclear"}
    verdict = decided.pop() if len(decided) == 1 else "unclear"
    return {"verdict": verdict, **votes}


def place(rows: Sequence[Dict[str, str]], pieces: np.ndarray, broadcast_state: Optional[np.ndarray],
          offset_s: float = 0.0) -> Tuple[List[Dict], int, int]:
    """Rows with a tactical "frame" added; rows the alignment does not cover, or said while the broadcast
    showed a replay, are left out. Returns (kept, outside, during_replay)."""
    kept, outside, replay = [], 0, 0
    for row in rows:
        b = int(round((float(row["time_s"]) + offset_s) * FPS))
        if broadcast_state is not None and 0 <= b < len(broadcast_state) and broadcast_state[b] == REPLAY:
            replay += 1
            continue
        frame = next((b + int(shift) for first, after, shift in pieces if first <= b < after), None)
        if frame is None:
            outside += 1
            continue
        kept.append(dict(row, frame=frame))
    return kept, outside, replay


def _window(frames: np.ndarray, half: np.ndarray, halves, frame: int, before_s: float, after_s: float) -> slice:
    """Samples from before_s before to after_s after a tactical frame, inside its half."""
    for h, (start, stop) in enumerate(halves):
        if start <= frame < stop:
            rows = np.flatnonzero(half == h)
            if not len(rows):
                break
            f = frames[rows]
            lo = rows[0] + int(np.searchsorted(f, frame - before_s * FPS, side="left"))
            hi = rows[0] + int(np.searchsorted(f, frame + after_s * FPS, side="right"))
            return slice(lo, hi)
    return slice(0, 0)


def masks(mentions: Sequence[Dict], events: Sequence[Dict], frames: np.ndarray, half: np.ndarray, halves,
          w: CommentaryWeights = CommentaryWeights()) -> Dict[str, np.ndarray]:
    n = len(frames)
    team = np.zeros((n, 2), dtype=bool)
    dead, alive = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    code = {"home": H, "away": A}
    for m in mentions:
        side, role = code.get(m.get("side", "")), m.get("role", "")
        if side is None:
            continue
        if role in ("on_ball", "wins_ball"):
            team[_window(frames, half, halves, m["frame"], w.team_before_s, w.team_after_s), side] = True
        elif role == "loses_ball":
            team[_window(frames, half, halves, m["frame"], w.lost_s, w.lost_s), 1 - side] = True
        elif role in ("fouled", "commits_foul"):
            dead[_window(frames, half, halves, m["frame"], w.dead_before_s, w.dead_after_s)] = True
    for e in events:
        label, status, basis = e.get("label", ""), e.get("status", ""), e.get("basis", "")
        if label in DEAD_LABELS and status == "happened" and basis != "speculation":
            dead[_window(frames, half, halves, e["frame"], w.dead_before_s, w.dead_after_s)] = True
        elif label in DENIABLE and status == "did_not_happen":
            alive[_window(frames, half, halves, e["frame"], w.alive_before_s, w.alive_after_s)] = True
    both = team.all(axis=1)
    team[both] = False                  # the two teams named at once: no team evidence there
    alive &= ~dead                      # a reported stoppage outweighs a denial nearby
    return {"team": team, "dead": dead, "alive": alive}


def prior_from(m: Dict[str, np.ndarray], w: CommentaryWeights = CommentaryWeights()) -> np.ndarray:
    """[n, 3] log prior for the decoder, to be added to Smart Tagging's."""
    prior = np.zeros((len(m["dead"]), 3))
    prior[:, H] = w.team * m["team"][:, 0]
    prior[:, A] = w.team * m["team"][:, 1]
    prior[:, D] = w.dead * m["dead"] - w.alive * m["alive"]
    return prior


def gate(m: Dict[str, np.ndarray], prob: np.ndarray, min_samples: int = MIN_SAMPLES) -> Dict:
    """Does the commentary's team agree with the model where the model is sure? Uses no labels."""
    named = m["team"].any(axis=1)
    sure = prob[:, (H, A)].max(axis=1) >= SURE
    pick = named & sure
    n = int(pick.sum())
    agree = float(np.mean(np.argmax(m["team"][pick], axis=1) == np.argmax(prob[pick][:, (H, A)], axis=1))) if n else float("nan")
    if n < min_samples:
        verdict = "too little"
    elif agree >= GATE:
        verdict = "on"
    elif agree <= 1 - GATE:
        verdict = "swapped"
    else:
        verdict = "noise"
    return {"samples": n, "agree": agree, "verdict": verdict,
            "covered_team": float(named.mean()), "covered_dead": float(m["dead"].mean())}


def load(mentions_path: Optional[str], events_path: Optional[str], alignment_path: str, frames: np.ndarray,
         half: np.ndarray, halves, w: CommentaryWeights = CommentaryWeights()) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Masks from the two streams' CSV files (either may be None), and counts for the printout."""
    with np.load(alignment_path) as z:
        pieces = z["pieces"].reshape(-1, 3)
        state = z["broadcast_state"] if "broadcast_state" in z.files else None
    info = {}
    placed = {}
    for name, path in (("mentions", mentions_path), ("events", events_path)):
        rows = read_csv(path) if path else []
        if name == "mentions":
            rows, info["cues"] = sure_cues(rows)
        kept, outside, replay = place(rows, pieces, state)
        placed[name] = kept
        info[name] = {"rows": len(rows), "kept": len(kept), "outside": outside, "replay": replay}
    return masks(placed["mentions"], placed["events"], frames, half, halves, w), info


def team_gate(m: Dict[str, np.ndarray], prob: np.ndarray, lineups_path: str, meta_path: str) -> Dict:
    """The side check first; the gate over SMALL_MIN samples if it agrees, MIN_SAMPLES if not, and the
    team evidence never if the sides look swapped."""
    sides = side_check(lineups_path, meta_path)
    check = gate(m, prob, SMALL_MIN if sides["verdict"] == "agrees" else MIN_SAMPLES)
    if sides["verdict"] == "swapped":
        check["verdict"] = "swapped"
    check["sides"] = sides
    return check


# The committed cues. The commentary chain (transcript, line-ups, local models, broadcast alignment)
# runs on a Mac and its language-model steps can differ on other hardware, so its end product is
# committed to the repository as small files: cues already placed on the tactical timeline, with no
# names and no commentary text, each row saying which rule made it. With the organisers' agreement
# the submission is rebuilt from the match files plus these; scripts/rebuild_cues.sh regenerates
# them from the match files on a Mac and checks they come out the same.
POSSESSION_CUES, STOPPAGE_CUES, CUE_MANIFEST = "possession_cues.csv", "stoppage_cues.csv", "manifest.json"
POSSESSION_COLUMNS = ["tactical_frame", "broadcast_time_s", "side", "role", "rule", "segment"]
STOPPAGE_COLUMNS = ["tactical_frame", "broadcast_time_s", "label", "status", "basis", "segment"]


def placed_cues(mentions_path: str, events_path: str, alignment_path: str) -> Tuple[List[Dict], List[Dict], Dict]:
    """The sure possession cues and the events, placed on the tactical timeline (load's first half)."""
    with np.load(alignment_path) as z:
        pieces = z["pieces"].reshape(-1, 3)
        state = z["broadcast_state"] if "broadcast_state" in z.files else None
    rows, counts = sure_cues(read_csv(mentions_path))
    poss, outside_p, replay_p = place(rows, pieces, state)
    events = read_csv(events_path)
    stop, outside_e, replay_e = place(events, pieces, state)
    info = {"cues": counts,
            "mentions": {"rows": len(rows), "kept": len(poss), "outside": outside_p, "replay": replay_p},
            "events": {"rows": len(events), "kept": len(stop), "outside": outside_e, "replay": replay_e}}
    return poss, stop, info


def write_cues(folder: str, poss: Sequence[Dict], stop: Sequence[Dict]) -> None:
    os.makedirs(folder, exist_ok=True)
    for name, columns, rows, source in ((POSSESSION_CUES, POSSESSION_COLUMNS, poss, "cue"),
                                        (STOPPAGE_CUES, STOPPAGE_COLUMNS, stop, None)):
        with open(os.path.join(folder, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
            w.writeheader()
            for r in sorted(rows, key=lambda r: (int(r["frame"]), r.get("segment", ""))):
                out = {"tactical_frame": int(r["frame"]), "broadcast_time_s": f"{float(r['time_s']):.2f}",
                       "segment": r.get("segment", "")}
                if source:
                    out.update(side=r["side"], role=r["role"], rule=r[source])
                else:
                    out.update(label=r.get("label", ""), status=r.get("status", ""), basis=r.get("basis", ""))
                w.writerow(out)


def read_cues(folder: str) -> Tuple[List[Dict], List[Dict], Dict]:
    """The committed cues as load() would have placed them, and their manifest."""
    def rows(name, columns):
        out = []
        for r in read_csv(os.path.join(folder, name)):
            if list(r) != columns:
                raise ValueError(f"{name}: expected columns {columns}")
            out.append(dict(r, frame=int(r["tactical_frame"]), time_s=r["broadcast_time_s"]))
        return out
    with open(os.path.join(folder, CUE_MANIFEST), encoding="utf-8") as f:
        manifest = json.load(f)
    return rows(POSSESSION_CUES, POSSESSION_COLUMNS), rows(STOPPAGE_CUES, STOPPAGE_COLUMNS), manifest


def gate_from_cues(m: Dict[str, np.ndarray], prob: np.ndarray, manifest: Dict) -> Dict:
    """team_gate, with the side check as recorded when the cues were made (the line-ups stay on the Mac)."""
    sides = manifest["side_check"]
    check = gate(m, prob, SMALL_MIN if sides["verdict"] == "agrees" else MIN_SAMPLES)
    if sides["verdict"] == "swapped":
        check["verdict"] = "swapped"
    check["sides"] = sides
    return check
