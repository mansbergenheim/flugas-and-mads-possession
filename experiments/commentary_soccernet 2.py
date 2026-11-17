"""How reliable is the commentary extractor? Measured on SoccerNet LaLiga matches, never on ours.

The extractor (scripts/extract_commentary_events.py, local Qwen through Ollama) turns a commentary
transcript into events: label, status (happened / did not / uncertain / replay), basis, team, player.
Before its events can nudge the possession model we need to know, on real matches with answers:

  1. when the commentary says something happened, how often it did (and against chance);
  2. how late the commentary is: the time from the event to the line that reports it;
  3. how often a named team is the right one;
  4. whether "did not happen" and "replay" really are not live events.

Data, all open (research use under the SoccerNet NDA; nothing from these matches leaves ~/soccernet
except the aggregate numbers this script prints):
  - transcripts: SoccerNet-Echoes, whisper_v3 (original language), ~/soccernet/echoes/
  - answers: SoccerNet Labels-v2.json, ~/soccernet/spain_laliga/<season>/<game>/
Both are timed on the same half videos, so a transcript second and a label millisecond are one clock.

Rule fixed on 16 Nov 2026 before any run (do not change after seeing results):
  - The extractor runs exactly as on the real match: same prompt, model qwen3.5:4b, default settings
    (60 s windows, 10 s overlap, model reloaded before every window, merge 10 s, lag 0).
  - An event "matches" a label when the label is of a kind the event's label maps to (KINDS) and the
    commentary line starts between 10 s before and 20 s after the label (NEAR). The looser test
    "some stoppage of any kind nearby" (STOPPAGES) is reported beside it, because for the possession
    model a dead ball matters more than its kind.
  - Chance: the same events slid by 1 to 10 minutes either way (SLIDES), wrapping within the half.
  - The matches are sorted by name; even positions are "tune", odd are "check". Anything chosen from
    these results later (a commentary lag, a prompt change, a weight) is chosen on tune and reported
    on check. This first run changes nothing, so it reports both and all.
  - turnover, injury, var and half_end have no SoccerNet label: counted, not judged.
  - Sample (fixed 16 Nov, after timing one 10-minute test and before any results): the extractor
    needs about 30 s per minute of commentary, about an hour a match, so the first run covers the
    first 12 matches in name order (6 tune, 6 check). More are added later in the same order.

    python3 experiments/commentary_soccernet.py --run            # extract (hours; resumable)
    python3 experiments/commentary_soccernet.py --run --limit 1  # one match first
    python3 experiments/commentary_soccernet.py                  # score what has been extracted
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOCCERNET = os.path.expanduser("~/soccernet")
ECHOES = os.path.join(SOCCERNET, "echoes", "Dataset", "whisper_v3", "spain_laliga")
WORK = os.path.join(SOCCERNET, "commentary_work")
EXTRACTOR = os.path.join(ROOT, "scripts", "extract_commentary_events.py")
OUT = os.path.join(ROOT, "runs", "commentary_soccernet")

NEAR = (-10.0, 20.0)
SLIDES = tuple(float(s) for s in list(range(-600, -59, 60)) + list(range(60, 601, 60)))
KINDS = {
    "goal": {"Goal"},
    "shot": {"Shots on target", "Shots off target", "Goal"},
    "save": {"Shots on target"},
    "foul": {"Foul"},
    "card": {"Yellow card", "Red card", "Yellow->red card"},
    "penalty": {"Penalty"},
    "corner": {"Corner"},
    "free_kick": {"Direct free-kick", "Indirect free-kick"},
    "throw_in": {"Throw-in"},
    "goal_kick": {"Ball out of play"},
    "offside": {"Offside"},
    "ball_out": {"Ball out of play"},
    "substitution": {"Substitution"},
    "kickoff": {"Kick-off"},
}
STOPPAGES = {"Ball out of play", "Throw-in", "Foul", "Indirect free-kick", "Direct free-kick", "Corner",
             "Offside", "Goal", "Kick-off", "Substitution", "Yellow card", "Red card", "Yellow->red card",
             "Penalty"}
LIVE = ("happened",)


def games():
    """Matches with both a transcript and labels, sorted by name, with their tune/check part."""
    labels = {os.path.basename(os.path.dirname(p)): p
              for p in glob.glob(os.path.join(SOCCERNET, "spain_laliga", "*", "*", "Labels-v2.json"))}
    talk = {os.path.basename(d): d for d in glob.glob(os.path.join(ECHOES, "*", "*"))}
    names = sorted(set(labels) & set(talk))
    return [{"name": n, "labels": labels[n], "talk": talk[n], "part": "tune" if i % 2 == 0 else "check"}
            for i, n in enumerate(names)]


def transcript(game, half):
    """Echoes' {"segments": {i: [start, end, text]}} as the whisper JSON the extractor reads."""
    path = os.path.join(game["talk"], f"{half}_asr.json")
    if not os.path.exists(path):
        return None
    segs = json.load(open(path, encoding="utf-8"))["segments"]
    out = [{"start": float(v[0]), "end": float(v[1]), "text": str(v[2])}
           for _, v in sorted(segs.items(), key=lambda kv: int(kv[0])) if str(v[2]).strip()]
    folder = os.path.join(WORK, game["name"])
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, f"{half}_transcript.json")
    with open(target, "w", encoding="utf-8") as f:
        json.dump({"segments": out}, f, ensure_ascii=False)
    return target


def events_path(game, half):
    return os.path.join(WORK, game["name"], f"{half}_events.csv")


def extract(game, half):
    """Run the extractor on one half unless it has finished before (its manifest is written last)."""
    out = events_path(game, half)
    if os.path.exists(out.replace(".csv", ".manifest.json")):
        return "done before"
    source = transcript(game, half)
    if source is None:
        return "no transcript"
    started = time.time()
    result = subprocess.run([sys.executable, EXTRACTOR, source, "-o", out], capture_output=True, text=True)
    if result.returncode != 0:
        tail = result.stderr.strip().splitlines()[-1:] or ["?"]
        return f"failed: {tail[0]}"
    return f"{(time.time() - started) / 60:.1f} min"


def read_labels(game):
    data = json.load(open(game["labels"], encoding="utf-8"))
    by_half = defaultdict(list)
    for a in data["annotations"]:
        half = int(a["gameTime"].split(" - ")[0])
        by_half[half].append({"t": int(a["position"]) / 1000.0, "label": a["label"], "team": a["team"]})
    return by_half, data["gameHomeTeam"], data["gameAwayTeam"]


def read_events(game, half):
    path = events_path(game, half)
    if not os.path.exists(path.replace(".csv", ".manifest.json")):
        return None
    with open(path, encoding="utf-8") as f:
        return [dict(r, t=float(r["time_s"])) for r in csv.DictReader(f)]


def words(name):
    return [w for w in "".join(c.lower() if c.isalnum() else " " for c in name).split() if len(w) > 2
            and w not in ("club", "real", "atl", "cf", "fc", "cd", "ud", "rcd", "sd")]


def side_of(team, home, away):
    """home or away when the spoken team clearly names one of them, else None."""
    said = words(team)
    if not said:
        return None

    def alike(name):
        return max((SequenceMatcher(None, a, b).ratio() for a in said for b in words(name) or [""]), default=0.0)

    h, a = alike(home), alike(away)
    if max(h, a) >= 0.8 and abs(h - a) >= 0.1:
        return "home" if h > a else "away"
    return None


def nearest(t, labels, kinds, slide=0.0, length=None):
    """The label of one of these kinds nearest after the window rule, or None."""
    best = None
    for lab in labels:
        if lab["label"] not in kinds:
            continue
        lt = lab["t"] + slide
        if length:
            lt %= length
        d = t - lt
        if NEAR[0] <= d <= NEAR[1] and (best is None or abs(d) < abs(best[0])):
            best = (d, lab)
    return best


def score(games_done):
    rows = []   # one per judged event
    unjudged = Counter()
    for game in games_done:
        by_half, home, away = read_labels(game)
        for half in (1, 2):
            events = read_events(game, half)
            if events is None:
                continue
            labels = by_half.get(half, [])
            length = max([e["t"] for e in events] + [lab["t"] for lab in labels] + [1.0]) + 1.0
            for e in events:
                kinds = KINDS.get(e["label"])
                if kinds is None:
                    unjudged[e["label"]] += 1
                    continue
                hit = nearest(e["t"], labels, kinds)
                stop = nearest(e["t"], labels, STOPPAGES)
                chance = np.mean([nearest(e["t"], labels, kinds, s, length) is not None for s in SLIDES])
                chance_stop = np.mean([nearest(e["t"], labels, STOPPAGES, s, length) is not None for s in SLIDES])
                said = side_of(e["team"], home, away) if e["team"] else None
                truth = hit[1]["team"] if hit and hit[1]["team"] in ("home", "away") else None
                rows.append({"game": game["name"], "part": game["part"], "label": e["label"],
                             "status": e["status"], "basis": e["basis"], "p": float(e["p_event"]),
                             "hit": hit is not None, "lag": hit[0] if hit else None,
                             "stop": stop is not None, "chance": chance, "chance_stop": chance_stop,
                             "said": said, "truth": truth, "player": bool(e["player"])})
    return rows, unjudged


def summary(rows, label=None, status=None, part=None, basis=None):
    sel = [r for r in rows if (label is None or r["label"] == label) and (status is None or r["status"] == status)
           and (part is None or r["part"] == part) and (basis is None or r["basis"] == basis)]
    if not sel:
        return None
    lags = [r["lag"] for r in sel if r["lag"] is not None]
    teamed = [r for r in sel if r["said"] and r["truth"]]
    return {"n": len(sel), "hit": float(np.mean([r["hit"] for r in sel])),
            "chance": float(np.mean([r["chance"] for r in sel])),
            "stop": float(np.mean([r["stop"] for r in sel])),
            "chance_stop": float(np.mean([r["chance_stop"] for r in sel])),
            "lag_q": [float(x) for x in np.percentile(lags, [25, 50, 75])] if lags else None,
            "team_n": len(teamed),
            "team_same": float(np.mean([r["said"] == r["truth"] for r in teamed])) if teamed else None,
            "player_share": float(np.mean([r["player"] for r in sel]))}


def line(name, s):
    if s is None:
        return f"  {name:<28} none"
    lag = "" if s["lag_q"] is None else f"  lag {s['lag_q'][1]:+.0f} s ({s['lag_q'][0]:+.0f} to {s['lag_q'][2]:+.0f})"
    team = "" if not s["team_n"] else f"  team agrees {s['team_same']:.0%} of {s['team_n']}"
    return (f"  {name:<28} {s['n']:5d}  same kind {s['hit']:4.0%} (chance {s['chance']:3.0%})"
            f"  any stoppage {s['stop']:4.0%} (chance {s['chance_stop']:3.0%}){lag}{team}")


def report(rows, unjudged, n_games):
    out = [f"{n_games} SoccerNet LaLiga matches extracted (whisper_v3 transcripts, qwen3.5:4b, default settings)",
           f"window: commentary line from {NEAR[0]:+.0f} s to {NEAR[1]:+.0f} s of the label; chance = events slid 1-10 min",
           "", "by status (all labels judged):"]
    for status in ("happened", "did_not_happen", "uncertain", "replay_or_recap"):
        out.append(line(status, summary(rows, status=status)))
    out += ["", "happened, by basis:"]
    for basis in ("referee_decision", "described_live", "inferred", "speculation"):
        out.append(line(basis, summary(rows, status="happened", basis=basis)))
    out += ["", "happened, by label:"]
    for label in sorted(KINDS):
        out.append(line(label, summary(rows, label=label, status="happened")))
    out += ["", "happened, tune and check halves of the matches:"]
    for part in ("tune", "check"):
        out.append(line(part, summary(rows, status="happened", part=part)))
    out += ["", "not judged (no SoccerNet label of that kind): "
            + (", ".join(f"{k} {v}" for k, v in unjudged.most_common()) or "none")]
    return "\n".join(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="store_true", help="extract the matches not done yet")
    parser.add_argument("--limit", type=int, default=12, help="the first this many matches (default 12)")
    args = parser.parse_args()
    todo = games()[: args.limit] if args.limit else games()
    if args.run:
        for i, game in enumerate(todo, 1):
            for half in (1, 2):
                print(f"[{i}/{len(todo)}] {game['name'][:45]} half {half}: {extract(game, half)}", flush=True)
    done = [g for g in todo if read_events(g, 1) is not None or read_events(g, 2) is not None]
    if not done:
        sys.exit("nothing extracted yet: run with --run")
    rows, unjudged = score(done)
    text = report(rows, unjudged, len(done))
    print(text)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".txt", "w") as f:
        f.write(text + "\n")
    with open(OUT + ".json", "w") as f:
        json.dump({"games": len(done), "rows": len(rows), "unjudged": unjudged,
                   "by_status": {s: summary(rows, status=s) for s in ("happened", "did_not_happen", "uncertain", "replay_or_recap")},
                   "by_label": {k: summary(rows, label=k, status="happened") for k in KINDS},
                   "by_part": {p: summary(rows, status="happened", part=p) for p in ("tune", "check")}}, f, indent=1)
