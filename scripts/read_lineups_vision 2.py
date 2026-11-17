#!/usr/bin/env python3
"""Read the two line-ups off the broadcast's line-up graphics with a LOCAL vision model. Run this yourself.

    python3 scripts/read_lineups_vision.py

Nothing leaves this computer: frames go only to Ollama on localhost (the same qwen3.5:4b the
commentary extractor uses; it reads images). Nobody types a name in and nobody picks a frame by eye:
code finds the graphics, the local model reads them, and Apple's text recognition must agree.
The printout is words only. The line-ups are written to <data folder>/lineups/lineups.json, in the
format scripts/commentary_add_teams.py reads; open that file yourself to check it. If a name is
wrong, do not type a correction in: that would be hand-made data about the match. Change the method.

How it works
  1. Text recognition (scripts/ocr_video_text.swift, as read_lineups.py runs it) reads one frame
     every two seconds, from the start of the broadcast to a few minutes after kick-off. The reading
     is saved and reused (<data folder>/lineups/ocr_text.json).
  2. Where to look: stretches where at least MIN_NAMES name-like pieces of text are on screen at once
     (a list of players), frames at most MAX_GAP_S apart. The busiest stretches are kept, and from
     each the frames showing the most names (the graphic fully drawn in).
  3. Each of those frames is cut from the video with the pinned ffmpeg and given to the local model,
     which says whether it is a team line-up and, if so, copies the team name and every player name
     with shirt number and starter/substitute as shown. Greedy decoding, model reloaded before every
     frame, model pinned by digest; every raw answer is saved (vision_raw.jsonl), and --from-raw
     rebuilds the result from it without the model.
  4. Safety: a name is kept only if (a) the model read it in at least half of that graphic's line-up
     frames and (b) Apple's text recognition saw the same name (80% of the letters alike) in the same
     stretch. So two independent readers must agree, and a name the model "knows" but that is not on
     screen is dropped.
  5. Which team: the team name the model read, else the team name in the recognised text, both
     against the names in the tracking metadata; else the first graphic is home, the next away.
  6. Starters: the names the model marked as starting when that gives eleven; else, with more than
     eleven names, the eleven in the largest type; the rest are kept as "others" (normally the bench).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from mtah.audio import find_ffmpeg  # noqa: E402
from mtah.loaders import gated  # noqa: E402
from extract_commentary_events import http, model_digest, ollama_version, unload  # noqa: E402
from read_lineups import is_name, kickoff_on_broadcast, names_team, norm, read_text, team_names  # noqa: E402

MODEL = "qwen3.5:4b"
OPTIONS = {"temperature": 0, "seed": 0, "top_k": 1, "num_ctx": 8192, "num_predict": 2048}
MIN_NAMES = 8           # name-like pieces of text on screen at once before a frame may show a line-up
MAX_GAP_S = 6.0         # frames further apart belong to different stretches
MAX_STRETCHES = 6       # the busiest stretches looked at
FRAMES_EACH = 4         # frames per stretch given to the model
WIDTH = 1600            # frames are scaled down to at most this width before the model sees them
ALIKE = 0.80            # letters alike for two readings of one name
SAME = 0.85             # two model readings this alike are one name

SCHEMA = {
    "type": "object",
    "properties": {
        "is_lineup": {"type": "boolean"},
        "team": {"type": "string"},
        "players": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "number": {"type": "string"},
                           "role": {"enum": ["starter", "substitute", "unclear"]}},
            "required": ["name", "number", "role"]}},
    },
    "required": ["is_lineup", "team", "players"],
}
PROMPT = """This is one frame of a football (soccer) TV broadcast.

Does it show a team line-up graphic, that is a list of one team's players for this match? If not,
answer is_lineup false, team "" and no players.

If it does:
- team: the team name exactly as written on the graphic ("" if none is written).
- players: every player name exactly as written, in the order shown, with the shirt number if one
  is shown ("" if not), and role "starter" if the graphic shows the player as starting, "substitute"
  if as a substitute or on the bench, "unclear" if it does not say.
- Copy only what is visible in this image. Do not add, complete or correct names from your own
  knowledge of football. Leave out the coach and the referee."""


# ---------- where to look ----------

def stretches(frames: list[dict], team_words: set) -> list[list[dict]]:
    """Busy stretches: frames with many name-like texts, close in time; busiest first."""
    def count(frame):
        return sum(1 for it in frame["items"] if is_name(it["text"]) and not set(norm(it["text"]).split()) <= team_words)

    busy = [(f, count(f)) for f in frames]
    busy = [(f, c) for f, c in busy if c >= MIN_NAMES]
    groups: list[list[tuple]] = []
    for f, c in busy:
        if groups and (f["t_ms"] - groups[-1][-1][0]["t_ms"]) / 1000.0 <= MAX_GAP_S:
            groups[-1].append((f, c))
        else:
            groups.append([(f, c)])
    groups.sort(key=lambda g: (-max(c for _, c in g), -len(g), g[0][0]["t_ms"]))
    chosen = sorted(groups[:MAX_STRETCHES], key=lambda g: g[0][0]["t_ms"])
    out = []
    for g in chosen:
        best = sorted(g, key=lambda fc: (-fc[1], fc[0]["t_ms"]))[:FRAMES_EACH]
        out.append({"frames": sorted((f for f, _ in best), key=lambda f: f["t_ms"]),
                    "ocr": [f for f, _ in g]})
    return out


def frame_png(video: Path, t_s: float) -> bytes:
    command = [find_ffmpeg(), "-nostdin", "-v", "error", "-ss", f"{t_s:.3f}", "-i", str(video), "-frames:v", "1",
               "-vf", f"scale='min({WIDTH},iw)':-2", "-f", "image2pipe", "-vcodec", "png", "-"]
    result = subprocess.run(command, capture_output=True)
    if result.returncode or not result.stdout:
        sys.exit(f"could not cut a frame at {t_s:.1f} s from the video")
    return result.stdout


def ask(model: str, png: bytes) -> str:
    body = {"model": model, "format": SCHEMA, "options": OPTIONS, "stream": False, "think": False,
            "keep_alive": "5m",
            "messages": [{"role": "user", "content": PROMPT, "images": [base64.b64encode(png).decode()]}]}
    return http("/api/chat", body)["message"]["content"]


def parse(raw: str) -> dict:
    try:
        answer = json.loads(raw)
    except json.JSONDecodeError:
        return {"is_lineup": False, "team": "", "players": []}
    players = [p for p in answer.get("players", []) if isinstance(p, dict) and norm(str(p.get("name", "")))]
    return {"is_lineup": bool(answer.get("is_lineup")), "team": str(answer.get("team", "")), "players": players}


# ---------- combining ----------

def alike(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def on_screen(name: str, texts: list[tuple[str, int]]) -> tuple[bool, int]:
    """Did text recognition see this name? Whole text or any of its longer words; with the type size."""
    best, height = 0.0, 0
    long_words = [w for w in name.split() if len(w) >= 4] or name.split()
    for text, h in texts:
        score = max([alike(name, text)] + [alike(w, v) for w in long_words for v in text.split() if len(v) >= 4])
        if score > best:
            best, height = score, h
    return best >= ALIKE, height


def combine(stretch: dict, answers: list[dict]) -> dict | None:
    """One graphic: the names read in at least half of its line-up frames and seen by text recognition."""
    lineup = [a for a in answers if a["is_lineup"] and len(a["players"]) >= MIN_NAMES]
    if not lineup:
        return None
    clusters: list[dict] = []          # {"name", "frames": set, "numbers": Counter, "roles": Counter}
    for k, answer in enumerate(lineup):
        for p in answer["players"]:
            name = norm(str(p["name"]))
            if len(name) < 3:
                continue
            home = next((c for c in clusters if alike(c["name"], name) >= SAME), None)
            if home is None:
                home = {"name": name, "frames": set(), "numbers": Counter(), "roles": Counter(), "spellings": Counter()}
                clusters.append(home)
            home["frames"].add(k)
            home["spellings"][name] += 1
            number = re.sub(r"\D", "", str(p.get("number", "")))
            if number:
                home["numbers"][int(number)] += 1
            home["roles"][p.get("role", "unclear")] += 1
    texts = [(norm(it["text"]), it["h"]) for f in stretch["ocr"] for it in f["items"] if norm(it["text"])]
    players = []
    for c in clusters:
        if len(c["frames"]) < max(1, (len(lineup) + 1) // 2):
            continue
        name = sorted(c["spellings"].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        seen, height = on_screen(name, texts)
        if not seen:
            continue
        number = c["numbers"].most_common(1)[0][0] if c["numbers"] else None
        role = sorted(c["roles"].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        players.append({"name": name, "number": number, "role": role, "h": height})
    team_votes = Counter(norm(a["team"]) for a in lineup if norm(a["team"]))
    return {"start_s": stretch["frames"][0]["t_ms"] / 1000.0, "frames": len(lineup), "players": players,
            "read": sum(len(a["players"]) for a in lineup) / len(lineup),
            "team_read": team_votes.most_common(1)[0][0] if team_votes else "",
            "text": " ".join(t for t, _ in texts)}


def starters_of(players: list[dict]) -> tuple[list[dict], list[dict], str]:
    marked = [p for p in players if p["role"] == "starter"]
    if len(marked) == 11:
        return marked, [p for p in players if p["role"] != "starter"], "marked as starting on the graphic"
    if len(players) <= 11:
        return players, [], "all names on the graphic"
    ranked = sorted(players, key=lambda p: (-p["h"], p["name"]))
    return ranked[:11], ranked[11:], "the eleven in the largest type"


def deduce(graphics: list[dict], names: list[str]) -> tuple[dict, list[str]]:
    sides = ("home", "away")
    owner, how = [], []
    for g in graphics:
        by_model = [names_team(g["team_read"], n) for n in names] if g["team_read"] else [False, False]
        by_text = [names_team(g["text"], n) for n in names]
        if sum(by_model) == 1:
            owner.append(sides[by_model.index(True)]), how.append("the team name the model read")
        elif sum(by_text) == 1:
            owner.append(sides[by_text.index(True)]), how.append("the team name in the recognised text")
        else:
            owner.append(None), how.append("the order of the graphics (home first)")
    for k, o in enumerate(owner):
        if o is None:
            free = [s for s in sides if s not in {x for x in owner if x}]
            owner[k] = free[0] if free else None
    result, words = {}, []
    for side, team in zip(sides, names):
        mine = [(g, h) for g, o, h in zip(graphics, owner, how) if o == side]
        if not mine:
            result[side] = {"team": team, "starting": [], "others": [], "how": "no line-up graphic found"}
            words.append(f"{side:<5} no line-up graphic was read")
            continue
        g, decided = sorted(mine, key=lambda gh: (abs(len(starters_of(gh[0]["players"])[0]) - 11),
                                                   -gh[0]["frames"], gh[0]["start_s"]))[0]
        starters, others, which = starters_of(g["players"])
        known = {p["name"] for p in starters + others}
        for other, _ in mine:
            for p in other["players"]:
                if p["name"] not in known and not any(alike(p["name"], k) >= SAME for k in known):
                    others.append(p)
                    known.add(p["name"])
        clean = lambda ps: [{"number": p["number"], "name": p["name"]} for p in  # noqa: E731
                            sorted(ps, key=lambda p: (p["number"] is None, p["number"] or 0, p["name"]))]
        result[side] = {"team": team, "starting": clean(starters), "others": clean(others),
                        "how": f"local vision model + text recognition; team by {decided}; starters: {which}"}
        dropped = max(0, round(g["read"]) - len(g["players"]))
        count = "eleven starters" if len(starters) == 11 else f"only {len(starters)} starters (eleven expected)"
        words.append(f"{side:<5} {count} ({which}), {len(others)} others; team decided by {decided}; "
                     f"{'no' if not dropped else 'about ' + str(dropped)} name(s) per frame dropped because "
                     "text recognition did not see them or the model read them only once")
    return result, words


def agreement(new: dict, old_path: Path) -> list[str]:
    if not old_path.exists():
        return []
    old = json.loads(old_path.read_text(encoding="utf-8"))
    out = []
    for side in ("home", "away"):
        a = [p["name"] for p in new.get(side, {}).get("starting", [])]
        b = [p["name"] for p in old.get(side, {}).get("starting", [])]
        if a and b:
            same = sum(any(alike(x, y) >= ALIKE for y in b) for x in a)
            out.append(f"{side:<5} {same} of {len(a)} starters also in the earlier text-only reading ({len(b)} there)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="the data folder; default MTAH_DATA_DIR, then ~/mtah-data")
    ap.add_argument("--video", type=Path, default=None, help="default: <data folder>/broadcast/BAR-ATH_broadcast.mp4")
    ap.add_argument("--start", type=float, default=0.0, help="first second of the broadcast to look at")
    ap.add_argument("--end", type=float, default=None, help="last second; default: five minutes after kick-off")
    ap.add_argument("--step", type=float, default=2.0, help="seconds between the frames text recognition reads")
    ap.add_argument("--home", default=None, help="home team's name; default: from the tracking metadata")
    ap.add_argument("--away", default=None, help="away team's name; default: from the tracking metadata")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--from-raw", action="store_true", help="rebuild from the saved model answers, without the model")
    args = ap.parse_args()

    root = args.data or os.environ.get("MTAH_DATA_DIR") or os.path.expanduser("~/mtah-data")
    folder = Path(gated.data_dir(root)) / "lineups"
    folder.mkdir(parents=True, exist_ok=True)
    video = args.video or Path(gated.asset("video", root, "broadcast"))
    names = [args.home, args.away] if args.home and args.away else team_names(root)
    end = args.end if args.end is not None else kickoff_on_broadcast(root) + 300.0

    reading = folder / "ocr_text.json"
    wanted = {"start_ms": round(args.start * 1000), "end_ms": round(end * 1000), "step_ms": round(args.step * 1000)}
    saved = json.loads(reading.read_text(encoding="utf-8")) if reading.exists() else None
    if saved is None or any(saved.get(k) != v for k, v in wanted.items()):
        if not video.exists():
            sys.exit(f"{video} not found")
        print("1 READING      text recognition over the broadcast up to a few minutes after kick-off (a few minutes)",
              flush=True)
        read_text(video, reading, args.start, end, args.step)
        saved = json.loads(reading.read_text(encoding="utf-8"))
    else:
        print("1 READING      using the saved text recognition of the broadcast")

    team_words = {w for n in names for w in norm(n).split()}
    found = stretches(saved["frames"], team_words)
    n_frames = sum(len(s["frames"]) for s in found)
    print(f"2 WHERE        {len(found)} stretch(es) with a list of names on screen; {n_frames} frame(s) go to the model")
    if not found:
        print("VERDICT        no list of names was found on screen: nothing to read")
        return 1

    raw_path = folder / "vision_raw.jsonl"
    prompt_hash = hashlib.sha256((PROMPT + json.dumps(SCHEMA, sort_keys=True)).encode()).hexdigest()
    jobs = [(s, f) for s in found for f in s["frames"]]
    pngs = [frame_png(video, f["t_ms"] / 1000.0) for _, f in jobs] if not args.from_raw or video.exists() else None
    if args.from_raw:
        records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
        if len(records) != len(jobs) or any(r["t_ms"] != f["t_ms"] or r["prompt_sha256"] != prompt_hash
                                            for r, (_, f) in zip(records, jobs)):
            sys.exit(f"{raw_path} does not match these frames and this prompt")
        if pngs is not None and any(r["image_sha256"] != hashlib.sha256(p).hexdigest() for r, p in zip(records, pngs)):
            sys.exit("the frames cut now differ from the ones the saved answers were given")
        raws, digest = [r["answer"] for r in records], records[0]["model_digest"]
        print(f"3 MODEL        rebuilt from {len(raws)} saved answers")
    else:
        digest = model_digest(args.model)
        raws = []
        with raw_path.open("w", encoding="utf-8", newline="\n") as out:
            for n, ((_, f), png) in enumerate(zip(jobs, pngs), 1):
                unload(args.model)
                raw = ask(args.model, png)
                raws.append(raw)
                out.write(json.dumps({"t_ms": f["t_ms"], "image_sha256": hashlib.sha256(png).hexdigest(),
                                      "prompt_sha256": prompt_hash, "model": args.model, "model_digest": digest,
                                      "options": OPTIONS, "answer": raw}, sort_keys=True, ensure_ascii=False) + "\n")
                out.flush()
                print(f"\r3 MODEL        frame {n}/{len(jobs)}", end="", flush=True)
        print(f"\r3 MODEL        {len(jobs)} frames read by {args.model} ({digest[:12]}), Ollama {ollama_version()}")
    answers = [parse(r) for r in raws]
    print(f"               the model called {sum(a['is_lineup'] for a in answers)} of {len(answers)} frames a line-up graphic")

    per_stretch = defaultdict(list)
    for (s, _), a in zip(jobs, answers):
        per_stretch[id(s)].append(a)
    graphics = [g for g in (combine(s, per_stretch[id(s)]) for s in found) if g and len(g["players"]) >= MIN_NAMES]
    result, words = deduce(graphics, names)
    for k, line in enumerate(words):
        print(f"{'4 LINE-UPS' if not k else '':<15}{line}")

    target = folder / "lineups.json"
    earlier = folder / "lineups_ocr.json"
    if target.exists() and not earlier.exists():
        target.rename(earlier)                       # keep the text-only reading for comparison
    for k, line in enumerate(agreement(result, earlier)):
        print(f"{'5 COMPARED' if not k else '':<15}{line}")
    target.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"6 SAVED        {target}  (about the real match: keep it on this computer)")
    print(f"               sha256 {hashlib.sha256(target.read_bytes()).hexdigest()}; raw answers {raw_path.name}")
    good = all(len(result[s]["starting"]) == 11 for s in ("home", "away"))
    print("VERDICT        " + ("two elevens read by two readers that agree: open the file and check the names"
                               if good else "not two full elevens: open the file to see what was read"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
