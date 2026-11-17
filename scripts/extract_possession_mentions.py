#!/usr/bin/env python3
"""Who has the ball, from the commentary: player names found by code, their role read by a LOCAL LLM.

    python3 scripts/extract_possession_mentions.py TRANSCRIPT.srt -o mentions.csv

Nothing leaves this computer: the transcript goes only to Ollama on localhost. Run it yourself; it
prints words and hashes only.

Three steps, each doing what it is good at:
  1. Code finds the players (mtah/names.py): every word of the transcript that is a name from the
     line-up graphics (<data folder>/lineups/lineups.json), with strict rules, so a name that could
     belong to either team is never used. No team names or nicknames.
  2. The local model (qwen3.5:4b, the one the event extractor uses) is shown a minute of commentary
     with the found names marked, and says for each mention only what that player is doing now:
     on_ball, wins_ball, loses_ball, fouled, commits_foul, not_live, replay or unclear. It never
     decides the team and never adds names.
  3. Code gives the team: the side of the line-up the name belongs to.

Each mention is asked about once, in the window whose middle is nearest to it. The time of a
mention is its place in the subtitle line: a line from 10 s to 14 s with the name halfway through
gives 12 s. Times are seconds into the BROADCAST, like the event extractor's; mtah/commentary.py puts
them on the tactical timeline through the broadcast alignment and drops what was said during replays.

Reproducibility as in extract_commentary_events.py: greedy decoding, fixed seed and context, the
model reloaded before every window, its digest recorded, every raw answer saved
(OUT.raw.jsonl), and --from-raw rebuilds the CSV from them without the model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from extract_commentary_events import (OPTIONS, http, load_segments, model_digest, ollama_version,  # noqa: E402
                                       sha256_bytes, sha256_file, unload)
from mtah.loaders import gated  # noqa: E402
from mtah.names import Roster, counts  # noqa: E402

MODEL = "qwen3.5:4b"
ROLES = ["on_ball", "wins_ball", "loses_ball", "fouled", "commits_foul", "not_live", "replay", "unclear"]
SCHEMA = {
    "type": "object",
    "properties": {"mentions": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "string"}, "role": {"enum": ROLES}},
        "required": ["id", "role"]}}},
    "required": ["mentions"],
}
SYSTEM = """You read live football (soccer) commentary in English. Each line is "[line_id | seconds] text".
Some player names in the lines are mentions, listed at the end as "[mention_id] line_id: name".
For EVERY mention, say what that player is doing at that moment of the match:

on_ball      = the player has the ball now: receives it, controls it, carries it, passes, crosses,
               shoots, heads it on, dribbles, or takes a set piece.
wins_ball    = the player takes the ball from the other team now: tackle, interception, recovery,
               block that keeps the ball, or a goalkeeper catching or collecting it.
loses_ball   = the player loses the ball to the other team now: dispossessed, tackled, pass cut out
               or misplaced, gives it away, miscontrol.
fouled       = the player is fouled now.
commits_foul = the player commits a foul now.
not_live     = the player is named but is not acting in the play right now: statistics, history,
               opinions, praise, substitutions, injuries, line-ups, a player waiting for the ball,
               or a pass aimed at him that never arrives.
replay       = the commentators are describing a replay or an earlier moment, not live play.
unclear      = it cannot be told from these lines.

Rules:
1. Judge only the named player of each mention, at the moment of that line.
2. "A to B": A passed (on_ball) and B received it (on_ball), unless the lines say it did not arrive.
3. Use the lines around a mention for context, but a later line does not make an earlier mention live.
4. Do not decide teams. Do not add mentions. Answer every mention id exactly once.

Examples (invented, not from the input):
"Keller slides it through to Duarte" -> Keller on_ball, Duarte on_ball
"Duarte has scored eight this season" -> Duarte not_live
"and Morales nicks it off Keller's toe" -> Morales wins_ball, Keller loses_ball
"Keller goes down, the referee gives the free kick, Morales the man penalised" -> Keller fouled, Morales commits_foul
"Let's see that again, Duarte drifting in behind" -> Duarte replay
"Morales waiting at the far post, but it's cleared" -> Morales not_live"""


def windows(segs: list, length: float, overlap: float) -> list:
    """(start second, segment indices) per window, as the event extractor makes them."""
    out, t, last = [], segs[0]["start"], segs[-1]["end"]
    step = max(length - overlap, 1.0)
    while t <= last:
        idx = [i for i, s in enumerate(segs) if s["start"] < t + length and s["end"] > t]
        if idx:
            out.append((t, idx))
        t += step
    return out


def mention_time(seg: dict, start: int, stop: int) -> float:
    text = seg["text"]
    share = ((start + stop) / 2.0) / max(len(text), 1)
    return seg["start"] + share * max(seg["end"] - seg["start"], 0.0)


def ask(model: str, prompt: str) -> str:
    body = {"model": model, "format": SCHEMA, "options": OPTIONS, "stream": False, "think": False,
            "keep_alive": "30m", "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    return http("/api/chat", body)["message"]["content"]


def parse(raw: str) -> dict:
    try:
        items = json.loads(raw).get("mentions", [])
    except (json.JSONDecodeError, AttributeError):
        return {}
    out = {}
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and item.get("role") in ROLES and str(item.get("id", "")) not in out:
            out[str(item["id"]).strip()] = item["role"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcript", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("possession_mentions.csv"))
    ap.add_argument("--lineups", type=Path, default=None, help="default <data folder>/lineups/lineups.json")
    ap.add_argument("--data", default=None, help="the data folder; default MTAH_DATA_DIR, then ~/mtah-data")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--overlap", type=float, default=10.0)
    ap.add_argument("--from-raw", action="store_true", help="rebuild the CSV from the saved answers, without the model")
    args = ap.parse_args()
    root = args.data or os.environ.get("MTAH_DATA_DIR") or os.path.expanduser("~/mtah-data")
    lineups = args.lineups or Path(gated.data_dir(root)) / "lineups" / "lineups.json"
    if not lineups.exists():
        sys.exit(f"{lineups} not found: run scripts/read_lineups_vision.py first")
    roster = Roster.from_lineups(str(lineups))
    players, usable, dropped = counts(roster)
    print(f"1 LINE-UPS     {players} players; {usable} name words usable, {dropped} dropped because the teams could confuse them")

    segs = load_segments(args.transcript)
    if not segs:
        sys.exit("no lines found in the transcript")
    mentions = []
    for i, seg in enumerate(segs):
        for m in roster.find(seg["text"]):
            mentions.append({"id": f"m{len(mentions)}", "segment": i, "time": mention_time(seg, m.start, m.stop),
                             "said": m.said, "player": m.player, "side": m.side})
    wins = windows(segs, args.window, args.overlap)
    for m in mentions:          # each mention belongs to the window whose middle is nearest
        m["window"] = min(range(len(wins)), key=lambda k: (abs(wins[k][0] + args.window / 2 - m["time"]), k))
    asked = [k for k in range(len(wins)) if any(m["window"] == k for m in mentions)]
    print(f"2 NAMES        {len(mentions)} mentions of line-up players in {len(segs)} lines; "
          f"{len(asked)} of {len(wins)} minutes of commentary have one")

    prompts = {}
    for k in asked:
        lines = "\n".join(f"[s{i} | {segs[i]['start']:.1f}] {segs[i]['text']}" for i in wins[k][1])
        marks = "\n".join(f"[{m['id']}] s{m['segment']}: {m['said']}" for m in mentions if m["window"] == k)
        prompts[k] = f"{lines}\n\nMentions:\n{marks}"
    system_hash = sha256_bytes(SYSTEM.encode())
    raw_path = args.out.with_suffix(".raw.jsonl")
    if args.from_raw:
        records = {r["window"]: r for r in map(json.loads, raw_path.read_text(encoding="utf-8").splitlines())}
        if sorted(records) != asked or any(records[k]["prompt_sha256"] != sha256_bytes(prompts[k].encode())
                                           or records[k]["system_sha256"] != system_hash for k in asked):
            sys.exit(f"{raw_path} does not match this transcript, these line-ups and this prompt")
        answers = {k: records[k]["answer"] for k in asked}
        digest = records[asked[0]]["model_digest"] if asked else ""
        print(f"3 MODEL        rebuilt from {len(answers)} saved answers")
    else:
        digest = model_digest(args.model)
        answers = {}
        with raw_path.open("w", encoding="utf-8", newline="\n") as f:
            for n, k in enumerate(asked, 1):
                unload(args.model)
                answers[k] = ask(args.model, prompts[k])
                f.write(json.dumps({"window": k, "prompt_sha256": sha256_bytes(prompts[k].encode()),
                                    "system_sha256": system_hash, "model": args.model, "model_digest": digest,
                                    "answer": answers[k]}, ensure_ascii=False, sort_keys=True) + "\n")
                f.flush()
                print(f"\r3 MODEL        minute {n}/{len(asked)}", end="", file=sys.stderr, flush=True)
        print(f"\r3 MODEL        {len(asked)} minutes read by {args.model} ({digest[:12]})")

    missing = 0
    for m in mentions:
        role = parse(answers[m["window"]]).get(m["id"])
        missing += role is None
        m["role"] = role or "unclear"
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, lineterminator="\n", fieldnames=[
            "time_s", "segment", "segment_start", "segment_end", "side", "player", "said", "role", "line"])
        w.writeheader()
        for m in mentions:
            seg = segs[m["segment"]]
            w.writerow({"time_s": f"{m['time']:.2f}", "segment": f"s{m['segment']}", "segment_start": f"{seg['start']:.2f}",
                        "segment_end": f"{seg['end']:.2f}", "side": m["side"], "player": m["player"],
                        "said": m["said"], "role": m["role"], "line": seg["text"]})
    roles = {r: sum(m["role"] == r for m in mentions) for r in ROLES}
    useful = roles["on_ball"] + roles["wins_ball"] + roles["loses_ball"]
    share = lambda n: "most" if n >= 0.5 * max(len(mentions), 1) else "a third to half" if n >= len(mentions) / 3 \
        else "a few" if n else "none"  # noqa: E731
    print(f"4 ROLES        {share(useful)} of the mentions say who has the ball; {share(roles['not_live'])} are not live play; "
          f"{share(roles['replay'])} are replays; {share(roles['fouled'] + roles['commits_foul'])} are fouls"
          + (f"; the model skipped {missing} and they count as unclear" if missing else ""))
    if not args.from_raw:
        manifest = {"script_sha256": sha256_file(Path(__file__)), "system_prompt_sha256": system_hash,
                    "transcript_sha256": sha256_file(args.transcript), "lineups_sha256": sha256_file(lineups),
                    "model": args.model, "model_digest": digest, "ollama_version": ollama_version(),
                    "options": OPTIONS, "window": args.window, "overlap": args.overlap,
                    "raw_sha256": sha256_file(raw_path), "csv_sha256": sha256_file(args.out)}
        args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"5 SAVED        {args.out}  (about the real match: keep it on this computer)")
    print(f"               sha256 {hashlib.sha256(args.out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
