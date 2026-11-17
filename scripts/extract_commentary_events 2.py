#!/usr/bin/env python3
"""Extract key match events from a commentary transcript with a LOCAL LLM (Ollama).

Nothing leaves this machine: the transcript goes only to the Ollama server on localhost.
No Python packages are needed beyond the standard library.

What the model decides, per event: which line it is on, the exact words that support it, the kind
of event, whether the commentator says it HAPPENED, DID NOT HAPPEN, is UNCERTAIN, or is a REPLAY,
and what that rests on (a referee decision, a live description, an inference, speculation).
What the code decides: the time (from the transcript, never from the model) and the probability
(from the fixed table P_BASIS below, never a number invented by the model).

Input: a timestamped transcript in one of these formats
  - whisper / mlx-whisper / whisper.cpp JSON  ({"segments": [{"start", "end", "text"}, ...]})
  - .srt or .vtt subtitles
Output, next to OUT.csv:
  OUT.csv            the events
  OUT.raw.jsonl      every raw model answer, with the hash of the prompt that produced it
  OUT.manifest.json  hashes of script, transcript, raw answers and CSV; model digest; Ollama version

Run (Ollama app open, model pulled):
  python3 scripts/extract_commentary_events.py TRANSCRIPT.json -o events.csv
Rebuild the CSV from the saved answers, on any machine, without the model:
  python3 scripts/extract_commentary_events.py TRANSCRIPT.json -o events.csv --from-raw

Reproducibility: decoding is greedy (temperature 0, top_k 1, fixed seed), the context size is fixed,
the model is reloaded before every window (so no answer depends on an earlier one; see unload),
and the model is pinned by digest. On the same Ollama version, model
digest and hardware type the raw answers repeat byte for byte; run twice and compare the printed
hashes. On other hardware they may differ, which is what --from-raw is for.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3.5:4b"
OPTIONS = {"temperature": 0, "seed": 0, "top_k": 1, "num_ctx": 8192, "num_predict": 2048}

LABELS = ["goal", "shot", "save", "foul", "card", "penalty", "corner", "free_kick", "throw_in",
          "goal_kick", "offside", "ball_out", "turnover", "substitution", "injury", "var",
          "kickoff", "half_end"]
STATUSES = ["happened", "did_not_happen", "uncertain", "replay_or_recap"]
BASES = ["referee_decision", "described_live", "inferred", "speculation"]

# How sure the commentary makes us, by what the statement rests on. These are fixed judgement
# values, not measured ones; tune them on open matches that have both commentary and event labels.
P_BASIS = {"referee_decision": 0.95, "described_live": 0.85, "inferred": 0.65, "speculation": 0.55}
BASIS_RANK = {b: r for r, b in enumerate(reversed(BASES))}  # referee_decision highest

# Property order is generation order: the model must copy the supporting words first, and only
# then choose the label and how certain it is.
SCHEMA = {
    "type": "object",
    "properties": {"events": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "segment_id": {"type": "string"},
            "quote": {"type": "string"},
            "label": {"enum": LABELS},
            "status": {"enum": STATUSES},
            "basis": {"enum": BASES},
            "team": {"type": "string"},
            "player": {"type": "string"},
        },
        "required": ["segment_id", "quote", "label", "status", "basis", "team", "player"],
    }}},
    "required": ["events"],
}

SYSTEM = """You read live football (soccer) broadcast commentary and extract match events.

Each input line is "[segment_id | seconds] text". Commentators mix live description of play with \
chat, statistics, history, opinions and general remarks about football. Extract only events of \
THIS match that the commentary is about at THIS moment. The commentary may be in any language.

For each event return:
- segment_id: the id of the line where the event is first mentioned.
- quote: the exact words copied from that line that support the event.
- label: one of """ + ", ".join(LABELS) + """.
- status:
  happened        = the commentator says it took place.
  did_not_happen  = the commentator says it did not take place ("no offside", "the flag stays \
down", "play goes on", "nothing given").
  uncertain       = they wonder, ask or disagree and nothing in these lines settles it.
  replay_or_recap = they describe a replay, an earlier moment or a summary, not live play.
- basis:
  referee_decision = an official decision is reported (whistle, flag up or kept down, card shown, \
goal given or ruled out, VAR outcome).
  described_live   = the commentator describes seeing it happen.
  inferred         = not said directly, but it follows from what is said.
  speculation      = an opinion, a question, a guess, or an appeal by players or crowd.
- team, player: as named in the line, otherwise "".

Rules:
1. A word is not an event. "Offside" inside a question, a denial or general football talk does \
not mean an offside happened.
2. Questions, rhetorical questions and appeals are never decisions.
3. General talk produces no event: season statistics, history, tactics in general, other matches, \
transfers, the weather, the crowd.
4. If a later line settles an earlier doubt, report the event once, with the settled status.
5. Never invent. If the lines contain no event, return an empty list.

Examples (invented lines, not from the input):
"Morales slides it through, Keller is away... and the flag goes up. Offside."
  -> label offside, status happened, basis referee_decision, player Keller
"Is he off? Doesn't appear to be offside, the flag stays down, and they play on."
  -> label offside, status did_not_happen, basis referee_decision
"You know, he was caught offside eleven times last season, more than anyone in the league."
  -> no event
"Looking at it again, you can see the defender clips his heel there."
  -> label foul, status replay_or_recap, basis described_live
"The crowd want a penalty! Was there a hand there? I'm not sure."
  -> label penalty, status uncertain, basis speculation
"Keller loses it in midfield and Duarte picks it up."
  -> label turnover, status happened, basis described_live, player Keller
"That one runs out of play, and it will be a throw."
  -> label throw_in, status happened, basis described_live"""


# ---------- helpers ----------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def http(path: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(HOST + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Ollama answered {error.code}: {error.read().decode(errors='replace')}") from None
    except urllib.error.URLError:
        sys.exit("cannot reach Ollama on localhost:11434; open the Ollama app (or run `ollama serve`)")


def model_digest(model: str) -> str:
    for m in http("/api/tags").get("models", []):
        if m.get("name") in (model, model + ":latest"):
            return m["digest"]
    sys.exit(f"model {model} is not pulled; run: ollama pull {model}")


def ollama_version() -> str:
    return http("/api/version").get("version", "unknown")


def unload(model: str) -> None:
    """Drop the model from memory and wait until it is gone.

    Ollama keeps the previous prompt's internal state and reuses it for the next one. That reuse
    changes the arithmetic slightly: a run started warm was measured to give different bytes
    from a run started cold. Unloading before every window (--reload every, the default) makes
    each answer depend only on its own prompt: not on the window before it, not on an earlier
    run, not on the order. Unloading only before the first window (--reload once) is 2-4 times
    faster and a full run still repeats byte for byte, but only as a whole, in the same order.
    """
    http("/api/generate", {"model": model, "keep_alive": 0})
    for _ in range(600):
        if not any(m.get("name") == model for m in http("/api/ps").get("models", [])):
            return
        time.sleep(0.1)
    sys.exit(f"could not unload {model}")


def ask(model: str, lines: str, think: bool) -> str:
    """Return the model's raw answer text for one window."""
    body = {"model": model, "format": SCHEMA, "options": OPTIONS, "stream": False, "think": think,
            "keep_alive": "30m",
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": lines}]}
    return http("/api/chat", body)["message"]["content"]


def parse_answer(raw: str) -> list[dict]:
    try:
        events = json.loads(raw).get("events", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    return [e for e in events if isinstance(e, dict)] if isinstance(events, list) else []


def squash(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def p_event(status: str, basis: str) -> float:
    """Probability that the event took place at this moment, from the fixed table."""
    sure = P_BASIS[basis]
    if status == "did_not_happen":
        return 1.0 - sure
    if status == "uncertain":
        return 0.5
    return sure


# ---------- transcript loading ----------

def _ts(s: str) -> float:
    parts = [float(p) for p in s.strip().replace(",", ".").split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, sec = parts
    return h * 3600 + m * 60 + sec


def load_segments(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        segs = data.get("segments", data.get("transcription")) if isinstance(data, dict) else data
        out = []
        for s in segs:
            if "start" in s:
                start, end = float(s["start"]), float(s.get("end", s["start"]))
            else:  # whisper.cpp -oj format
                start = s["offsets"]["from"] / 1000.0
                end = s["offsets"]["to"] / 1000.0
            out.append({"start": start, "end": end, "text": " ".join(s["text"].split())})
        return out
    out = []  # SRT / VTT
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        m = re.search(r"([\d:.,]+)\s*-->\s*([\d:.,]+)", block)
        if not m:
            continue
        body = " ".join(re.sub(r"<[^>]+>", "", block[m.end():]).split())
        if body:
            out.append({"start": _ts(m.group(1)), "end": _ts(m.group(2)), "text": body})
    return out


def windows(segs: list[dict], length: float, overlap: float):
    if not segs:
        return
    t, last = segs[0]["start"], segs[-1]["end"]
    step = max(length - overlap, 1.0)
    while t <= last:
        idx = [i for i, s in enumerate(segs) if s["start"] < t + length and s["end"] > t]
        if idx:
            yield idx
        t += step


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcript", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("commentary_events.csv"))
    ap.add_argument("--raw", type=Path, default=None,
                    help="raw model answers, one JSON line per window (default: OUT with .raw.jsonl)")
    ap.add_argument("--from-raw", action="store_true",
                    help="do not run the model; rebuild the CSV from the saved raw answers")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--expect-digest", default=None,
                    help="refuse to run unless the pulled model has this digest")
    ap.add_argument("--reload", choices=["every", "once"], default="every",
                    help="reload the model before every window (each answer independent; default) "
                         "or once at the start (faster; repeats only as a whole run)")
    ap.add_argument("--think", action="store_true",
                    help="let the model reason before answering (slower; still greedy and repeatable)")
    ap.add_argument("--window", type=float, default=60.0, help="window length, seconds")
    ap.add_argument("--overlap", type=float, default=10.0, help="window overlap, seconds")
    ap.add_argument("--lag", type=float, default=0.0,
                    help="seconds to subtract from event times (commentator delay; tune on labelled matches)")
    ap.add_argument("--merge", type=float, default=10.0,
                    help="merge same-label events closer than this many seconds")
    ap.add_argument("--drop-replays", action="store_true", help="leave out replay_or_recap rows")
    args = ap.parse_args()
    raw_path = args.raw or args.out.with_suffix(".raw.jsonl")
    manifest_path = args.out.with_suffix(".manifest.json")

    segs = load_segments(args.transcript)
    if not segs:
        print("No segments found in the transcript.", file=sys.stderr)
        return 1
    wins = list(windows(segs, args.window, args.overlap))
    prompts = ["\n".join(f"[s{i} | {segs[i]['start']:.1f}] {segs[i]['text']}" for i in idx) for idx in wins]
    system_hash = sha256_bytes(SYSTEM.encode())

    if args.from_raw:
        records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
        if len(records) != len(prompts) or any(
                r["prompt_sha256"] != sha256_bytes(p.encode()) or r["system_sha256"] != system_hash
                for r, p in zip(records, prompts)):
            sys.exit(f"{raw_path} does not match this transcript, these window settings and this prompt")
        answers = [r["answer"] for r in records]
        digest = records[0]["model_digest"] if records else ""
        print(f"rebuilding from {len(answers)} saved answers in {raw_path}", file=sys.stderr)
    else:
        digest = model_digest(args.model)
        if args.expect_digest and digest != args.expect_digest:
            sys.exit(f"model digest {digest} != expected {args.expect_digest}")
        print(f"{len(segs)} segments, {len(wins)} windows, model {args.model} ({digest[:12]})", file=sys.stderr)
        answers = []
        with raw_path.open("w", encoding="utf-8", newline="\n") as f:
            for n, prompt in enumerate(prompts, 1):
                if n == 1 or args.reload == "every":
                    unload(args.model)
                answer = ask(args.model, prompt, args.think)
                answers.append(answer)
                f.write(json.dumps({"window": n - 1, "prompt_sha256": sha256_bytes(prompt.encode()),
                                    "system_sha256": system_hash, "model": args.model,
                                    "model_digest": digest, "think": args.think, "answer": answer},
                                   ensure_ascii=False, sort_keys=True) + "\n")
                f.flush()
                print(f"\rwindow {n}/{len(wins)}", end="", file=sys.stderr)
        print(file=sys.stderr)

    rows = []
    for idx, answer in zip(wins, answers):
        valid = {f"s{i}" for i in idx}
        for ev in parse_answer(answer):
            sid = str(ev.get("segment_id", "")).strip()
            label, status, basis = ev.get("label"), ev.get("status"), ev.get("basis")
            if sid not in valid or label not in LABELS or status not in STATUSES or basis not in BASES:
                continue  # made-up segment ids or values outside the lists
            if status == "replay_or_recap" and args.drop_replays:
                continue
            i = int(sid[1:])
            quote = " ".join(str(ev.get("quote") or "").split())
            rows.append({
                "time_s": f"{max(segs[i]['start'] - args.lag, 0.0):.2f}",
                "label": label,
                "status": status,
                "basis": basis,
                "p_event": f"{p_event(status, basis):.2f}",
                "team": " ".join(str(ev.get("team") or "").split()),
                "player": " ".join(str(ev.get("player") or "").split()),
                "segment": sid,
                # 1 when the quoted words really are in that line: a check against invention
                "quote_ok": int(bool(quote) and squash(quote) in squash(segs[i]["text"])),
                "quote": quote,
            })

    # Overlapping windows report the same incident twice. Per label, events within --merge seconds
    # are one incident (replays are kept apart from live play); keep the most settled report:
    # anything beats "uncertain", then the stronger basis, then the earlier line.
    def strength(e: dict) -> tuple:
        return (e["status"] != "uncertain", BASIS_RANK[e["basis"]], e["quote_ok"])

    rows.sort(key=lambda e: (e["label"], e["status"] == "replay_or_recap", float(e["time_s"]),
                             tuple(-x for x in strength(e)), e["segment"], e["quote"]))
    merged: list[dict] = []
    group_start = 0.0
    for ev in rows:
        prev = merged[-1] if merged else None
        same = (prev is not None and prev["label"] == ev["label"]
                and (prev["status"] == "replay_or_recap") == (ev["status"] == "replay_or_recap")
                and float(ev["time_s"]) - group_start <= args.merge)
        if same:
            if strength(ev) > strength(prev):
                merged[-1] = ev
            continue
        merged.append(ev)
        group_start = float(ev["time_s"])
    merged.sort(key=lambda e: (float(e["time_s"]), e["label"], e["status"], e["segment"]))

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, lineterminator="\n", fieldnames=[
            "time_s", "label", "status", "basis", "p_event", "team", "player", "segment", "quote_ok", "quote"])
        w.writeheader()
        w.writerows(merged)

    raw_hash, csv_hash = sha256_file(raw_path), sha256_file(args.out)
    if not args.from_raw:
        manifest = {
            "script_sha256": sha256_file(Path(__file__)),
            "system_prompt_sha256": system_hash,
            "transcript": args.transcript.name,
            "transcript_sha256": sha256_file(args.transcript),
            "model": args.model,
            "model_digest": digest,
            "ollama_version": ollama_version(),
            "machine": f"{platform.machine()} {platform.system()} {platform.mac_ver()[0]}".strip(),
            "options": OPTIONS,
            "think": args.think,
            "reload": args.reload,
            "settings": {k: getattr(args, k) for k in ("window", "overlap", "lag", "merge", "drop_replays")},
            "raw_sha256": raw_hash,
            "csv_sha256": csv_hash,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(merged)} events to {args.out}", file=sys.stderr)
    print(f"raw answers  {raw_hash}", file=sys.stderr)
    print(f"events csv   {csv_hash}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
