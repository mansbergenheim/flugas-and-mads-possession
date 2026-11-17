#!/usr/bin/env python3
"""Read the two starting line-ups off the broadcast's own line-up graphic. Run this yourself.

    python3 scripts/read_lineups.py

Nobody types a name in: the names come from the text the broadcast shows on screen, read by the
Mac's built-in text recognition (scripts/ocr_video_text.swift), and are sorted into teams by rules.
The printout is words only. The names are written to <data folder>/lineups/lineups.json, which is
about the real match and stays on this computer; open it yourself to check it.

How it works
  1. Where to look: from the start of the broadcast to a few minutes after kick-off. Kick-off on
     the broadcast's timeline comes from the alignment (scripts/align_broadcast.py) and the
     organisers' published first frame of the first half.
  2. One frame every two seconds is read for text, with the place of each piece of text.
  3. A frame counts as a line-up graphic when it shows at least nine shirt numbers each with a name
     beside or under it. Frames close in time with mostly the same names form one graphic; a number and name must be seen in
     most of its frames to count, which removes reading slips.
  4. Which team a graphic belongs to: the team name from the tracking metadata is looked for on the
     graphic. If that does not settle it, the first graphic is the home team and the next the away
     team (the printout says which rule was used).
  5. If a graphic shows more than eleven names, the eleven in the largest type are the starters and
     the rest are kept as "others" (normally the bench).
  6. Graphics without shirt numbers (the real broadcast's are like this) are found another way, and
     only if step 3 found nothing: with --transcript, a name on screen is "heard" when the
     commentators say it at least twice during the match. A stretch of frames with several heard
     names is a line-up graphic; its players are the names that stay on screen and share the type
     size of the heard names, the heard ones first. So the commentary confirms which pieces of
     on-screen text are players; it does not supply the names or their spelling.

The text reading is saved (<data folder>/lineups/ocr_text.json) and reused, so step 3 onwards gives
the same bytes on any machine; the reading itself needs a Mac.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from mtah.loaders import gated  # noqa: E402

MIN_PAIRS = 9          # shirt numbers with names in one frame before it counts as a line-up graphic
MAX_GAP_S = 6.0        # frames further apart than this belong to different graphics
SEEN_SHARE = 0.5       # a number and name must be in this share of a graphic's frames
NUM = re.compile(r"^\d{1,2}$")
NUM_NAME = re.compile(r"^(\d{1,2})[\s.:\-]+(\D{3,})$")
NAME_NUM = re.compile(r"^(\D{3,}?)[\s.:\-]+(\d{1,2})$")


def norm(text: str) -> str:
    """Upper case, no accents, letters and single spaces only."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^A-Za-z ]+", " ", plain).upper().split())


def is_name(text: str) -> bool:
    words = norm(text).split()
    return (not re.search(r"\d", text)) and 1 <= len(words) <= 4 and sum(map(len, words)) >= 3


def pairs_in_frame(items: list[dict]) -> list[dict]:
    """(shirt number, name) pairs in one frame: written together, side by side, or number above name."""
    found, numbers, names = [], [], []
    for it in items:
        text = " ".join(it["text"].split())
        m = NUM_NAME.match(text)
        n = NAME_NUM.match(text) if not m else None
        if m and is_name(m.group(2)):
            found.append({"number": int(m.group(1)), "name": m.group(2).strip(), "h": it["h"]})
        elif n and is_name(n.group(1)):
            found.append({"number": int(n.group(2)), "name": n.group(1).strip(), "h": it["h"]})
        elif NUM.match(text):
            numbers.append(it)
        elif is_name(text):
            names.append(it)
    options = []
    for a, num in enumerate(numbers):
        for b, name in enumerate(names):
            ncx, ncy = num["x"] + num["w"] / 2, num["y"] + num["h"] / 2
            mcx, mcy = name["x"] + name["w"] / 2, name["y"] + name["h"] / 2
            tall = max(num["h"], name["h"])
            gap_x = max(name["x"] - (num["x"] + num["w"]), num["x"] - (name["x"] + name["w"]), 0)
            gap_y = max(name["y"] - (num["y"] + num["h"]), num["y"] - (name["y"] + name["h"]), 0)
            if abs(ncy - mcy) <= 0.6 * tall and gap_x <= 1200:            # same row
                options.append((gap_x, a, b))
            elif abs(ncx - mcx) <= max(name["w"] / 2, 300) and gap_y <= 3 * name["h"]:   # stacked
                options.append((gap_y + 200, a, b))
    used_a, used_b = set(), set()
    for _, a, b in sorted(options):
        if a in used_a or b in used_b:
            continue
        used_a.add(a), used_b.add(b)
        found.append({"number": int(numbers[a]["text"].strip()), "name": " ".join(names[b]["text"].split()),
                      "h": names[b]["h"]})
    best = {}
    for p in sorted(found, key=lambda p: (p["number"], -p["h"], p["name"])):
        if 1 <= p["number"] <= 99:
            best.setdefault(p["number"], p)
    return list(best.values())


def graphics(frames: list[dict]) -> list[dict]:
    """Group line-up frames into graphics and keep the numbers and names seen in most of each one."""
    lineup_frames = []
    for frame in frames:
        pairs = pairs_in_frame(frame["items"])
        numbers = sorted(p["number"] for p in pairs)
        looks_like_a_table = len(numbers) >= 12 and numbers == list(range(1, len(numbers) + 1))
        if len(pairs) >= MIN_PAIRS and not looks_like_a_table:
            lineup_frames.append((frame["t_ms"] / 1000.0, pairs, frame["items"]))
    groups, out = [], []
    def same_graphic(a, b) -> bool:
        """Close in time and mostly the same numbers and names: one graphic, not two shown in a row."""
        x, y = ({(p["number"], norm(p["name"])) for p in e[1]} for e in (a, b))
        return b[0] - a[0] <= MAX_GAP_S and len(x & y) >= 0.5 * min(len(x), len(y))

    for entry in lineup_frames:
        if groups and same_graphic(groups[-1][-1], entry):
            groups[-1].append(entry)
        else:
            groups.append([entry])
    for group in groups:
        if len(group) < 2:
            continue
        names, heights = defaultdict(Counter), defaultdict(list)
        for _, pairs, _ in group:
            for p in pairs:
                names[p["number"]][norm(p["name"])] += 1
                heights[p["number"]].append(p["h"])
        players = []
        for number in sorted(names):
            name, seen = sorted(names[number].items(), key=lambda kv: (-kv[1], kv[0]))[0]
            if sum(names[number].values()) >= SEEN_SHARE * len(group) and name:
                players.append({"number": number, "name": name, "h": float(np.median(heights[number]))})
        if len(players) >= MIN_PAIRS:
            text = " ".join(norm(it["text"]) for _, _, items in group for it in items)
            out.append({"start_s": group[0][0], "frames": len(group), "players": players, "text": text})
    return out


HEARD_TIMES = 2        # a name is "heard" when the commentary says one of its words this often
HEARD_IN_FRAME = 4     # heard names in a frame before it may belong to a numberless line-up graphic
HEARD_PEAK = 6         # ...and at least one frame of the graphic must show this many
SIZE_BAND = (0.75, 1.3)  # type size of a player's name, relative to the median of the heard names


def spoken_words(transcript: Path) -> Counter:
    sys.path.insert(0, os.path.dirname(__file__))
    from extract_commentary_events import load_segments
    return Counter(w for seg in load_segments(transcript) for w in norm(seg["text"]).split() if len(w) >= 4)


def graphics_without_numbers(frames: list[dict], spoken: Counter, team_names: list[str]) -> list[dict]:
    """Line-up graphics that show names only: found through the names the commentators say."""
    team_words = {w for name in team_names for w in norm(name).split()}
    often = sorted(w for w, c in spoken.items() if c >= HEARD_TIMES)

    def heard(name: str) -> bool:
        return any(spoken[w] >= HEARD_TIMES for w in name.split() if len(w) >= 4)

    def nearly_heard(name: str) -> bool:
        """Speech recognition spells names its own way: same first letter and 80% of the letters alike."""
        return any(v[0] == w[0] and abs(len(v) - len(w)) <= 2 and SequenceMatcher(None, w, v).ratio() >= 0.8
                   for w in name.split() if len(w) >= 4 for v in often)

    per_frame = []
    for frame in frames:
        seen = {}
        for it in frame["items"]:
            text = " ".join(it["text"].split())
            name = norm(text)
            if is_name(text) and not set(name.split()) <= team_words:
                seen.setdefault(name, it["h"])
        per_frame.append(seen)
    count = [sum(map(heard, seen)) for seen in per_frame]
    rich = [k for k, c in enumerate(count) if c >= HEARD_IN_FRAME]
    groups = []
    for k in rich:
        if groups and (frames[k]["t_ms"] - frames[groups[-1][-1]]["t_ms"]) / 1000.0 <= MAX_GAP_S:
            groups[-1].append(k)
        else:
            groups.append([k])
    groups = [g for g in groups if len(g) >= 3 and max(count[k] for k in g) >= HEARD_PEAK]
    out = []
    for n_group, group in enumerate(groups):
        # the graphic draws itself in and fades out: take up to four frames either side that still show heard names
        first, last = group[0], group[-1]
        low = groups[n_group - 1][-1] + 1 if n_group else 0
        high = groups[n_group + 1][0] - 1 if n_group + 1 < len(groups) else len(frames) - 1
        while first - 1 >= low and group[0] - (first - 1) <= 4 and count[first - 1] >= 2:
            first -= 1
        while last + 1 <= high and (last + 1) - group[-1] <= 4 and count[last + 1] >= 2:
            last += 1
        span = list(range(first, last + 1))
        votes, heights = Counter(), defaultdict(list)
        for k in span:
            for name, h in per_frame[k].items():
                votes[name] += 1
                heights[name].append(h)
        stable = [n for n, v in votes.items() if v >= 2]

        def together(a: str, b: str) -> bool:
            return any(a in per_frame[k] and b in per_frame[k] for k in span)

        # one name read two ways (a slip, or a name still being drawn): keep the reading seen most often.
        # Two alike names on screen at the same moment are two players (brothers, say) and both stay.
        kept = []
        for n in sorted(stable, key=lambda n: (-votes[n], -len(n), n)):
            if not any(SequenceMatcher(None, n, m).ratio() >= 0.85 and not together(n, m) for m in kept):
                kept.append(n)
        heard_sizes = sorted(float(np.median(heights[n])) for n in kept if heard(n))
        if not heard_sizes:
            continue
        size = heard_sizes[len(heard_sizes) // 2]
        in_band = [n for n in kept if SIZE_BAND[0] * size <= float(np.median(heights[n])) <= SIZE_BAND[1] * size]
        level = {n: 2 if heard(n) else 1 if nearly_heard(n) else 0 for n in in_band}
        ranked = sorted(in_band, key=lambda n: (-level[n], -votes[n], n))
        players = [{"number": None, "name": n, "h": float(np.median(heights[n])), "heard": level[n] > 0}
                   for n in ranked]
        if len(players) >= MIN_PAIRS:
            text = [" ".join(norm(it["text"]) for it in frames[k]["items"]) for k in span]
            out.append({"start_s": frames[span[0]]["t_ms"] / 1000.0, "frames": len(span), "players": players,
                        "text": " ".join(text), "frame_texts": text, "numberless": True})
    return out


def names_team(text: str, team_name: str) -> bool:
    """Is this team's name written somewhere in the graphic's text?"""
    words = [w for w in norm(team_name).split() if len(w) >= 4]
    seen = set(text.split())
    return any(w in seen or any(SequenceMatcher(None, w, s).ratio() >= 0.85 for s in seen if abs(len(s) - len(w)) <= 2)
               for w in words)


def split_starters(players: list[dict]) -> tuple[list[dict], list[dict]]:
    if len(players) <= 11:
        return players, []
    if all(p["number"] is None for p in players):     # already ranked: heard in the commentary first
        return players[:11], players[11:]
    ranked = sorted(players, key=lambda p: (-p["h"], p["number"]))
    return order(ranked[:11]), order(ranked[11:])


def order(players: list[dict]) -> list[dict]:
    return sorted(players, key=lambda p: (p["number"] is None, p["number"] or 0, p["name"]))


def deduce(frames: list[dict], team_names: list[str], spoken: Counter | None = None) -> tuple[dict, list[str]]:
    found, words = graphics(frames), []
    if not found and spoken is not None:
        found = graphics_without_numbers(frames, spoken, team_names)
    sides = ("home", "away")
    owner = []
    for g in found:
        hits = [names_team(g["text"], name) for name in team_names]
        if sum(hits) == 2 and "frame_texts" in g:
            # both names appear somewhere: the graphic's own team is the one named in clearly more of its frames
            counts = [sum(names_team(t, name) for t in g["frame_texts"]) for name in team_names]
            hits = [counts[0] >= 2 * max(counts[1], 1), counts[1] >= 2 * max(counts[0], 1)]
        owner.append(sides[hits.index(True)] if sum(hits) == 1 else None)
    by_name = [o for o in owner if o]
    for k, o in enumerate(owner):            # graphics the team name did not settle: home first, then away
        if o is None:
            taken = {x for x in owner if x}
            free = [s for s in sides if s not in taken]
            owner[k] = free[0] if free else None
    result = {}
    for side, team in zip(sides, team_names):
        mine = [g for g, o in zip(found, owner) if o == side]
        if not mine:
            result[side] = {"team": team, "starting": [], "others": [], "how": "no line-up graphic found"}
            words.append(f"{side:<5} no line-up graphic was found")
            continue
        # the graphic nearest to eleven names gives the starters; any other graphic of this team adds "others"
        main = sorted(mine, key=lambda g: (abs(len(split_starters(g["players"])[0]) - 11),
                                           len(g["players"]) != 11, -g["frames"], g["start_s"]))[0]
        starters, others = split_starters(main["players"])
        known = {(p["number"], p["name"] if p["number"] is None else "") for p in starters + others}
        for g in mine:
            for p in g["players"]:
                key = (p["number"], p["name"] if p["number"] is None else "")
                if key not in known:
                    others.append(p)
                    known.add(key)
        how = "team name on the graphic" if side in by_name else "order of the graphics (home first)"
        clean = lambda ps: [{"number": p["number"], "name": p["name"]} for p in order(ps)]  # noqa: E731
        result[side] = {"team": team, "starting": clean(starters), "others": clean(others), "how": how}
        count = "eleven starters read" if len(starters) == 11 else \
            f"only {len(starters)} starters read (eleven expected): check the file"
        if main.get("numberless"):
            confirmed = sum(p["heard"] for p in starters)
            extra = (", from a graphic without shirt numbers, "
                     + ("all" if confirmed == len(starters) else "most" if confirmed >= 8 else "only some")
                     + " of them also heard in the commentary")
        else:
            extra = ", more than eleven names on the graphic so the eleven in the largest type were taken" \
                if len(main["players"]) > 11 else ""
        bench = ", other names kept as well" if others else ""
        words.append(f"{side:<5} {count}{extra}{bench}; team decided by the {how}")
    return result, words


def kickoff_on_broadcast(root: str) -> float:
    path = Path(gated.data_dir(root)) / "alignment" / "broadcast_alignment.npz"
    if not path.exists():
        sys.exit(f"{path} not found: run scripts/align_broadcast.py first, or give --end")
    first = gated.HALVES[0][0]
    for start, after_last, shift in np.load(path)["pieces"].reshape(-1, 3):
        if start + shift <= first < after_last + shift:
            return float(first - shift) / gated.FPS
    sys.exit("the alignment does not cover the first kick-off: give --end in seconds")


def team_names(root: str) -> list[str]:
    meta = ET.parse(gated.asset("atd_meta", root, "tactical")).getroot()
    names = [(t.findtext("Name") or "").strip() for t in meta.iter("Team")]
    if len(names) != 2:
        sys.exit("expected two teams in the tracking metadata")
    return names   # first team in the file is home, as in the organisers' baseline


def read_text(video: Path, out: Path, start: float, end: float, step: float) -> None:
    source = Path(__file__).with_name("ocr_video_text.swift")
    tool = out.parent / "ocr_video_text"
    if not tool.exists() or tool.stat().st_mtime < source.stat().st_mtime:
        print("reading tool   building it (once, about half a minute)", flush=True)
        built = subprocess.run(["swiftc", "-O", str(source), "-o", str(tool)], capture_output=True, text=True)
        if built.returncode:
            sys.exit("could not build the reading tool (needs the Xcode command line tools: xcode-select --install)\n"
                     + built.stderr[-2000:])
    if subprocess.run([str(tool), str(video), str(out), str(start), str(end), str(step)]).returncode:
        sys.exit("reading the video failed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="the data folder; default MTAH_DATA_DIR, then ~/mtah-data")
    ap.add_argument("--video", type=Path, default=None, help="default: <data folder>/broadcast/BAR-ATH_broadcast.mp4")
    ap.add_argument("--start", type=float, default=0.0, help="first second of the broadcast to look at")
    ap.add_argument("--end", type=float, default=None, help="last second; default: five minutes after kick-off")
    ap.add_argument("--step", type=float, default=2.0, help="seconds between the frames that are read")
    ap.add_argument("--home", default=None, help="home team's name; default: from the tracking metadata")
    ap.add_argument("--away", default=None, help="away team's name; default: from the tracking metadata")
    ap.add_argument("--transcript", type=Path, default=None,
                    help="the commentary transcript (.srt, .vtt or whisper .json): needed for graphics without shirt numbers")
    ap.add_argument("--out", type=Path, default=None, help="folder for the results; default <data folder>/lineups")
    ap.add_argument("--read-again", action="store_true", help="read the video again even if a reading is saved")
    args = ap.parse_args()
    root = args.data or os.environ.get("MTAH_DATA_DIR") or os.path.expanduser("~/mtah-data")
    folder = args.out or Path(gated.data_dir(root)) / "lineups"
    folder.mkdir(parents=True, exist_ok=True)
    video = args.video or Path(gated.asset("video", root, "broadcast"))
    names = [args.home, args.away] if args.home and args.away else team_names(root)
    end = args.end if args.end is not None else kickoff_on_broadcast(root) + 300.0

    reading = folder / "ocr_text.json"
    wanted = {"start_ms": round(args.start * 1000), "end_ms": round(end * 1000), "step_ms": round(args.step * 1000)}
    saved = json.loads(reading.read_text(encoding="utf-8")) if reading.exists() else None
    if args.read_again or saved is None or any(saved.get(k) != v for k, v in wanted.items()):
        if not video.exists():
            sys.exit(f"{video} not found")
        print("1 READING      looking at the broadcast up to a few minutes after kick-off (takes a few minutes)",
              flush=True)
        read_text(video, reading, args.start, end, args.step)
        saved = json.loads(reading.read_text(encoding="utf-8"))
    else:
        print("1 READING      using the saved reading of the broadcast")
    frames = saved["frames"]
    with_text = sum(bool(f["items"]) for f in frames)
    print(f"               text was found in {'most' if with_text >= 0.5 * len(frames) else 'some' if with_text else 'none'}"
          " of the frames")

    spoken = spoken_words(args.transcript) if args.transcript else None
    result, words = deduce(frames, names, spoken)
    if spoken is None and not any(result[s]["starting"] for s in result):
        words.append("      if the graphic shows names without shirt numbers, run again with --transcript PATH")
    for k, line in enumerate(words):
        print(f"{'2 LINE-UPS' if not k else '':<15}{line}")
    target = folder / "lineups.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    good = all(len(result[s]["starting"]) == 11 for s in ("home", "away"))
    print(f"3 SAVED        {target}  (about the real match: keep it on this computer)")
    print(f"               sha256 {hashlib.sha256(target.read_bytes()).hexdigest()}")
    print("VERDICT        " + ("two elevens read: open the file and check the names look like players"
                               if good else "not two full elevens: open the file, and see the notes in this script"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
