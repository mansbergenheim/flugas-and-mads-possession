#!/usr/bin/env python3
"""Say which team each commentary event belongs to, from the line-ups. Run this yourself.

    python3 scripts/commentary_add_teams.py EVENTS.csv

EVENTS.csv is any events file with the columns `player` and `team` as the commentators said them
(from extract_commentary_events.py, with or without the match clock). The line-ups come from
scripts/read_lineups.py (<data folder>/lineups/lineups.json). Three columns are added after `team`,
and the result is written to EVENTS_teams.csv:

    side          home or away (empty when it cannot be told)
    side_from     player (the named player is in that team's line-up) or team (the team was named)
    lineup_name   the player's name as the line-up graphic spells it

Rules, no judgement: speech recognition spells names its own way, so a spoken name matches a
line-up name when their letters are at least 80% alike (whole name or any word of it; a different
first letter costs five points), and only if
no name in the other team is nearly as close. A named team matches a side by its name from the
tracking metadata, by words such as "home" and "visitors", or by an alias you pass in. The same
inputs always give the same bytes. The printout is words only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from mtah.loaders import gated  # noqa: E402
from read_lineups import norm  # noqa: E402

ALIKE = 0.80       # letters alike before a spoken name counts as a line-up name
FIRST_LETTER = 0.05   # taken off when the two names start with different letters
CLEAR = 0.10       # ...and this much closer than the best name in the other team
SIDE_WORDS = {"home": ("HOME", "HOSTS", "HOST"), "away": ("AWAY", "VISITORS", "VISITING", "VISITOR")}


def words(text: str) -> list[str]:
    return [w for w in norm(text).split() if len(w) >= 3]


def alike(spoken: str, written: str) -> float:
    a, b = words(spoken), words(written)
    if not a or not b:
        return 0.0
    def ratio(x: str, y: str) -> float:
        # speech recognition rarely gets the first sound wrong, so a different first letter counts against
        return SequenceMatcher(None, x, y).ratio() - (0.0 if x[0] == y[0] else FIRST_LETTER)

    return max(max(ratio(x, y) for x in a for y in b), ratio(" ".join(a), " ".join(b)))


def side_of_player(spoken: str, lineups: dict) -> tuple[str, str]:
    best = {}
    for side in ("home", "away"):
        scored = sorted(((-alike(spoken, p["name"]), p["number"] or 0, p["name"])
                         for p in lineups[side]["starting"] + lineups[side]["others"]))
        best[side] = (-scored[0][0], scored[0][2]) if scored else (0.0, "")
    for side, other in (("home", "away"), ("away", "home")):
        if best[side][0] >= ALIKE and best[side][0] - best[other][0] >= CLEAR:
            return side, best[side][1]
    return "", ""


def side_of_team(spoken: str, names: dict) -> str:
    said = words(spoken)
    hits = set()
    for side in ("home", "away"):
        if any(w in SIDE_WORDS[side] for w in said):
            hits.add(side)
        for name in names[side]:
            for target in words(name):
                if len(target) < 4:
                    continue
                if any(SequenceMatcher(None, w, target).ratio() >= ALIKE or (len(w) >= 4 and w[:4] == target[:4])
                       for w in said):
                    hits.add(side)
    return hits.pop() if len(hits) == 1 else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("events", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None, help="default: EVENTS_teams.csv beside the input")
    ap.add_argument("--data", default=None, help="the data folder; default MTAH_DATA_DIR, then ~/mtah-data")
    ap.add_argument("--lineups", type=Path, default=None, help="default: <data folder>/lineups/lineups.json")
    ap.add_argument("--home-alias", action="append", default=[], help="another word commentators use for the home team")
    ap.add_argument("--away-alias", action="append", default=[], help="another word commentators use for the away team")
    args = ap.parse_args()
    out = args.out or args.events.with_name(args.events.stem + "_teams.csv")
    root = args.data or os.environ.get("MTAH_DATA_DIR") or os.path.expanduser("~/mtah-data")
    path = args.lineups or Path(gated.data_dir(root)) / "lineups" / "lineups.json"
    if not path.exists():
        sys.exit(f"{path} not found: run scripts/read_lineups.py first")
    lineups = json.loads(path.read_text(encoding="utf-8"))
    names = {"home": [lineups["home"]["team"]] + args.home_alias, "away": [lineups["away"]["team"]] + args.away_alias}

    with args.events.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, rows = list(reader.fieldnames or []), list(reader)
    if "player" not in fields or "team" not in fields:
        sys.exit(f"{args.events} needs the columns player and team")
    added = ["side", "side_from", "lineup_name"]
    if any(name in fields for name in added):
        sys.exit(f"{args.events} already has team columns; give the file without them")

    by_player = by_team = named = 0
    for row in rows:
        row.update(dict.fromkeys(added, ""))
        named += bool(row["player"].strip() or row["team"].strip())
        side, name = side_of_player(row["player"], lineups) if row["player"].strip() else ("", "")
        if side:
            row.update(side=side, side_from="player", lineup_name=name)
            by_player += 1
            continue
        side = side_of_team(row["team"], names) if row["team"].strip() else ""
        if side:
            row.update(side=side, side_from="team")
            by_team += 1

    k = fields.index("team") + 1
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields[:k] + added + fields[k:], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    def share(n: int, of: int) -> str:
        part = n / max(of, 1)
        return "all" if of and n == of else "most" if part >= 0.75 else "more than half" if part >= 0.5 \
            else "less than half" if n else "none"
    print(f"named          {share(named, len(rows))} of the events name a player or a team")
    print(f"given a side   {share(by_player + by_team, named)} of those got a side "
          f"({share(by_player, by_player + by_team)} of them from the player's name)")
    print(f"wrote          {out}")
    print(f"sha256         {hashlib.sha256(out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
