#!/usr/bin/env python3
"""Does any organiser file name the players? Run this yourself. Prints words and field names only.

    python3 scripts/check_player_fields.py            # data folder: MTAH_DATA_DIR, then ~/mtah-data

DATA.md says the tracking has no player identities: 46 slots called "Track 1".."Track 46" whose
ShirtNumber is the slot index. This checks that claim on the real files, for the tactical and the
broadcast tracking metadata and the Smart Tagging file. For each file it prints:

  - which fields exist (element and attribute names: the file's layout, not its contents);
  - whether every player entry is a "Track N" slot, and whether its shirt number is just N;
  - how many pieces of text are something other than a slot name, a team name or a number, and in
    which fields. Those are the only places a real name could hide. The text itself is not printed:
    open the file yourself and look at those fields if the count is not zero.
"""

import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from mtah.loaders import gated  # noqa: E402

SLOT = re.compile(r"^track\s*(\d+)$", re.IGNORECASE)
TEAM_WORDS = ("barcelona", "athletic", "atletic", "home", "away")
NUMBER = re.compile(r"^[-+]?[\d.,:;eE\s/-]*$")
# Smart Tagging values the organisers list in DATA.md: words, but not names.
KNOWN_WORDS = {"n/a", "central", "left", "right", "horizontal", "vertical", "diagonal", "1st half",
               "2nd half", "throw-in", "corner kick", "goal kick", "free kick", "kick off (after goal)",
               "kick off (start)", "7+", "true", "false", "team", "half", "type", "side",
               "direction of ball entry", "max players in the box"}


def plain(value):
    return value is None or not value.strip() or bool(NUMBER.match(value.strip()))


def ordinary(value):
    v = value.strip().lower()
    upper_code = value.strip() == value.strip().upper()          # Smart Tagging codes: SET PIECES, ...
    return (plain(value) or bool(SLOT.match(v)) or v in KNOWN_WORDS or upper_code
            or any(word in v for word in TEAM_WORDS))


def inspect(label, path):
    print(f"\n{label}")
    if not os.path.exists(path):
        print("   not found")
        return
    root = ET.parse(path).getroot()
    layout = Counter()
    other = Counter()
    players = slots = shirt_is_slot = 0
    for element in root.iter():
        tag = element.tag.split("}")[-1]
        layout[tag] += 1
        for name, value in element.attrib.items():
            if not ordinary(value):
                other[f"{tag} @{name}"] += 1
        text = (element.text or "").strip()
        if text and not ordinary(text):
            other[f"{tag} (text)"] += 1
        if tag == "Player":
            players += 1
            name = element.get("name") or element.get("Name") or (element.findtext("Name") or "")
            match = SLOT.match(name.strip())
            if match:
                slots += 1
                shirt = element.get("ShirtNumber") or element.get("shirtNumber") or element.findtext("ShirtNumber")
                if shirt is not None and shirt.strip() == match.group(1):
                    shirt_is_slot += 1
    attributes = sorted({f"{e.tag.split('}')[-1]} @{a}" for e in root.iter() for a in e.attrib})
    print("   elements:  " + ", ".join(f"{t} ({n})" for t, n in sorted(layout.items())))
    print("   attributes: " + (", ".join(attributes) if attributes else "none"))
    if players:
        print(f"   player entries: {players}; named 'Track N': "
              f"{'all' if slots == players else f'{slots} of {players}'}; shirt number is just N: "
              f"{'all' if shirt_is_slot == players else f'{shirt_is_slot} of {players}'}")
    else:
        print("   no player entries")
    if not other:
        print("   no other text anywhere: this file cannot name a player")
        return
    where(root)


NAME_LIKE = re.compile(r"^[A-ZÀ-Þ][A-Za-zÀ-ÿ'’.\-]*( [A-Za-zÀ-ÿ'’.\-]+)*$")


def kind(value):
    """What a piece of text looks like, without saying what it is."""
    v = (value or "").strip()
    if not v:
        return "empty"
    if plain(v):
        return "number"
    if v.lower() in ("true", "false"):
        return "true/false"
    if SLOT.match(v):
        return "Track N"
    if any(word in v.lower() for word in TEAM_WORDS):
        return "team word"
    if NAME_LIKE.match(v) and v != v.upper():
        return "looks like a name (capitalised words, letters only)"
    if re.search(r"[_\d]", v) or v == v.lower() or v == v.upper():
        return "code or identifier (digits, underscores, or one case)"
    return "other text"


def where(root):
    """Where the leftover text sits, by the part of the file it is in. Prints the file's own field
    and parameter names (its layout) and what kind of value each holds, never a value."""
    parent = {child: element for element in root.iter() for child in element}

    def tag(e):
        return e.tag.split("}")[-1]

    def section(e):
        """The nearest enclosing part that says what this is about (players, teams, session, ...)."""
        while e in parent:
            e = parent[e]
            if tag(e) in ("ProviderPlayerParameters", "ProviderTeamsParameters", "ProviderGlobalParameters",
                          "ProviderSessionParameters", "Player", "Team", "Competition", "Session",
                          "Sensor", "Channel", "PlayerChannel", "Device", "Metadata"):
                return tag(e)
        return "top"

    names = Counter()
    for e in root.iter():
        if tag(e) == "Name" and tag(parent.get(e, e)) != "ProviderParameter":
            names[(tag(parent.get(e, e)), kind(e.text))] += 1
    print("   where the leftover text sits (values never printed):")
    print("     <Name> fields, by what they name:")
    for (owner, k), n in sorted(names.items()):
        print(f"        {owner}: {n} x {k}")

    params = Counter()
    for e in root.iter():
        if tag(e) == "ProviderParameter":
            key = (e.findtext("Name") or "").strip()[:40] or "(no name)"
            params[(section(e), key, kind(e.findtext("Value")))] += 1
    if params:
        print("     provider parameters, by section and parameter name, with the kind of value:")
        for (sec, key, k), n in sorted(params.items()):
            print(f"        {sec} / {key}: {n} x {k}")

    flagged = Counter()
    for e in root.iter():
        for a, value in e.attrib.items():
            if kind(value).startswith("looks like a name"):
                flagged[f"{tag(e)} @{a}"] += 1
        if tag(e) not in ("Name", "Value") and kind(e.text).startswith("looks like a name"):
            flagged[f"{tag(e)} (text)"] += 1
    print("     other fields holding name-like text: "
          + (", ".join(f"{f} ({n})" for f, n in flagged.most_common()) if flagged else "none"))
    print("   -> a player could only be named where a row above says 'looks like a name' inside a")
    print("      player section; a team, competition, session or sensor name is not a player")

if __name__ == "__main__":
    root = os.environ.get("MTAH_DATA_DIR") or os.path.expanduser("~/mtah-data")
    for label, name, source in (("TACTICAL tracking metadata", "atd_meta", "tactical"),
                                ("BROADCAST tracking metadata", "atd_meta", "broadcast"),
                                ("SMART TAGGING", "smart_tagging", "tactical")):
        inspect(label, gated.asset(name, root, source))
