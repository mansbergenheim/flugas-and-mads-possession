"""Find the players of the two line-ups in commentary text, by rules only.

The line-ups come from the broadcast's own graphics (scripts/read_lineups_vision.py writes
<data folder>/lineups/lineups.json). Speech recognition spells names its own way, so a word of the
commentary is a player when it is close to a word of one player's name. The rules are strict on
purpose, because a name pinned on the wrong team is worse than no name:

- only words of at least MIN_LETTERS letters that the transcript writes with a capital (speech
  recognition capitalises names; ordinary words in mid-sentence are lower case);
- the same first letter, at most two letters longer or shorter, and SAME of the letters alike;
- a name word that is even loosely close (RIVAL) to a word of a player in the OTHER team is never
  used: such a word cannot tell the teams apart (a shared surname, or two names a misspelling could
  confuse);
- a commentary word that matches name words of both teams is skipped.

Team names and nicknames are not used: only names read from the graphics. The same text and
line-ups always give the same answer.
"""

import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Sequence, Tuple

MIN_LETTERS = 4
SAME = 0.85         # letters alike for a commentary word to be a name word
RIVAL = 0.75        # name words of the two teams this alike are too easily confused: both dropped
SIDES = ("home", "away")


def norm(text: str) -> str:
    """Upper case, no accents, letters and single spaces only."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^A-Za-z ]+", " ", plain).upper().split())


def alike(a: str, b: str, level: float = SAME) -> bool:
    return a == b or (a[:1] == b[:1] and abs(len(a) - len(b)) <= 2 and SequenceMatcher(None, a, b).ratio() >= level)


@dataclass(frozen=True)
class Player:
    name: str
    side: str


@dataclass(frozen=True)
class Mention:
    start: int          # character offsets in the text
    stop: int
    said: str           # the words as written in the transcript
    player: str         # the line-up name; "" when the word fits more than one player of that team
    side: str           # home or away


class Roster:
    def __init__(self, players: Sequence[Player]):
        self.players = list(players)
        words: Dict[str, List[Player]] = {}
        for p in self.players:
            for w in norm(p.name).split():
                if len(w) >= MIN_LETTERS:
                    words.setdefault(w, [])
                    if p not in words[w]:
                        words[w].append(p)
        # A word close to a word of the other team cannot tell the teams apart: drop it.
        self.words = {}
        for w, owners in sorted(words.items()):
            sides = {p.side for p in owners}
            rivals = [v for v, o in words.items() if v != w and alike(w, v, RIVAL) and {p.side for p in o} - sides]
            if len(sides) == 1 and not rivals:
                self.words[w] = owners
        self.dropped = sorted(set(words) - set(self.words))

    @classmethod
    def from_lineups(cls, path: str) -> "Roster":
        data = json.load(open(path, encoding="utf-8"))
        players = []
        for side in SIDES:
            for group in ("starting", "others"):
                for p in data.get(side, {}).get(group, []):
                    if norm(p.get("name", "")):
                        players.append(Player(norm(p["name"]), side))
        return cls(players)

    def find(self, text: str) -> List[Mention]:
        """Players named in this text, in order. Two words of one name in a row are one mention."""
        found: List[Mention] = []
        for m in re.finditer(r"[^\W\d_][\w'’.\-]*", text):
            raw = m.group(0)
            if not raw[:1].isupper():
                continue
            word = norm(raw).replace(" ", "")
            if len(word) < MIN_LETTERS:
                continue
            owners = [p for w, ps in self.words.items() if alike(word, w) for p in ps]
            if not owners or len({p.side for p in owners}) != 1:
                continue
            names = sorted({p.name for p in owners})
            player = names[0] if len(names) == 1 else ""
            last = found[-1] if found else None
            if (last and last.side == owners[0].side and last.player == player
                    and not text[last.stop:m.start()].strip(" -")):
                found[-1] = Mention(last.start, m.end(), text[last.start:m.end()], player, last.side)
                continue
            found.append(Mention(m.start(), m.end(), raw, player, owners[0].side))
        return found


def counts(roster: Roster) -> Tuple[int, int, int]:
    """(players, usable name words, name words dropped because they could fit both teams)."""
    return len(roster.players), len(roster.words), len(roster.dropped)
