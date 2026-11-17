"""Registry and cache for the open matches. Parsing is slow, so each match is parsed once."""

import glob
import hashlib
import json
import os
from typing import List

from .loaders import dfl, metrica, skillcorner
from .match import Match, load_cache, save_cache

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OPEN_DIR = os.environ.get("MTAH_OPEN_DIR", os.path.join(PROJECT, "data_open"))
CACHE = os.path.join(OPEN_DIR, "cache")
# SHA-256 of every DFL and Metrica file the submitted model trains on, as downloaded on 7 Nov 2026
# and checked against the pinned versions in scripts/download_open_data.sh on 8 Nov.
OPEN_FILES = os.path.join(os.path.dirname(__file__), "open_data_sha256.json")

DFL_IDS = ("J03WPY", "J03WOH", "J03WOY", "J03WQQ", "J03WR9", "J03WMX", "J03WN1")
METRICA_IDS = ("METRICA1", "METRICA2", "METRICA3")
# SkillCorner's 20 open matches, split once on 7 Nov 2026 before any was scored. 1886347 had been
# looked at briefly, so it was put in the working half; the other 19 were shuffled with
# numpy.random.default_rng(2027) and the first 9 joined it. The reserve is kept for one final check.
SKILLCORNER_DEV = ("SC1886347", "SC1899585", "SC1925299", "SC1953632", "SC1986691", "SC1996435",
                   "SC2006363", "SC2010085", "SC2011166", "SC2016236")
SKILLCORNER_RESERVE = ("SC1874553", "SC1927964", "SC1959846", "SC1996436", "SC2006229", "SC2007448",
                       "SC2007721", "SC2013725", "SC2015213", "SC2017461")


def _dfl_file(kind: str, match_id: str) -> str:
    hits = glob.glob(os.path.join(OPEN_DIR, "dfl", f"DFL_{kind}_*{match_id}.xml"))
    if not hits:
        raise FileNotFoundError(f"{kind} for {match_id}: run scripts/download_open_data.sh")
    return hits[0]


def _parse(match_id: str) -> Match:
    m = os.path.join(OPEN_DIR, "metrica")
    if match_id in DFL_IDS:
        return dfl.load_dfl(_dfl_file("02_01_matchinformation", match_id),
                            _dfl_file("04_03_positions_raw_observed", match_id), match_id)
    if match_id in ("METRICA1", "METRICA2"):
        g = f"Sample_Game_{match_id[-1]}"
        return metrica.load_csv_game(os.path.join(m, f"{g}_RawTrackingData_Home_Team.csv"),
                                     os.path.join(m, f"{g}_RawTrackingData_Away_Team.csv"),
                                     os.path.join(m, f"{g}_RawEventsData.csv"), match_id)
    if match_id == "METRICA3":
        return metrica.load_epts_game(os.path.join(m, "Sample_Game_3_metadata.xml"),
                                      os.path.join(m, "Sample_Game_3_tracking.txt"),
                                      os.path.join(m, "Sample_Game_3_events.json"), match_id)
    if match_id in SKILLCORNER_DEV or match_id in SKILLCORNER_RESERVE:
        d = os.path.join(OPEN_DIR, "skillcorner", match_id[2:])
        f = os.path.join(d, match_id[2:])
        if not os.path.exists(f + "_tracking_extrapolated.jsonl"):
            raise FileNotFoundError(f"{match_id}: run scripts/download_skillcorner.sh")
        return skillcorner.load_skillcorner(f + "_match.json", f + "_tracking_extrapolated.jsonl",
                                            f + "_phases_of_play.csv", match_id)
    raise KeyError(match_id)


def load_open(match_id: str, refresh: bool = False) -> Match:
    # SkillCorner caches sit in their own folder, so run manifests keep listing only the data the
    # submitted model trains on.
    folder = os.path.join(CACHE, "skillcorner") if match_id.startswith("SC") else CACHE
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{match_id}.npz")
    if os.path.exists(path) and not refresh:
        return load_cache(path)
    match = _parse(match_id)
    match.validate()
    save_cache(match, path)
    return match


def available() -> List[str]:
    """DFL and Metrica matches whose files are on disk. SkillCorner is kept apart on purpose."""
    out = []
    for match_id in DFL_IDS:
        if glob.glob(os.path.join(OPEN_DIR, "dfl", f"DFL_04_03_positions_raw_observed_*{match_id}.xml")):
            out.append(match_id)
    if glob.glob(os.path.join(OPEN_DIR, "metrica", "Sample_Game_1_*")):
        out.extend(METRICA_IDS)
    return out


def check_open_files() -> List[str]:
    """Problems with the downloaded DFL and Metrica files, or an empty list if every one matches.

    A missing or changed file would quietly train a different model and write a different CSV, so
    make_submission.py stops on any problem instead.
    """
    with open(OPEN_FILES) as handle:
        expected = json.load(handle)
    problems = []
    for name, digest in expected.items():
        path = os.path.join(OPEN_DIR, name)
        if not os.path.exists(path):
            problems.append(f"missing {name}")
            continue
        h = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != digest:
            problems.append(f"changed {name}")
    return problems
