"""Write and check the competition CSV.

check_submission covers the array: row count, label codes, and X on graded frames, which is valid but
costs points. check_file then reads the written CSV back. Neither replaces the organisers' validator:
run that on the exact file before any upload.

Everything here reports counts only, so its output is safe to read aloud.
"""

import csv
from typing import Dict, List

import numpy as np

from .match import D, LABELS, X

N_FRAMES = 199319
# Published in DATA.md. These are competition boundaries, not observations from watching the game.
GRADED_BLOCKS = ((11185, 90652), (110808, 190198))


def invalid_codes(codes: np.ndarray) -> int:
    return int((~np.isin(codes, np.arange(len(LABELS)))).sum())


def write_submission(path: str, codes: np.ndarray) -> None:
    if len(codes) != N_FRAMES:
        raise ValueError(f"{len(codes)} frames, expected {N_FRAMES}")
    if invalid_codes(codes):
        raise ValueError(f"{invalid_codes(codes)} invalid label codes")
    letters = np.array(list(LABELS))[codes]
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{i},{v}\n" for i, v in enumerate(letters))


def read_submission(path: str) -> np.ndarray:
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    header = [c.strip().lower() for c in rows[0]]
    if header[:2] != ["frame", "label"]:
        raise ValueError(f"header is {rows[0]}, expected frame,label")
    lookup = {c: i for i, c in enumerate(LABELS)}
    frames = np.array([int(r[0]) for r in rows[1:]], dtype=np.int64)
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError("frame column is not 0..n-1 in order")
    labels = [r[1].strip().upper() for r in rows[1:]]
    bad = sum(v not in lookup for v in labels)
    if bad:
        raise ValueError(f"{bad} rows have a label outside {LABELS}")
    return np.array([lookup[v] for v in labels], dtype=np.int8)


def force_no_abstain(codes: np.ndarray) -> np.ndarray:
    """Replace X on graded frames with the label the scorer's fill would give.

    spellF1 is unchanged by construction. macroF1 stops charging a guaranteed miss on those frames.
    A half that is entirely X is left alone, because there is nothing to carry.
    """
    out = codes.copy()
    for start, stop in GRADED_BLOCKS:
        window = out[start:stop]
        known = np.flatnonzero(window != X)
        if not len(known):
            continue
        carry = np.maximum.accumulate(np.where(window != X, np.arange(len(window)), -1))
        carry[carry < 0] = known[0]
        out[start:stop] = window[carry]
    return out


def check_submission(codes: np.ndarray) -> List[str]:
    """Problems as one-line strings. An empty list means nothing was found."""
    problems = []
    if len(codes) != N_FRAMES:
        problems.append(f"row count {len(codes)} != {N_FRAMES}")
        return problems
    if invalid_codes(codes):
        problems.append(f"{invalid_codes(codes)} invalid label codes")
        return problems
    graded = np.concatenate([codes[a:b] for a, b in GRADED_BLOCKS])
    n_x = int((graded == X).sum())
    if n_x:
        problems.append(f"{n_x} X labels on graded frames: use force_no_abstain")
    return problems


def check_file(path: str, codes: np.ndarray) -> List[str]:
    """Read the written CSV back. An empty list means it holds exactly `codes`."""
    try:
        written = read_submission(path)
    except (OSError, ValueError, IndexError) as error:
        return [f"written file can't be read back: {type(error).__name__}"]
    if not np.array_equal(written, codes):
        return ["written file differs from the predictions"]
    return []


def summary(codes: np.ndarray) -> Dict[str, float]:
    """Counts only. No frame-level values."""
    graded = np.concatenate([codes[a:b] for a, b in GRADED_BLOCKS])
    n_spells = 0
    for a, b in GRADED_BLOCKS:
        w = codes[a:b]
        starts = np.r_[True, w[1:] != w[:-1]]
        n_spells += int((starts & (w < D)).sum())
    return {"graded_frames": int(len(graded)), "possessions": n_spells,
            "share_H": float((graded == 0).mean()), "share_A": float((graded == 1).mean()),
            "share_D": float((graded == 2).mean()), "share_X": float((graded == 3).mean())}
