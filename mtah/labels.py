"""Bring labels from other providers closer to the competition's definition of possession.

DFL's BallPossession flips on touches the competition would not count: its possessions are about
half as long and more than twice as often under a second as Metrica's or the public ground truth.
The competition says a touch without control changes nothing, so a short spell squeezed between
two spells of the other team is folded back into them.
"""

import numpy as np

from .match import D, H, A, X
from . import metrics as m


def controlled(labels: np.ndarray, min_control_s: float, fps: int = 25) -> np.ndarray:
    """Remove H/A spells shorter than min_control_s that sit between two spells of the other team.

    Only direct neighbours count: a spell next to a stoppage began or ended with a restart or a
    ball going out, which is a real event. Shortest spells go first, because removing one can
    merge its neighbours into a spell that then survives.
    """
    out = labels.copy()
    limit = int(round(min_control_s * fps))
    if limit <= 0:
        return out
    runs = m.valid_runs(out)
    while True:
        segs = m.segments(out, runs)
        candidates = []
        for i in range(1, len(segs) - 1):
            a, b, k = segs[i]
            if k in (H, A) and b - a < limit:
                before, after = segs[i - 1], segs[i + 1]
                if before[1] == a and after[0] == b and before[2] == after[2] == (A if k == H else H):
                    candidates.append((b - a, a, b, before[2]))
        if not candidates:
            return out
        candidates.sort()
        touched = -1
        for _, a, b, other in candidates:
            if a <= touched:
                continue  # a neighbour changed this pass; re-evaluate it on the next one
            out[a:b] = other
            touched = b
