"""What do the labels on the open matches actually look like?

Prints per-match statistics so label meaning can be compared with the competition's rules. Open
data only. Writes runs/label_stats.json.

    python3 scripts/inspect_labels.py [MATCH_ID ...]
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mtah import datasets as ds, metrics as m  # noqa: E402
from mtah.match import A, D, H, X  # noqa: E402


def stats(x):
    runs = m.valid_runs(x.labels)
    g = x.labels != X
    sp = m.possession_spells(x.labels, runs)
    seg = m.segments(x.labels, runs)
    dur = np.array([(b - a) / x.fps for a, b, _ in sp])
    dead = np.array([(b - a) / x.fps for a, b, k in seg if k == D])
    trans = m.transitions(x.labels, runs)
    direct = sum(1 for _, f, t in trans if {f, t} == {H, A})
    seen = np.isfinite(x.ball[:, 0])
    players = np.isfinite(x.players[..., 0]).sum(1)
    return {
        "match": x.match_id, "minutes": round(float(g.sum()) / x.fps / 60, 1),
        "share_H": round(float((x.labels[g] == H).mean()), 3),
        "share_A": round(float((x.labels[g] == A).mean()), 3),
        "share_D": round(float((x.labels[g] == D).mean()), 3),
        "spells": len(sp), "per_5min": round(len(sp) / (float(g.sum()) / x.fps / 300), 1),
        "spell_mean_s": round(float(dur.mean()), 1), "spell_median_s": round(float(np.median(dur)), 1),
        "spell_under_1s": round(float((dur < 1).mean()), 3),
        "spell_under_3s": round(float((dur < 3).mean()), 3),
        "dead_n": len(dead), "dead_median_s": round(float(np.median(dead)), 1),
        "direct_turnover_share": round(direct / max(len(trans), 1), 3),
        "ball_seen_alive": round(float(seen[(x.labels == H) | (x.labels == A)].mean()), 3),
        "ball_seen_dead": round(float(seen[x.labels == D].mean()), 3),
        "players_per_frame": round(float(players[g].mean()), 1),
    }


if __name__ == "__main__":
    ids = sys.argv[1:] or ds.available()
    rows = []
    for match_id in ids:
        row = stats(ds.load_open(match_id))
        rows.append(row)
        print(json.dumps(row))
    os.makedirs("runs", exist_ok=True)
    with open("runs/label_stats.json", "w") as handle:
        json.dump(rows, handle, indent=1)
