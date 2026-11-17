"""The candidate (trees + TCN, averaged) against v5, and its scores by supplier and by label reading.

Two questions for comparing ourselves with what other teams publish. Open data only.

1. Does the candidate clear, against v5, the bar that was set for the v6 candidate on 10 November?
   That bar: at least +0.015 on the 10 open matches with at least 8 up, no stress row more than 0.01
   below v5, and a lead on SkillCorner's working half that is not smaller.
2. What does it score on the German matches and on Metrica's separately, and under three readings of
   the labels: ours, single touches under 2 s folded, and under 4 s folded? Other teams report scores on
   the same public matches under label rules of their own, so one number of ours is not enough.

Everything is at v5's decoder setting. The reserve half of SkillCorner is never loaded.

    python3 experiments/candidate_against_v5.py      # about 2 minutes
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import compare_networks as cn  # noqa: E402

OUT = os.path.join("runs", "candidate_against_v5.json")
READINGS = (("our labels", None), ("touches under 2 s folded", 2.0), ("touches under 4 s folded", 4.0))


def labels_for(members, test, key):
    fold = members["trees"][test][key]
    parts = {"v5": members["v5"][test][key]["prob"], "trees": fold["prob"], "tcn": members["tcn"][test][key]["prob"]}
    parts["trees + tcn"] = (parts["trees"] + parts["tcn"]) / 2.0
    return fold, {name: cn.predicted(fold, prob) for name, prob in parts.items()}


if __name__ == "__main__":
    os.chdir(cn.ROOT)
    record = cn.manifest()
    members = cn.members(False)
    ids = sorted(members["tcn"]["open matches"])

    print("1. the candidate against v5")
    against = {}
    for test in ("open matches", "stress table", "unseen source", "SkillCorner"):
        keys = sorted(set(members["v5"][test]) & set(members["trees"][test]) & set(members["tcn"][test]))
        if test == "open matches":
            keys = [k for k in keys if k in ids]
        gain = {}
        for key in keys:
            fold, labels = labels_for(members, test, key)
            gain[key] = (cn.m.evaluate(fold["labels"], labels["trees + tcn"])["spell_f1"]
                         - cn.m.evaluate(fold["labels"], labels["v5"])["spell_f1"])
        values = np.array(list(gain.values()))
        row = {"gain": float(values.mean()), "up": int((values > 0).sum()), "down": int((values < 0).sum()), "n": len(values)}
        tail = ""
        if test == "stress table":
            rows = [np.mean([gain[(c, h)] for h in cn.HELD_OUT if (c, h) in gain]) for c in cn.CONDITIONS
                    if any((c, h) in gain for h in cn.HELD_OUT)]
            row.update(worst_row=float(min(rows)), rows_up=int(sum(r > 0 for r in rows)), rows=len(rows))
            tail = f"; worst row {row['worst_row']:+.3f}, {row['rows_up']} of {row['rows']} rows up"
        against[test] = row
        print(f"  {test:<16} {row['gain']:+.3f}, {row['up']} up / {row['down']} down of {row['n']}{tail}", flush=True)

    print("\n2. the 10 open matches held out, by supplier and by reading of the labels (spellF1)")
    scores = {}
    for reading, seconds in READINGS:
        for i in ids:
            fold, labels = labels_for(members, "open matches", i)
            truth = fold["labels"] if seconds is None else cn.follow_definition(cn.harness.harmonise(cn.ds.load_open(i)), seconds)
            for name, pred in labels.items():
                scores[(reading, name, i)] = cn.m.evaluate(truth, pred)["spell_f1"]
    groups = (("7 German matches", [i for i in ids if not i.startswith("METRICA")]),
              ("3 Metrica matches", [i for i in ids if i.startswith("METRICA")]), ("all 10", ids))
    names = ("v5", "trees", "tcn", "trees + tcn")
    by_group = {}
    for reading, _ in READINGS:
        print(f"  {reading}")
        print(f"    {'':<20}" + "".join(f"{name:>13}" for name in names))
        for label, group in groups:
            cells = {name: float(np.mean([scores[(reading, name, i)] for i in group])) for name in names}
            by_group[f"{reading}: {label}"] = cells
            print(f"    {label:<20}" + "".join(f"{cells[name]:>13.3f}" for name in names))
    print(flush=True)
    record.update(matches=ids, against_v5=against, by_group=by_group)
    cn.save_manifest(OUT, record)
