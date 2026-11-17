"""Three follow-ups to compare_networks.py. For understanding the result, not for picking a weight. Open data only.

1. How often the trees and the two networks give a different answer at the same moment.
2. Head to head, match by match, between the plain averages that pass the rule, and against the same
   split without the second network (trees 1/3, TCN 2/3). That last one shows whether the BiLSTM adds
   anything the TCN's larger share does not.
3. What the trees' share does: the trees get a share in tenths and the TCN the rest, on the five tests.

Everything is at v5's decoder setting. The reserve half of SkillCorner is never loaded.

    python3 experiments/compare_networks_detail.py      # about 4 minutes
"""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import compare_networks as cn  # noqa: E402

THIRD = 1.0 / 3.0
FORMATS = {
    "trees": {"trees": 1.0},
    "tcn": {"tcn": 1.0},
    "bilstm": {"bilstm": 1.0},
    "trees + tcn": {"trees": 0.5, "tcn": 0.5},
    "trees + bilstm": {"trees": 0.5, "bilstm": 0.5},
    "all three": {"trees": THIRD, "tcn": THIRD, "bilstm": THIRD},
    "trees 1/3, tcn 2/3": {"trees": THIRD, "tcn": 2 * THIRD},
}
PAIRS = (("all three", "trees + tcn"), ("all three", "trees 1/3, tcn 2/3"), ("trees + tcn", "trees + bilstm"))
SHARES = {0: "tcn", 5: "trees + tcn", 10: "trees"}      # tenths already scored above
OUT = os.path.join("runs", "compare_networks_detail.json")


def scores(all_members, weights, test, ids, strict):
    """For each match (stress: each copy): spellF1, and how many possessions were predicted and how many are true."""
    folds = [all_members[name][test] for name in weights]
    keys = sorted(set.intersection(*(set(f) for f in folds)))
    if test in ("open matches", "stricter labels"):
        keys = [k for k in keys if k in ids]
    out = {}
    for key in keys:
        prob = sum(w * f[key]["prob"] for w, f in zip(weights.values(), folds))
        truth = strict[key] if test == "stricter labels" else folds[0][key]["labels"]
        result = cn.m.evaluate(truth, cn.predicted(folds[0][key], prob))
        out[key] = (result["spell_f1"], result["n_pred_spells"], result["n_gt_spells"])
    return out


def answers(all_members, ids):
    """Before the decoder: how often each member names the wrong class, and how often two members differ."""
    names = ("trees", "tcn", "bilstm")
    wrong = {name: [] for name in names}
    pairs = {pair: {"differ": [], "both_wrong": []} for pair in itertools.combinations(names, 2)}
    for i in ids:
        fold = all_members["trees"]["open matches"][i]
        truth = fold["labels"][fold["frames"]]
        keep = truth != cn.X
        said = {name: all_members[name]["open matches"][i]["prob"][keep].argmax(1) for name in names}
        for name in names:
            wrong[name].append(float((said[name] != truth[keep]).mean()))
        for (a, b), row in pairs.items():
            row["differ"].append(float((said[a] != said[b]).mean()))
            row["both_wrong"].append(float(((said[a] != truth[keep]) & (said[b] != truth[keep])).mean()))
    return ({name: float(np.mean(v)) for name, v in wrong.items()},
            {f"{a} and {b}": {k: float(np.mean(v)) for k, v in row.items()} for (a, b), row in pairs.items()})


def mean(rows):
    return float(np.mean([v[0] for v in rows.values()]))


def worst_row(mine, base):
    return float(min(np.mean([mine[(c, h)][0] - base[(c, h)][0] for h in cn.HELD_OUT if (c, h) in mine])
                     for c in cn.CONDITIONS if any((c, h) in mine for h in cn.HELD_OUT)))


if __name__ == "__main__":
    os.chdir(cn.ROOT)
    record = cn.manifest()
    all_members = cn.members(False)
    ids = sorted(all_members["tcn"]["open matches"])
    wrong, pairs = answers(all_members, ids)
    print("answers at the same moment, before the decoder, 10 open matches held out")
    print("  wrong: " + ", ".join(f"{name} {100 * share:.1f}%" for name, share in wrong.items()))
    for pair, row in pairs.items():
        print(f"  {pair}: a different answer on {100 * row['differ']:.1f}% of moments, both wrong on "
              f"{100 * row['both_wrong']:.1f}%")
    print(flush=True)

    strict = {i: cn.follow_definition(cn.harness.harmonise(cn.ds.load_open(i)), 4.0) for i in ids}
    table = {name: {test: scores(all_members, weights, test, ids, strict) for test in cn.TESTS}
             for name, weights in FORMATS.items()}

    print("spellF1 (stress table: mean of 57 copies)")
    print(f"  {'':<22}" + "".join(f"{test:>17}" for test in cn.TESTS))
    for name in FORMATS:
        print(f"  {name:<22}" + "".join(f"{mean(table[name][test]):>17.3f}" for test in cn.TESTS))

    print("\nhead to head: mean difference, first minus second, and on how many matches (stress: copies) the first is "
          "ahead / behind")
    print(f"  {'':<40}" + "".join(f"{test:>17}" for test in cn.TESTS))
    head = {}
    for first, second in PAIRS:
        cells = []
        for test in cn.TESTS:
            gain = np.array([table[first][test][k][0] - table[second][test][k][0] for k in table[first][test]])
            head[f"{first} minus {second}: {test}"] = {"gain": float(gain.mean()), "ahead": int((gain > 0).sum()),
                                                      "behind": int((gain < 0).sum())}
            cells.append(f"{gain.mean():+.3f} {(gain > 0).sum():>2}/{(gain < 0).sum():<2}")
        print(f"  {first + ' minus ' + second:<40}" + "".join(f"{c:>17}" for c in cells))

    print("\npredicted possessions for each true one, 10 open matches: " + ", ".join(
        f"{name} x{sum(v[1] for v in table[name]['open matches'].values()) / sum(v[2] for v in table[name]['open matches'].values()):.2f}"
        for name in FORMATS), flush=True)

    print("\nthe trees' share, the TCN taking the rest (spellF1; the last column is the stress table's worst row against "
          "the trees)")
    print(f"  {'share':<8}" + "".join(f"{test:>17}" for test in cn.TESTS) + f"{'worst row':>12}")
    curve = {}
    for tenth in range(11):
        share = tenth / 10
        rows = table[SHARES[tenth]] if tenth in SHARES else {
            test: scores(all_members, {"trees": share, "tcn": 1 - share}, test, ids, strict) for test in cn.TESTS}
        worst = worst_row(rows["stress table"], table["trees"]["stress table"])
        curve[f"{share:.1f}"] = {**{test: mean(rows[test]) for test in cn.TESTS}, "worst_row": worst}
        print(f"  {share:<8.1f}" + "".join(f"{mean(rows[test]):>17.3f}" for test in cn.TESTS) + f"{worst:>+12.3f}", flush=True)

    record.update(matches=ids, wrong=wrong, pairs=pairs, head_to_head=head, trees_share=curve,
                  formats={name: {test: {"__".join(k) if isinstance(k, tuple) else k: v[0]
                                         for k, v in table[name][test].items()} for test in cn.TESTS}
                           for name in FORMATS})
    cn.save_manifest(OUT, record)
