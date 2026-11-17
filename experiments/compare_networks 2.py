"""The trees, the networks, and every way of combining them, on the five tests. Open data only.

Members: the final tree model ("trees": the extra inputs, three draws averaged, no smoother) and
whichever networks scripts/import_network_results.py has brought in (tcn, bilstm). Formats:

  alone      one member
  average    the plain mean of the members' probabilities
  weighted   a weighted mean; on the 10 open matches each match is scored with the weights that did
             best on the other nine, and the other tests use the weights that did best on all ten
  fused      a small model on the members' log-probabilities at the same moment, fitted the same way

Tests, all with v5's decoder setting: the 10 open matches held out; the same predictions against the
stricter labels (single touches under 4 s folded, the written definition); the stress table; a label
source the models never saw; SkillCorner's working half (never the reserve half).

The rule, agreed on 11 Nov 2026 before any network result: a format replaces the trees only if it
gains at least 0.015 on the 10 open matches with at least 8 up, and loses none of the other four
(stress: no row more than 0.01 below the trees). Among those that pass the simplest wins: a plain
average, then weighted, then fused. Nothing is chosen on the competition's public 5 minutes.

    python3 experiments/compare_networks.py      # a few minutes
"""

import argparse
import itertools
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
from mtah import datasets as ds, harness, metrics as m  # noqa: E402
from mtah.config import DECODER  # noqa: E402
from mtah.decoder import decode_samples, expand  # noqa: E402
from mtah.match import X  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402
from mtah.smoother import sample_weights  # noqa: E402
from short_possessions import follow_definition  # noqa: E402
from stress_test import CONDITIONS, HELD_OUT  # noqa: E402
from v6_common import DRAWS, MAX_STRESS_LOSS, MIN_GAIN, MIN_UP, averaged, load  # noqa: E402

NETWORKS = ("tcn", "bilstm")
TESTS = ("open matches", "stricter labels", "stress table", "unseen source", "SkillCorner")
TREE_STRESS = os.path.join("runs", "stress_v6_parts")
OUT = os.path.join("runs", "compare_networks.json")
STEPS = 10          # weights are tried in tenths


def stress_from(folder, key):
    out = {}
    for condition in CONDITIONS:
        for held in HELD_OUT:
            path = os.path.join(folder, condition.replace(" ", "_").replace("%", "pct") + f"__{held}.npz")
            if os.path.exists(path):
                with np.load(path, allow_pickle=False) as z:
                    out[(condition, held)] = {"prob": z[key], **{k: z[k] for k in ("frames", "half", "labels", "halves")}}
    return out


def saved(tag):
    return load(tag) if os.path.isdir(os.path.join("runs", "oof", tag)) else {}


def members(demo):
    out = {"v5": {"open matches": load("v5"), "unseen source": saved("v5_unseen"), "SkillCorner": saved("v5_sc"),
                  "stress table": stress_from(TREE_STRESS, "prob0")},
           "trees": {"open matches": averaged(DRAWS), "unseen source": saved("trees_unseen"),
                     "SkillCorner": saved("trees_sc"), "stress table": stress_from(TREE_STRESS, "prob4")}}
    for name in NETWORKS:
        if os.path.isdir(os.path.join("runs", "oof", f"net_{name}")):
            out[name] = {"open matches": load(f"net_{name}"), "unseen source": saved(f"net_{name}_unseen"),
                         "SkillCorner": saved(f"net_{name}_sc"),
                         "stress table": stress_from(os.path.join("runs", "stress_networks", name), "prob")}
    if demo:
        out["demo"] = dict(out["v5"])
    for parts in out.values():
        parts["stricter labels"] = parts["open matches"]
    return out


def predicted(fold, prob):
    halves = [tuple(int(v) for v in row) for row in fold["halves"]]
    return expand(decode_samples(prob, fold["half"], DECODER), fold["frames"], fold["half"], halves, len(fold["labels"]))


def inputs_of(parts):
    return np.log(np.concatenate(parts, axis=1) + 1e-4)


def combined(kind, parts, how):
    if kind == "weighted":
        return sum(w * p for w, p in zip(how, parts))
    if kind == "fused":
        return how.predict_proba(inputs_of(parts))
    return sum(parts) / float(len(parts))


def fit_fusion(folds_by_member, ids):
    """A model from the members' log-probabilities at a moment to the label, weighted as the trees' training is."""
    from sklearn.linear_model import LogisticRegression
    xs, ys, ws = [], [], []
    for i in ids:
        first = folds_by_member[0][i]
        y = first["labels"][first["frames"]]
        keep = y != X
        xs.append(inputs_of([folds[i]["prob"] for folds in folds_by_member])[keep])
        ys.append(y[keep])
        ws.append(sample_weights(first["labels"], first["frames"])[keep])
    model = LogisticRegression(max_iter=300)
    model.fit(np.vstack(xs), np.concatenate(ys), sample_weight=np.concatenate(ws))
    return model


def simplex(n):
    return [tuple(v / STEPS for v in c) for c in itertools.product(range(STEPS + 1), repeat=n) if sum(c) == STEPS]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demo", action="store_true", help="treat v5 as a network called demo, to test this script")
    args = parser.parse_args()
    os.chdir(ROOT)
    record = manifest(demo=args.demo)
    all_members = members(args.demo)
    nets = [name for name in all_members if name not in ("v5", "trees")]
    if not nets:
        sys.exit("no network predictions yet: run scripts/import_network_results.py --model tcn (or bilstm) first")
    team = ["trees"] + nets
    formats = [("v5", ["v5"], "alone"), ("trees", ["trees"], "alone")] + [(n, [n], "alone") for n in nets]
    for size in range(1, len(nets) + 1):
        for chosen in itertools.combinations(nets, size):
            formats.append(("trees + " + " + ".join(chosen) + ", average", ["trees"] + list(chosen), "average"))
    if len(nets) > 1:
        formats.append((" + ".join(nets) + ", average", list(nets), "average"))
    formats.append((" + ".join(team) + ", weighted", team, "weighted"))
    formats.append((" + ".join(team) + ", fused", team, "fused"))

    home = [all_members[name]["open matches"] for name in team]
    ids = sorted(set.intersection(*(set(folds) for folds in home)) & set(all_members["v5"]["open matches"]))
    for i in ids:
        if any(not np.array_equal(folds[i]["frames"], home[0][i]["frames"]) for folds in home):
            raise ValueError(f"{i}: the members are not predicted at the same moments")
    strict = {i: follow_definition(harness.harmonise(ds.load_open(i)), 4.0) for i in ids}
    print(f"members: {', '.join(team)}; {len(ids)} open matches\n", flush=True)

    # Weights and fusion: chosen on the other matches for each open match, and on all of them for the other tests.
    grid = simplex(len(team))
    table = np.array([[m.evaluate(home[0][i]["labels"],
                                  predicted(home[0][i], combined("weighted", [f[i]["prob"] for f in home], w)))["spell_f1"]
                       for i in ids] for w in grid])
    nested_weights = {i: grid[int(np.delete(table, j, axis=1).mean(1).argmax())] for j, i in enumerate(ids)}
    final_weights = grid[int(table.mean(1).argmax())]
    nested_fusion = {i: fit_fusion(home, [k for k in ids if k != i]) for i in ids}
    final_fusion = fit_fusion(home, ids)
    print("weights that did best on all open matches: " + ", ".join(f"{n} {w:.1f}" for n, w in zip(team, final_weights)))
    print("weights used when a match is held out: " + "; ".join(
        sorted({", ".join(f"{n} {w:.1f}" for n, w in zip(team, ws)) for ws in nested_weights.values()})) + "\n", flush=True)

    scores = {}                                  # scores[test][format] = {fold: spellF1}
    for test in TESTS:
        scores[test] = {}
        for name, who, kind in formats:
            folds = [all_members[n].get(test, {}) for n in who]
            keys = sorted(set.intersection(*(set(f) for f in folds))) if all(folds) else []
            if test in ("open matches", "stricter labels"):
                keys = [k for k in keys if k in ids]
            out = {}
            for key in keys:
                parts = [f[key]["prob"] for f in folds]
                nested = test in ("open matches", "stricter labels")
                how = None
                if kind == "weighted":
                    how = nested_weights[key] if nested else final_weights
                elif kind == "fused":
                    how = nested_fusion[key] if nested else final_fusion
                pred = predicted(folds[0][key], combined(kind, parts, how))
                truth = strict[key] if test == "stricter labels" else folds[0][key]["labels"]
                out[key] = m.evaluate(truth, pred)["spell_f1"]
            scores[test][name] = out

    results, verdicts = {}, {}
    for test in TESTS:
        base = scores[test]["trees"]
        print(f"{test}  ({len(base)} {'copies' if test == 'stress table' else 'matches'})")
        print(f"  {'':<44}{'spellF1':>8}{'against the trees':>19}{'up/down':>9}" + ("   worst row" if test == "stress table" else ""))
        results[test] = {}
        for name, _, _ in formats:
            mine = scores[test][name]
            keys = [k for k in base if k in mine]
            if not keys:
                print(f"  {name:<44}  not there yet")
                continue
            gain = np.array([mine[k] - base[k] for k in keys])
            row = {"spell_f1": float(np.mean([mine[k] for k in keys])), "gain": float(gain.mean()),
                   "up": int((gain > 0).sum()), "down": int((gain < 0).sum()), "n": len(keys)}
            tail = ""
            if test == "stress table":
                rows = [np.mean([mine[(c, h)] - base[(c, h)] for h in HELD_OUT if (c, h) in mine]) for c in CONDITIONS
                        if any((c, h) in mine for h in HELD_OUT)]
                row.update(worst_row=float(min(rows)), rows_up=int(sum(r > 0 for r in rows)), rows=len(rows))
                tail = f"   {row['worst_row']:+.3f} ({row['rows_up']} of {row['rows']} rows up)"
            results[test][name] = row
            print(f"  {name:<44}{row['spell_f1']:>8.3f}{row['gain']:>+19.3f}{row['up']:>6}/{row['down']:<2}{tail}")
        print(flush=True)

    print(f"the rule: at least +{MIN_GAIN} on the open matches with {MIN_UP} of 10 up, and no loss on the other four")
    passing = []
    for name, who, kind in formats:
        if name in ("v5", "trees"):
            continue
        r = {test: results[test].get(name) for test in TESTS}
        if any(v is None for v in r.values()):
            verdicts[name] = "not judged yet: " + ", ".join(t for t, v in r.items() if v is None) + " missing"
        else:
            failed = []
            if r["open matches"]["gain"] < MIN_GAIN or r["open matches"]["up"] < MIN_UP or r["open matches"]["n"] < 10:
                failed.append("open matches")
            failed += [t for t in ("stricter labels", "unseen source", "SkillCorner") if r[t]["gain"] < 0]
            if r["stress table"]["worst_row"] < -MAX_STRESS_LOSS:
                failed.append("stress table")
            verdicts[name] = "PASSES" if not failed else "does not pass: " + ", ".join(failed)
            if not failed:
                passing.append((("average", "weighted", "fused", "alone").index(kind), len(who), name))
        print(f"  {name:<44}{verdicts[name]}")
    # The rule ranks the kinds of format and says nothing more. Several formats of the simplest kind that
    # passes are a tie: they are all listed, and choosing between them is the team's decision.
    simplest = sorted(name for kind, _, name in passing if kind == min(passing)[0]) if passing else []
    choice = simplest[0] if len(simplest) == 1 else None
    if not passing:
        print("\nby the rule: nothing replaces the trees")
    elif choice:
        print(f"\nby the rule: '{choice}' replaces the trees (the simplest format that passes)")
    else:
        print("\nby the rule: a format of the simplest kind replaces the trees. Several pass, and the rule does not "
              "choose between them:\n  " + "\n  ".join(simplest))
    record.update(members=team, matches=ids, final_weights=dict(zip(team, final_weights)), results=results,
                  verdicts=verdicts, choice=choice, simplest_passing=simplest)
    if not args.demo:
        save_manifest(OUT, record)
