"""Where do v5, the trees, the TCN and trees + TCN still lose spellF1? Open data only.

The same questions research/analysis/optimisation_2026-11-09 asked of v5 on 10 November, now for four
models side by side, on the 10 open matches held out and at v5's decoder setting:

1. What would one thing done perfectly be worth: the timing of every change we do find, the dead-ball
   calls, or the team calls?
2. Which true possessions are found, by how long they last? "Found" means matched one to one with a
   predicted possession of the same team that overlaps it by at least half (IoU 0.5, the middle of
   the scorer's range).
3. How many possessions are predicted for each true one, and how many of the predicted ones are real.

    python3 experiments/candidate_errors.py      # about 3 minutes
"""

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from mtah import metrics as m  # noqa: E402
from mtah.config import DECODER  # noqa: E402
from mtah.decoder import decode_samples, expand  # noqa: E402
from mtah.match import A, D, H, X  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402
from v6_common import DRAWS, averaged, load  # noqa: E402

BINS = (("under 2 s", 0, 50), ("2 to 4 s", 50, 100), ("4 to 8 s", 100, 200), ("over 8 s", 200, 10 ** 9))
FOUND_AT = 0.5
OUT = os.path.join("runs", "candidate_errors.json")


def snap(gt, pred, tol, kinds="all"):
    """Move each predicted change to the nearest true change with the same from and to labels within tol frames."""
    out = pred.copy()
    for a, b in m.valid_runs(gt):
        g, p = gt[a:b], pred[a:b].copy()
        gt_t = [(i, g[i - 1], g[i]) for i in np.flatnonzero(g[1:] != g[:-1]) + 1]
        pr_t = [(i, p[i - 1], p[i]) for i in np.flatnonzero(p[1:] != p[:-1]) + 1]
        taken, moved = set(), {}
        for i, f, t in pr_t:
            if (kinds == "turnover" and D in (f, t)) or (kinds == "dead" and D not in (f, t)):
                continue
            near = [(abs(j - i), j) for j, f2, t2 in gt_t if f2 == f and t2 == t and abs(j - i) <= tol and j not in taken]
            if near:
                _, j = min(near)
                taken.add(j)
                moved[i] = j
        points = sorted((moved.get(i, i), t) for i, f, t in pr_t)
        q = np.empty_like(p)
        current, last, ok = p[0], 0, True
        for position, t in points:
            if position < last:
                ok = False
                break
            q[last:position] = current
            current, last = t, position
        q[last:] = current
        out[a:b] = q if ok else p
    return out


def perfect_dead(g, p):
    """The true alive and dead. In alive stretches we called dead, the nearest team we did predict."""
    out = p.copy()
    out[g == D] = D
    idx = np.flatnonzero((g != X) & (g != D) & (p == D))
    team = np.flatnonzero((p == H) | (p == A))
    if len(idx) and len(team):
        after = team[np.clip(np.searchsorted(team, idx), 0, len(team) - 1)]
        before = team[np.clip(np.searchsorted(team, idx) - 1, 0, len(team) - 1)]
        out[idx] = p[np.where(np.abs(after - idx) <= np.abs(idx - before), after, before)]
    return out


def perfect_team(g, p):
    """Our alive and dead calls, with the true team wherever we and the truth both say alive."""
    out = p.copy()
    both = (p != D) & ((g == H) | (g == A))
    out[both] = g[both]
    return out


def found(gt, pred):
    """(length in frames, found or not) for each true possession, and (length, real or not) for each predicted one."""
    runs = m.valid_runs(gt)
    true, guessed = m.possession_spells(gt, runs), m.possession_spells(pred, runs)
    used_true, used_guess = set(), set()
    for iou, i, j in m._overlaps(true, guessed):
        if iou < FOUND_AT:
            break
        if i in used_true or j in used_guess:
            continue
        used_true.add(i)
        used_guess.add(j)
    return ([(s[1] - s[0], k in used_true) for k, s in enumerate(true)],
            [(s[1] - s[0], k in used_guess) for k, s in enumerate(guessed)])


if __name__ == "__main__":
    os.chdir(ROOT)
    record = manifest(found_at=FOUND_AT)
    trees, tcn = averaged(DRAWS), load("net_tcn")
    ids = sorted(trees)
    members = {"v5": load("v5"), "trees": trees, "tcn": tcn,
               "trees + tcn": {i: dict(trees[i], prob=(trees[i]["prob"] + tcn[i]["prob"]) / 2.0) for i in ids}}
    truth = {i: trees[i]["labels"] for i in ids}
    pred = {}
    for name, folds in members.items():
        pred[name] = {}
        for i in ids:
            f = folds[i]
            halves = [tuple(int(v) for v in row) for row in f["halves"]]
            pred[name][i] = expand(decode_samples(f["prob"], f["half"], DECODER), f["frames"], f["half"], halves,
                                   len(f["labels"]))

    def score(name, change=lambda g, p: p):
        return float(np.mean([m.evaluate(truth[i], change(truth[i], pred[name][i]))["spell_f1"] for i in ids]))

    rows = (("as predicted", lambda g, p: p),
            ("every change within 1 s put exactly right", lambda g, p: snap(g, p, 25)),
            ("  only changes of team", lambda g, p: snap(g, p, 25, "turnover")),
            ("  only starts and ends of stoppages", lambda g, p: snap(g, p, 25, "dead")),
            ("perfect dead-ball calls, our team calls", perfect_dead),
            ("perfect team calls, our dead-ball calls", perfect_team))
    results = {"one_thing_perfect": {}, "found": {}, "counts": {}}
    print("1. what one thing done perfectly would be worth (spellF1, 10 open matches held out)")
    print(f"  {'':<44}" + "".join(f"{name:>13}" for name in members))
    for label, change in rows:
        values = {name: score(name, change) for name in members}
        results["one_thing_perfect"][label.strip()] = values
        print(f"  {label:<44}" + "".join(f"{values[name]:>13.3f}" for name in members), flush=True)

    both = {name: [found(truth[i], pred[name][i]) for i in ids] for name in members}
    true_all = [row for per_match in both["v5"] for row in per_match[0]]
    print(f"\n2. true possessions found, by how long they last ({len(true_all)} in all)")
    print(f"  {'':<14}{'share of all':>14}" + "".join(f"{name:>13}" for name in members))
    for label, low, high in BINS:
        share = np.mean([low <= length < high for length, _ in true_all])
        cells = {}
        for name in members:
            hits = [ok for per_match in both[name] for length, ok in per_match[0] if low <= length < high]
            cells[name] = float(np.mean(hits))
        results["found"][label] = {"share": float(share), **cells}
        print(f"  {label:<14}{share:>14.0%}" + "".join(f"{cells[name]:>13.0%}" for name in members))
    cells = {name: float(np.mean([ok for per_match in both[name] for _, ok in per_match[0]])) for name in members}
    results["found"]["all"] = cells
    print(f"  {'all':<14}{'100%':>14}" + "".join(f"{cells[name]:>13.0%}" for name in members))

    print("\n3. predicted possessions")
    ratio = {name: float(np.mean([len(per_match[1]) / len(per_match[0]) for per_match in both[name]])) for name in members}
    real = {name: float(np.mean([ok for per_match in both[name] for _, ok in per_match[1]])) for name in members}
    short = {name: float(np.mean([length < 50 for per_match in both[name] for length, _ in per_match[1]])) for name in members}
    results["counts"] = {"predicted_per_true": ratio, "predicted_that_are_real": real, "predicted_under_2s": short}
    print(f"  {'predicted for each true one':<44}" + "".join(f"{ratio[name]:>13.2f}" for name in members))
    print(f"  {'share of predicted ones that are real':<44}" + "".join(f"{real[name]:>13.0%}" for name in members))
    print(f"  {'share of predicted ones under 2 s':<44}" + "".join(f"{short[name]:>13.0%}" for name in members))
    record.update(matches=ids, results=results)
    save_manifest(OUT, record)
