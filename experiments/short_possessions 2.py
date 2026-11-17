"""What are the short possessions the model misses? Open matches only, reads saved predictions.

Sorts every possession under 4 s by what the clean tracking shows happened, and by what the model
did with it. Writes one row per possession to runs/short_possessions_<tag>.csv, and optionally a
picture of random examples. Then rescores the same predictions against labels that follow the
organisers' written definition more closely, to size the gap between that and the open-match labels.

The sorting rules are rules of thumb on 2D tracking, not a second ground truth.

    python3 experiments/short_possessions.py --tag v5 --sheet ../research/analysis/short_possession_examples.png
"""

import argparse
import collections
import csv
import glob
import os
import sys
import types

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from mtah import datasets as ds, harness, metrics as m  # noqa: E402
from mtah.config import DECODER  # noqa: E402
from mtah.decoder import decode_samples, expand  # noqa: E402
from mtah.degrade import DegradeConfig, load_stats  # noqa: E402
from mtah.loaders import dfl  # noqa: E402
from mtah.match import A, D, H  # noqa: E402
from sweep_decoder import load  # noqa: E402

FPS = 25
LIMIT_S = 4.0
TOUCH_M = 1.5      # the nearest player is on the ball within this distance
CARRY_MPS = 8.0    # faster than this and the ball is travelling, not being carried
HOLD_S = 0.6       # a touch run this long at carrying speed counts as keeping the ball
BEFORE_S = 0.8     # DFL labels trail the tracking, so look earlier than the label says
AFTER_S = 0.2
KINDS = ("passed between teammates", "one player kept it", "one touch, then ball in flight",
         "one touch, quick change", "no clear touch")
LOOSE = KINDS[2:]  # no sign of control, so the written definition says possession hasn't changed


def event_reader():
    """The DFL event reader kept in experiments/dfl_event_labels.patch, loaded without applying it."""
    with open(os.path.join(ROOT, "experiments", "dfl_event_labels.patch")) as handle:
        block = handle.read().split("diff --git a/mtah/loaders/dfl_events.py")[1].split("\ndiff --git")[0]
    source = "\n".join(line[1:] for line in block.splitlines()
                       if line.startswith("+") and not line.startswith("+++"))
    source = source.replace("from .. import metrics as m", "from mtah import metrics as m")
    source = source.replace("from ..match import", "from mtah.match import")
    module = types.ModuleType("dfl_events")
    sys.modules["dfl_events"] = module  # dataclasses looks the module up by name
    exec(compile(source, "dfl_events", "exec"), module.__dict__)  # noqa: S102, our own saved code
    return module


def touch_runs(toucher):
    """(player, first, last) for each stretch of two or more frames with one player on the ball."""
    out, t = [], 0
    while t < len(toucher):
        if toucher[t] < 0:
            t += 1
            continue
        u = t
        while u + 1 < len(toucher) and toucher[u + 1] == toucher[t]:
            u += 1
        if u > t:
            out.append((int(toucher[t]), t, u))
        t = u + 1
    return out


def what_happened(spell, match, toucher, speed):
    """Which of KINDS this possession looks like, from the clean tracking."""
    start, stop, code = spell
    side = 0 if code == 0 else 1
    lo, hi = max(start - int(BEFORE_S * FPS), 0), min(stop + int(AFTER_S * FPS), len(toucher))
    own = [r for r in touch_runs(toucher[lo:hi]) if match.team[r[0]] == side]
    if not own:
        return "no clear touch"
    if len({r[0] for r in own}) >= 2:
        return "passed between teammates"
    hold = max(int((speed[lo + r[1]:lo + r[2] + 1] < CARRY_MPS).sum()) for r in own) / FPS
    if hold >= HOLD_S:
        return "one player kept it"
    flight = max(stop - (lo + own[-1][2]), 0) / FPS
    if flight >= 1.0 or flight >= 0.5 * (stop - start) / FPS:
        return "one touch, then ball in flight"
    return "one touch, quick change"


def ball_contact(match):
    """Who is on the ball on each frame (-1 for nobody), and how fast it is moving."""
    gap = match.players - match.ball[:, None, :]
    gap = np.hypot(gap[..., 0], gap[..., 1])
    gap = np.where(np.isfinite(gap), gap, np.inf)
    move = np.roll(match.ball, -2, 0) - np.roll(match.ball, 2, 0)
    return np.where(gap.min(1) < TOUCH_M, gap.argmin(1), -1), np.hypot(move[:, 0], move[:, 1]) / (4 / FPS)


def follow_definition(match, limit_s=LIMIT_S):
    """Labels with single-touch spells in open play given back to the team that had the ball.

    Needs clean tracking. A spell after a stoppage stays, because every restart has a clear possessor.
    """
    toucher, speed = ball_contact(match)
    out = match.labels.copy()
    for spell in m.possession_spells(match.labels, m.valid_runs(match.labels)):
        if spell[0] and (spell[1] - spell[0]) / FPS < limit_s and out[spell[0] - 1] in (H, A) \
                and what_happened(spell, match, toucher, speed) in LOOSE:
            out[spell[0]:spell[1]] = out[spell[0] - 1]
    return out


def what_the_model_did(spell, hit, path, log_p, frames):
    """found / ran on / decoder removed / never leaned. Evidence may sit early, as labels can trail."""
    if hit:
        return "found"
    idx = np.flatnonzero((frames >= spell[0]) & (frames < spell[1]))
    if len(idx) == 0 or (path[idx] == spell[2]).mean() > 0.5:
        return "ran on"
    best = -np.inf
    for shift in (0, -1, -2, -3):
        j = np.clip(idx + shift, 0, len(path) - 1)
        best = max(best, float((log_p[j, spell[2]] - log_p[j, path[j]]).sum()))
    return "decoder removed" if best > 0 else "never leaned"


def clock(frame, halves):
    half = 0 if frame < halves[1][0] else 1
    seconds = (frame - halves[half][0]) / FPS
    return half + 1, f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def analyse(tag):
    folds, rows = load(tag), []
    stats, cfg, events = load_stats(), DegradeConfig(), event_reader()
    for match_id in sorted(folds):
        fold = folds[match_id]
        labels, frames = fold["labels"], fold["frames"]
        halves = [tuple(int(v) for v in r) for r in fold["halves"]]
        clean = harness.harmonise(ds.load_open(match_id))
        if not np.array_equal(clean.labels, labels):
            raise ValueError(f"{match_id}: saved labels differ from the open match")
        seen = ~np.isnan(harness.prepared(match_id, cfg, harness.TEST_SEED_OFFSET, stats).ball[:, 0])

        toucher, speed = ball_contact(clean)

        marks = None
        if match_id.startswith("J"):
            info = glob.glob(os.path.join(ds.OPEN_DIR, "dfl", f"DFL_02_01_matchinformation_*{match_id}.xml"))[0]
            home, away, _ = dfl._match_info(info)
            marks = np.array(events.event_labels(ds.load_open(match_id), ds._dfl_file("03_02_events_raw", match_id),
                                                 ds._dfl_file("04_03_positions_raw_observed", match_id),
                                                 home, away, only_marks=True))

        path = decode_samples(fold["prob"], fold["half"], DECODER)
        pred = expand(path, frames, fold["half"], halves, len(labels))
        log_p = np.log(np.clip(fold["prob"], 1e-6, 1.0))
        runs = m.valid_runs(labels)
        spells, predicted = m.possession_spells(labels, runs), m.possession_spells(pred, runs)
        hits = m._matched(m._overlaps(spells, predicted), len(spells), 0.5)
        for i, spell in enumerate(spells):
            seconds = (spell[1] - spell[0]) / FPS
            if seconds >= LIMIT_S:
                continue
            half, time = clock(spell[0], halves)
            event = ""
            if marks is not None:
                event = bool(((marks[:, 1] == spell[2]) & (marks[:, 0] >= spell[0] - FPS)
                              & (marks[:, 0] < spell[1] + FPS)).any())
            rows.append({"match": match_id, "half": half, "clock": time, "start": spell[0], "stop": spell[1],
                         "seconds": round(seconds, 2), "team": "HA"[spell[2]],
                         "before": "dead" if spell[0] and labels[spell[0] - 1] == D else "other",
                         "after": "dead" if spell[1] < len(labels) and labels[spell[1]] == D else "other",
                         "ball_seen": round(float(seen[spell[0]:spell[1]].mean()), 2),
                         "model": what_the_model_did(spell, bool(hits[i]), path, log_p, frames),
                         "kind": what_happened(spell, clean, toucher, speed), "dfl_event": event})
    return folds, rows


def report(rows):
    def table(title, chosen):
        count = collections.Counter(r["kind"] for r in chosen)
        print(f"{title:<30}{len(chosen):>6}" + "".join(f"{count[k]:>6} {count[k] / max(len(chosen), 1):>4.0%}" for k in KINDS))

    missed = [r for r in rows if r["model"] == "never leaned"]
    visible = [r for r in missed if r["ball_seen"] >= 0.5]
    found = [r for r in rows if r["model"] == "found"]
    print(f"{len(rows)} possessions under {LIMIT_S:g} s\n")
    print(f"{'':<36}" + "".join(f"{k[:10]:>11}" for k in KINDS))
    table("found", found)
    table("never leaned, ball visible", visible)
    table("never leaned, ball hidden", [r for r in missed if r["ball_seen"] < 0.5])
    table("ran on into a neighbour", [r for r in rows if r["model"] == "ran on"])

    print("\nstart straight after a stoppage")
    for name, chosen in (("found", found), ("never leaned, ball visible", visible)):
        print(f"  {name:<28}{np.mean([r['before'] == 'dead' for r in chosen]):>5.0%}")
    print("end at a stoppage")
    for name, chosen in (("found", found), ("never leaned, ball visible", visible)):
        print(f"  {name:<28}{np.mean([r['after'] == 'dead' for r in chosen]):>5.0%}")
    print("DFL matches: logged control event by that team within 1 s")
    for name, chosen in (("found", found), ("never leaned, ball visible", visible)):
        print(f"  {name:<28}{np.mean([r['dfl_event'] for r in chosen if r['dfl_event'] != '']):>5.0%}")

    print("\nwhat the model did")
    for name, test in (("signs of control", lambda r: r["kind"] not in LOOSE), ("single touch or none", lambda r: r["kind"] in LOOSE)):
        chosen = [r for r in rows if test(r)]
        count = collections.Counter(r["model"] for r in chosen)
        print(f"  {name:<22}{len(chosen):>6}  " + "  ".join(f"{k} {count[k] / len(chosen):.0%}"
                                                          for k in ("found", "ran on", "decoder removed", "never leaned")))


def rescore(folds):
    print("\nsame predictions, labels closer to the written definition")
    for limit in (0.0, 2.0, 4.0):
        scores, ratio, total = [], [], 0
        for match_id, fold in sorted(folds.items()):
            halves = [tuple(int(v) for v in r) for r in fold["halves"]]
            pred = expand(decode_samples(fold["prob"], fold["half"], DECODER), fold["frames"], fold["half"],
                          halves, len(fold["labels"]))
            result = m.evaluate(follow_definition(harness.harmonise(ds.load_open(match_id)), limit), pred)
            scores.append(result["spell_f1"])
            ratio.append(result["n_pred_spells"] / result["n_gt_spells"])
            total += result["n_gt_spells"]
        print(f"  folding under {limit:g} s: spellF1 {np.mean(scores):.3f}  possessions {total}  predicted x{np.mean(ratio):.2f}")


def picture(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    visible = [r for r in rows if r["model"] == "never leaned" and r["ball_seen"] >= 0.5]
    groups = (("Passed between teammates", lambda r: r["kind"] == KINDS[0]),
              ("One touch, then ball out of play", lambda r: r["kind"] == KINDS[2] and r["after"] == "dead"),
              ("One touch, then the other team", lambda r: r["kind"] == KINDS[2] and r["after"] == "other"),
              ("No clear touch", lambda r: r["kind"] == KINDS[4]))
    rng, cache = np.random.default_rng(3), {}
    fig, axes = plt.subplots(4, 4, figsize=(16, 15))
    for row, (name, test) in zip(axes, groups):
        chosen = [r for r in visible if test(r)]
        for ax, i in zip(row, rng.choice(len(chosen), 4, replace=False)):
            r = chosen[i]
            match = cache.setdefault(r["match"], harness.harmonise(ds.load_open(r["match"])))
            side = 0 if r["team"] == "H" else 1
            lo, hi = r["start"] - 20, r["stop"] + 25
            ball = match.ball[lo:hi]
            ax.scatter(ball[:, 0], ball[:, 1], c=np.arange(hi - lo), cmap="viridis", s=12, zorder=3)
            ax.plot(*ball[r["start"] - lo:r["stop"] - lo].T, color="black", lw=2.2, zorder=2)
            for p in range(match.players.shape[1]):
                track = match.players[lo:hi, p]
                if np.isfinite(track[:, 0]).any() and np.nanmin(np.hypot(*(track - ball).T)) <= 6:
                    colour = "#c4372c" if match.team[p] == side else "#2f6db5"
                    ax.plot(*track.T, color=colour, lw=1.2, alpha=0.8)
                    ax.scatter(*track[-1], color=colour, s=18, zorder=4)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(f"{name}\n{r['match']}, half {r['half']}, {r['clock']}, {r['seconds']:.1f} s", fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle("Missed short possessions with the ball visible, on clean tracking. Metres.\n"
                 "Ball runs purple to yellow; black is the labelled possession. "
                 "Red is the labelled team, blue the other team; dots mark where players end.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="v5")
    parser.add_argument("--sheet", help="also save a picture of random examples here")
    args = parser.parse_args()

    os.chdir(ROOT)
    all_folds, all_rows = analyse(args.tag)
    report(all_rows)
    rescore(all_folds)
    out = os.path.join("runs", f"short_possessions_{args.tag}.csv")
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nwrote {out}")
    if args.sheet:
        picture(all_rows, args.sheet)
        print(f"wrote {args.sheet}")
