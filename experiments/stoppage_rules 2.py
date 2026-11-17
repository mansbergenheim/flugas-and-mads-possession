"""Two hunches about stoppages, tested on the open matches. Open data only.

1. Are offside stoppages harder for trees + TCN than other stoppages? The seven German matches log
   every offside and every foul with a time. Each is tied to the stoppage that begins nearest to it
   (from 3 seconds before to 20 seconds after). The table shows how the candidate handles the start
   of each kind of stoppage and what putting that kind exactly right would be worth, for the match
   and for one stoppage. The last lines scale the offside figure to a match with more offsides.

2. Would a hand-written rule add to what training learned: "the ball is seen outside the pitch, so
   the ball is dead, and the other team has it next"? Checked on the same degraded copies the
   candidate predicted on, where the ball is hidden 41% of the time and blurred by half a metre:
   how often the ball is seen outside, how often the truth and the candidate say dead there, what
   forcing dead does to the score, and how often "the other team has it next" is true.

The copies have no false ball detections and no ball that looks outside because it is in the air.
The competition's automatic tracking can have both, so part 2 shows the rule at its best.

    python3 experiments/stoppage_rules.py      # a few minutes
"""

import mmap
import os
import re
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from mtah import datasets, harness  # noqa: E402
from mtah import metrics as m  # noqa: E402
from mtah.config import DECODER  # noqa: E402
from mtah.decoder import decode_samples, expand  # noqa: E402
from mtah.degrade import DegradeConfig, load_stats  # noqa: E402
from mtah.match import A, D, H, X  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402
from v6_common import DRAWS, averaged, load  # noqa: E402
from whistle_prize import FPS, hear, predicted_start, stoppages  # noqa: E402

OUT = os.path.join("runs", "stoppage_rules.json")
BEFORE, AFTER = 3 * FPS, 20 * FPS      # how far from a logged event its stoppage may begin
MARGINS = (0.5, 1.0, 2.0)              # metres outside the lines
HOLD_CAP = 20 * FPS
KINDS = (("offside", "offside"), ("foul", "foul"), ("pitch", "other, ball on the pitch"), ("out", "ball goes out"))


def half_starts(path):
    """(first frame number, time of it) of the ball's track in each half."""
    out = {}
    with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        pattern = rb'<FrameSet GameSection="(\w+)"[^>]*TeamId="BALL"[^>]*>\s*<Frame N="(\d+)" T="([^"]+)"'
        for hit in re.finditer(pattern, data):
            out[hit.group(1).decode()] = (int(hit.group(2)), datetime.fromisoformat(hit.group(3).decode()))
    return out


def logged(path, starts, halves):
    """(frame, "offside" or "foul") for each offside and foul in the event log."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    second = starts["secondHalf"][1]
    out = []
    for attrs, tag in re.findall(r"<Event ([^>]*)>\s*<(Offside|Foul)\b", text):
        when = datetime.fromisoformat(re.search(r'EventTime="([^"]+)"', attrs).group(1))
        section, first = ("secondHalf", halves[1][0]) if when >= second else ("firstHalf", halves[0][0])
        out.append((first + int(round((when - starts[section][1]).total_seconds() * FPS)), tag.lower()))
    return out


def by_cause(stops, events):
    """The stoppages with "offside" or "foul" in place of where the ball was, where the log names a cause."""
    starts = np.array([s for s, _, _ in stops])
    kinds = [k for _, _, k in stops]
    gaps, lost = [], 0
    for cause in ("foul", "offside"):                      # an offside named last wins a shared stoppage
        for frame, tag in events:
            if tag != cause:
                continue
            near = np.flatnonzero((starts >= frame - BEFORE) & (starts <= frame + AFTER))
            if not len(near):
                lost += 1
                continue
            k = int(near[np.abs(starts[near] - frame).argmin()])
            kinds[k] = cause
            gaps.append((starts[k] - frame) / FPS)
    return [(s, e, k) for (s, e, _), k in zip(stops, kinds)], gaps, lost


def only(gt, pred, stops, kind):
    out = pred.copy()
    for s, e, k in stops:
        if k == kind:
            hear(out, pred, s, e, s, True)
    return out


def first_live(codes, start, stop, step=1):
    for t in range(start, stop, step):
        if codes[t] in (H, A):
            return int(codes[t])
    return None


if __name__ == "__main__":
    os.chdir(ROOT)
    record = manifest(before=BEFORE, after=AFTER, margins=list(MARGINS), hold_cap=HOLD_CAP)
    trees, tcn = averaged(DRAWS), load("net_tcn")
    ids = sorted(trees)
    german = [i for i in ids if i in datasets.DFL_IDS]
    stats = load_stats()
    truth, pred, stops, seen, pitch = {}, {}, {}, {}, {}
    gaps, lost = [], 0
    for i in ids:
        f = trees[i]
        halves = [tuple(int(v) for v in row) for row in f["halves"]]
        truth[i] = f["labels"]
        pred[i] = expand(decode_samples((f["prob"] + tcn[i]["prob"]) / 2.0, f["half"], DECODER), f["frames"], f["half"],
                         halves, len(f["labels"]))
        clean = datasets.load_open(i)
        stops[i] = stoppages(truth[i], clean.ball, clean.pitch)
        if i in german:
            events = logged(datasets._dfl_file("03_02_events_raw", i),
                            half_starts(datasets._dfl_file("04_03_positions_raw_observed", i)), halves)
            stops[i], g, n = by_cause(stops[i], events)
            gaps += g
            lost += n
        copy = harness.prepared(i, DegradeConfig(), harness.TEST_SEED_OFFSET, stats)
        if not np.array_equal(copy.labels, truth[i]):
            raise ValueError(f"{i}: this is not the copy the saved predictions were made on")
        seen[i], pitch[i] = copy.ball, copy.pitch
        del clean, copy
    base = {i: m.evaluate(truth[i], pred[i])["spell_f1"] for i in ids}
    results = {"causes": {}, "ball_out": {}}

    print("1. the seven German matches, by what stopped play")
    print(f"  logged fouls and offsides tied to a stoppage: {len(gaps)}, not tied: {lost}; the stoppage begins "
          f"{np.median(gaps):+.1f} s from the logged time (median)")
    print(f"  {'':<26}{'in all':>8}{'a match':>9}{'missed':>9}{'within 0.5 s':>14}{'late > 1 s':>12}{'late > 2 s':>12}"
          f"{'worth':>9}{'each':>10}")
    for kind, label in KINDS:
        rows = [(s, e, predicted_start(pred[i], s, e)) for i in german for s, e, k in stops[i] if k == kind]
        delays = np.array([(t - s) / FPS for s, e, t in rows if t is not None])
        gain = float(np.mean([m.evaluate(truth[i], only(truth[i], pred[i], stops[i], kind))["spell_f1"] - base[i]
                              for i in german]))
        cells = {"n": len(rows), "per_match": len(rows) / len(german),
                 "missed": float(np.mean([t is None for _, _, t in rows])),
                 "within_half_s": float((np.abs(delays) <= 0.5).sum() / len(rows)),
                 "late_1s": float((delays > 1.0).sum() / len(rows)), "late_2s": float((delays > 2.0).sum() / len(rows)),
                 "worth": gain, "worth_each": gain / (len(rows) / len(german))}
        results["causes"][kind] = cells
        print(f"  {label:<26}{cells['n']:>8}{cells['per_match']:>9.1f}{cells['missed']:>9.0%}{cells['within_half_s']:>14.0%}"
              f"{cells['late_1s']:>12.0%}{cells['late_2s']:>12.0%}{gain:>+9.4f}{cells['worth_each']:>+10.5f}")
    each = results["causes"]["offside"]["worth_each"]
    print("  a match with more offsides, every one put exactly right: "
          + ", ".join(f"{n} offsides {n * each:+.3f}" for n in (5, 8, 12)))

    print("\n2. the ball seen outside the pitch, on the copies the candidate predicted on (10 matches)")
    print(f"  {'':<34}{'share of frames':>16}{'truly dead':>12}{'we say dead':>13}{'forcing dead':>14}{'up':>7}")
    graded = {i: truth[i] != X for i in ids}

    def outside(i, margin):
        ball, (length, width) = seen[i], pitch[i]
        with np.errstate(invalid="ignore"):
            return graded[i] & ((np.abs(ball[:, 0]) > length / 2 + margin) | (np.abs(ball[:, 1]) > width / 2 + margin))

    for margin in MARGINS:
        masks = {i: outside(i, margin) for i in ids}
        gains = []
        for i in ids:
            forced = pred[i].copy()
            forced[masks[i]] = D
            gains.append(m.evaluate(truth[i], forced)["spell_f1"] - base[i])
        cells = {"share": float(np.mean([masks[i].sum() / graded[i].sum() for i in ids])),
                 "truly_dead": float(sum((truth[i][masks[i]] == D).sum() for i in ids) / sum(masks[i].sum() for i in ids)),
                 "we_say_dead": float(sum((pred[i][masks[i]] == D).sum() for i in ids) / sum(masks[i].sum() for i in ids)),
                 "gain": float(np.mean(gains)), "up": int(np.sum(np.array(gains) > 0))}
        results["ball_out"][f"margin_{margin}"] = cells
        print(f"  {'more than ' + str(margin) + ' m outside':<34}{cells['share']:>16.1%}{cells['truly_dead']:>12.1%}"
              f"{cells['we_say_dead']:>13.1%}{cells['gain']:>+14.4f}{cells['up']:>5}/{len(ids)}")

    gains = []
    for i in ids:
        ball, (length, width) = seen[i], pitch[i]
        with np.errstate(invalid="ignore"):
            inside = (np.abs(ball[:, 0]) < length / 2 - 0.5) & (np.abs(ball[:, 1]) < width / 2 - 0.5)
        forced, hold = pred[i].copy(), 0
        mask = outside(i, 1.0)
        for t in np.flatnonzero(graded[i]):
            if mask[t]:
                hold = HOLD_CAP
            elif inside[t] or truth[i][t] == X:
                hold = 0
            if hold:
                forced[t] = D
                hold -= 1
        gains.append(m.evaluate(truth[i], forced)["spell_f1"] - base[i])
    results["ball_out"]["hold_until_seen_inside"] = {"gain": float(np.mean(gains)), "up": int(np.sum(np.array(gains) > 0))}
    print(f"  {'1 m, and dead until seen inside':<34}{'':>16}{'':>12}{'':>13}{np.mean(gains):>+14.4f}"
          f"{int(np.sum(np.array(gains) > 0)):>5}/{len(ids)}")

    reach, switch, ours, rule_ours, n = [], [], [], [], 0
    for i in ids:
        for s, e, kind in stops[i]:
            if kind != "out" or truth[i][s - 1] not in (H, A) or e >= len(truth[i]) or truth[i][e] not in (H, A):
                continue
            n += 1
            before, after = int(truth[i][s - 1]), int(truth[i][e])
            reach.append(bool(outside(i, 0.5)[s:e].any()))
            switch.append(after != before)
            ours.append(first_live(pred[i], e, min(e + 5 * FPS, len(pred[i]))) == after)
            mine = first_live(pred[i], s - 1, max(s - 10 * FPS, 0), -1)
            rule_ours.append(mine is not None and (H if mine == A else A) == after)
    results["ball_out"]["after"] = {"n": n, "ball_seen_outside": float(np.mean(reach)), "other_team_next": float(np.mean(switch)),
                                    "candidate_right": float(np.mean(ours)), "rule_from_our_call": float(np.mean(rule_ours))}
    print(f"\n  stoppages that begin with the ball going out, with play on both sides: {n}")
    print(f"    the ball is seen outside at some point during them: {np.mean(reach):.0%}")
    print(f"    the other team has it next (the rule, knowing who truly had it before): {np.mean(switch):.0%}")
    print(f"    the rule, from who the candidate thought had it before: {np.mean(rule_ours):.0%}")
    print(f"    the candidate names the right team after the restart: {np.mean(ours):.0%}")
    record.update(matches=ids, results=results)
    save_manifest(OUT, record)
