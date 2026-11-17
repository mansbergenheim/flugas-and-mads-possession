"""What could hearing the referee's whistle be worth? Open data only, and no audio is involved.

A whistle can only help with stoppages that begin while the ball is still on the pitch (fouls,
offsides). When the ball goes out nobody whistles, and the tracking shows it anyway. This script
asks, on the 10 open matches held out, with trees + TCN at v5's decoder setting:

1. How many stoppages begin on the pitch, and how the candidate already handles their start
   (missed, late, on time), next to the stoppages that begin with the ball going out.
2. What a perfect stop-whistle would be worth: the start of every on-pitch stoppage put exactly
   right. Two versions bracket the truth. "Timing only" repairs the stoppages we do find. "And the
   missed ones" also creates those we miss, with the true end, which a whistle cannot give.
3. How fast that is lost with a weaker detector: one that finds only some whistles, hears them at
   the wrong moment, or hears whistles that are not the referee's. A false whistle is counted the
   crude way, as a forced two-second stoppage in the middle of play, so that row is a worst case.
4. What the other kind of whistle could be worth, the one that restarts play (kick-offs, penalties,
   free kicks the referee has managed): the end of every stoppage longer than 20 seconds put
   exactly right. That is a generous stand-in, since many long stoppages restart without a whistle.

5. The same simulated whistles given to the decoder as evidence instead of as a rule: each whistle
   adds a bonus to "dead" for the next 1.5 seconds and the decoder weighs it against the tracking.
   The bonus is tried at three strengths on the same ten matches, so the best row flatters itself.
   Restart whistles are left out here; a bonus after one of those would hold play dead too long.

A stoppage "begins on the pitch" when the ball stays inside the lines from just before its first
dead frame until a second after it. The open matches have no sound, so the whistles here are
simulated from the labels: this sizes the prize, it does not test a detector.

    python3 experiments/whistle_prize.py      # a few minutes
"""

import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from mtah import datasets  # noqa: E402
from mtah import metrics as m  # noqa: E402
from mtah.config import DECODER  # noqa: E402
from mtah.decoder import decode_samples, expand  # noqa: E402
from mtah.match import A, D, H  # noqa: E402
from mtah.provenance import manifest, save_manifest  # noqa: E402
from v6_common import DRAWS, averaged, load  # noqa: E402

FPS = 25
EDGE = 0.3            # metres: closer to a line than this counts as out
AFTER = 25            # frames after the first dead frame in which the ball must stay inside
LOOK = 50             # a predicted start up to this many frames early still belongs to the stoppage
FALSE_STOP = 50       # frames of forced stoppage for a whistle that was not the referee's
SEEDS = (0, 1, 2)
EVIDENCE_LASTS = 38   # frames a whistle's bonus lasts when it is given to the decoder as evidence
OUT = os.path.join("runs", "whistle_prize.json")


def stoppages(gt, ball, pitch):
    """(first dead frame, frame after the last, "pitch" or "out") for each stoppage that begins during play."""
    half_l, half_w = pitch[0] / 2 - EDGE, pitch[1] / 2 - EDGE
    out = []
    for a, b in m.valid_runs(gt):
        for s in np.flatnonzero((gt[a + 1:b] == D) & (gt[a:b - 1] != D)) + a + 1:
            e = int(s)
            while e < b and gt[e] == D:
                e += 1
            near = ball[max(s - 5, a):min(s + AFTER, b)]
            near = near[np.isfinite(near[:, 0])]
            inside = len(near) and (np.abs(near[:, 0]) < half_l).all() and (np.abs(near[:, 1]) < half_w).all()
            out.append((int(s), e, "pitch" if inside else "out"))
    return out


def predicted_start(pred, s, e):
    """First frame of the predicted stoppage that covers part of the true one; None if we miss it."""
    hit = np.flatnonzero(pred[s:e] == D)
    if not len(hit):
        return None
    t = s + int(hit[0])
    while t > 0 and pred[t - 1] == D:
        t -= 1
    return t


def hear(out, pred, s, e, w, create):
    """Put the start of one stoppage at frame w, the moment the whistle is heard."""
    t = predicted_start(pred, s, e)
    if t is None:
        if create and w < e:
            out[w:e] = D
    elif t > w:
        out[w:t] = D
    elif s - LOOK <= t < w and t > 0 and pred[t - 1] in (H, A):
        out[t:w] = pred[t - 1]


def heard(gt, pred, stops, recall=1.0, jitter=0.0, false=0, seed=0):
    """Frames at which a simulated detector reports a whistle: on-pitch stoppage starts, and false ones in play."""
    rng = np.random.default_rng(seed)
    out = []
    for s, _, kind in stops:
        if kind == "pitch" and rng.random() < recall:
            out.append(max(s + (int(round(rng.normal(0.0, jitter * FPS))) if jitter else 0), 1))
    if false:
        starts = np.array([s for s, _, _ in stops])
        live = np.flatnonzero(((gt == H) | (gt == A)) & (pred != D))
        order = np.searchsorted(starts, live)
        gap = np.minimum(np.abs(live - starts[np.clip(order, 0, len(starts) - 1)]),
                         np.abs(live - starts[np.clip(order - 1, 0, len(starts) - 1)]))
        live = live[gap > 3 * FPS]
        out += [int(t) for t in rng.choice(live, size=min(false, len(live)), replace=False)]
    return out


def with_restarts(pred, stops, longer_than):
    """Put the end of each stoppage we find, and that lasts longer than so many frames, exactly right."""
    out = pred.copy()
    for k, (s, e, _) in enumerate(stops):
        hit = np.flatnonzero(pred[s:e] == D)
        if e - s <= longer_than or not len(hit):
            continue
        t = s + int(hit[-1]) + 1
        while t < len(pred) and pred[t] == D:
            t += 1
        limit = stops[k + 1][0] if k + 1 < len(stops) else len(pred)
        if t < e:
            out[t:e] = D
        elif t > e and t < len(pred) and pred[t] in (H, A):
            out[e:min(t, limit)] = pred[t]
    return out


def with_whistles(gt, pred, stops, kinds, create, recall=1.0, jitter=0.0, false=0, seed=0):
    rng = np.random.default_rng(seed)
    out = pred.copy()
    for s, e, kind in stops:
        if kind not in kinds or rng.random() >= recall:
            continue
        w = s + (int(round(rng.normal(0.0, jitter * FPS))) if jitter else 0)
        hear(out, pred, s, e, max(w, 1), create)
    if false:
        starts = np.array([s for s, _, _ in stops])
        live = np.flatnonzero(((gt == H) | (gt == A)) & (pred != D))
        if len(starts):
            gap = np.abs(live[:, None] - starts[None, :]).min(1) if len(live) * len(starts) < 5e7 else None
            if gap is None:
                order = np.searchsorted(starts, live)
                gap = np.minimum(np.abs(live - starts[np.clip(order, 0, len(starts) - 1)]),
                                 np.abs(live - starts[np.clip(order - 1, 0, len(starts) - 1)]))
            live = live[gap > 3 * FPS]
        for t in rng.choice(live, size=min(false, len(live)), replace=False):
            stop = int(t) + FALSE_STOP
            out[int(t):stop][gt[int(t):stop] != D] = D
    return out


if __name__ == "__main__":
    os.chdir(ROOT)
    record = manifest(edge=EDGE, after=AFTER, look=LOOK, false_stop=FALSE_STOP, seeds=list(SEEDS))
    trees, tcn = averaged(DRAWS), load("net_tcn")
    ids = sorted(trees)
    german = [i for i in ids if i in datasets.DFL_IDS]
    truth, pred, stops, probs, spans = {}, {}, {}, {}, {}
    for i in ids:
        f = trees[i]
        prob = (f["prob"] + tcn[i]["prob"]) / 2.0
        halves = [tuple(int(v) for v in row) for row in f["halves"]]
        truth[i] = f["labels"]
        probs[i], spans[i] = prob, halves
        pred[i] = expand(decode_samples(prob, f["half"], DECODER), f["frames"], f["half"], halves, len(f["labels"]))
        match = datasets.load_open(i)
        if match.n_frames != len(truth[i]):
            raise ValueError(f"{i}: the saved labels and the match have different lengths")
        stops[i] = stoppages(truth[i], match.ball, match.pitch)

    print("1. stoppages that begin during play, and how trees + TCN handles their start")
    print(f"  {'':<26}{'a match':>9}{'share':>8}{'missed':>9}{'within 0.5 s':>14}{'late > 1 s':>12}{'late > 2 s':>12}"
          f"{'median late':>13}")
    results = {"starts": {}, "worth": {}, "weaker": {}}
    for kind, label in (("pitch", "begin on the pitch"), ("out", "ball goes out")):
        rows = [(s, e, predicted_start(pred[i], s, e)) for i in ids for s, e, k in stops[i] if k == kind]
        total = sum(len(stops[i]) for i in ids)
        delays = np.array([(t - s) / FPS for s, e, t in rows if t is not None])
        cells = {"per_match": len(rows) / len(ids), "share": len(rows) / total,
                 "missed": float(np.mean([t is None for _, _, t in rows])),
                 "within_half_s": float((np.abs(delays) <= 0.5).sum() / len(rows)),
                 "late_1s": float((delays > 1.0).sum() / len(rows)), "late_2s": float((delays > 2.0).sum() / len(rows)),
                 "median_late_s": float(np.median(delays))}
        results["starts"][kind] = cells
        print(f"  {label:<26}{cells['per_match']:>9.0f}{cells['share']:>8.0%}{cells['missed']:>9.0%}"
              f"{cells['within_half_s']:>14.0%}{cells['late_1s']:>12.0%}{cells['late_2s']:>12.0%}"
              f"{cells['median_late_s']:>11.1f} s")
    logged = []
    for i in german:
        with open(datasets._dfl_file("03_02_events_raw", i), encoding="utf-8") as handle:
            text = handle.read()
        logged.append(len(re.findall(r"<Foul ", text)) + len(re.findall(r"<Offside ", text)))
    ours = [sum(k == "pitch" for _, _, k in stops[i]) for i in german]
    results["starts"]["german_check"] = {"fouls_and_offsides_logged": logged, "begin_on_pitch": ours}
    print(f"  check on the German matches: fouls and offsides in the event logs {sum(logged) / len(logged):.0f} a match, "
          f"stoppages we class as beginning on the pitch {sum(ours) / len(ours):.0f}")

    def score(**how):
        per_seed = []
        for seed in SEEDS if (how.get("recall", 1.0) < 1.0 or how.get("jitter") or how.get("false")) else (0,):
            per_seed.append([m.evaluate(truth[i], with_whistles(truth[i], pred[i], stops[i], seed=seed, **how))["spell_f1"]
                             for i in ids])
        return np.mean(per_seed, axis=0)

    base = np.array([m.evaluate(truth[i], pred[i])["spell_f1"] for i in ids])
    is_german = np.array([i in german for i in ids])

    def line(label, values, store):
        gain = values - base
        store[label.strip()] = {"spell_f1": float(values.mean()), "gain": float(gain.mean()),
                                "matches_up": int((gain > 0).sum()), "gain_german": float(gain[is_german].mean()),
                                "gain_metrica": float(gain[~is_german].mean())}
        print(f"  {label:<58}{values.mean():>8.3f}{gain.mean():>+9.3f}{int((gain > 0).sum()):>6}/{len(ids)}"
              f"{gain[is_german].mean():>+10.3f}{gain[~is_german].mean():>+10.3f}", flush=True)

    head = f"  {'':<58}{'spellF1':>8}{'gain':>9}{'up':>9}{'German':>10}{'Metrica':>10}"
    print(f"\n2. what a perfect whistle would be worth (trees + TCN as predicted: {base.mean():.3f})")
    print(head)
    line("on-pitch stoppages: timing only", score(kinds=("pitch",), create=False), results["worth"])
    line("on-pitch stoppages: and the missed ones", score(kinds=("pitch",), create=True), results["worth"])
    line("for comparison, ball-out stoppages: timing only", score(kinds=("out",), create=False), results["worth"])
    line("for comparison, ball-out stoppages: and the missed ones", score(kinds=("out",), create=True), results["worth"])
    line("every stoppage start: and the missed ones", score(kinds=("pitch", "out"), create=True), results["worth"])

    print("\n3. with a weaker detector (on-pitch stoppages, and the missed ones)")
    print(head)
    for recall in (0.75, 0.5, 0.25):
        line(f"finds {recall:.0%} of the whistles", score(kinds=("pitch",), create=True, recall=recall), results["weaker"])
    for jitter in (0.25, 0.5, 1.0):
        line(f"hears them {jitter} s off (standard deviation)", score(kinds=("pitch",), create=True, jitter=jitter),
             results["weaker"])
    for false in (5, 10, 20, 40):
        line(f"no true whistles, {false} false ones a match", score(kinds=(), create=False, false=false), results["weaker"])
    line("finds 60%, 0.3 s off, 10 false ones a match",
         score(kinds=("pitch",), create=True, recall=0.6, jitter=0.3, false=10), results["weaker"])
    line("finds 80%, 0.2 s off, 3 false ones a match",
         score(kinds=("pitch",), create=True, recall=0.8, jitter=0.2, false=3), results["weaker"])
    print("\n4. the whistle that restarts play (the end of a stoppage put exactly right)")
    print(head)
    results["restart"] = {}
    long = np.mean([sum(e - s > 20 * FPS for s, e, _ in stops[i]) for i in ids])
    results["restart"]["stoppages_over_20s_per_match"] = float(long)
    for label, frames in ((f"stoppages longer than 20 s ({long:.0f} a match)", 20 * FPS), ("every stoppage", 0)):
        values = np.array([m.evaluate(truth[i], with_restarts(pred[i], stops[i], frames))["spell_f1"] for i in ids])
        line(label, values, results["restart"])
    print(f"\n5. the same whistles as evidence for the decoder (a bonus to dead for {EVIDENCE_LASTS / FPS:.1f} s)")
    print(head)
    results["evidence"] = {}

    def decoded(i, whistles, bonus):
        f = trees[i]
        prior = np.zeros_like(probs[i])
        for w in whistles:
            prior[(f["frames"] >= w) & (f["frames"] < w + EVIDENCE_LASTS), D] += bonus
        return expand(decode_samples(probs[i], f["half"], DECODER, prior), f["frames"], f["half"], spans[i],
                      len(f["labels"]))

    detectors = (("perfect whistles", dict()), ("no true whistles, 10 false ones a match", dict(recall=0.0, false=10)),
                 ("finds 60%, 0.3 s off, 10 false ones a match", dict(recall=0.6, jitter=0.3, false=10)),
                 ("finds 80%, 0.2 s off, 3 false ones a match", dict(recall=0.8, jitter=0.2, false=3)))
    for bonus in (1.0, 2.0, 4.0):
        for label, how in detectors:
            seeds = SEEDS if how else (0,)
            values = np.mean([[m.evaluate(truth[i], decoded(i, heard(truth[i], pred[i], stops[i], seed=seed, **how),
                                                            bonus))["spell_f1"] for i in ids] for seed in seeds], axis=0)
            line(f"bonus {bonus:.0f}: {label}", values, results["evidence"])
    record.update(matches=ids, base=float(base.mean()), results=results)
    save_manifest(OUT, record)
