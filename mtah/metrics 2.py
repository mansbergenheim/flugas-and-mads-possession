"""Our own implementation of the competition metrics, written from EVALUATION.md.

evaluation/possession_metrics.py in the organisers' repo is the authority. This copy exists so the
validation harness can score other matches with no dependency on their code. tests/test_metrics.py
pins it to the organisers' published and saved results, and tests/test_official_parity.py compares
the two directly when you run it on your machine.

Rules reproduced here, because they decide the score:
  - X on a graded frame is a miss for macro-F1.
  - Spell metrics see X forward-filled inside each half, back-filled at the head of a half.
  - A possession is a maximal run of H or of A inside a graded run. D is a stoppage, never a spell.
  - Spells match one-to-one, same team only, greedily by IoU, with IoU over the spans' hull.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .match import A, D, H, X

SPELL_IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)
TAUS = (12, 25)
SPELL_NOISE_BAND = 0.04   # organisers' figure for a full match
MACRO_NOISE_BAND = 0.01

Span = Tuple[int, int]
Segment = Tuple[int, int, int]


def valid_runs(gt: np.ndarray) -> List[Span]:
    """Half-open spans where the ground truth is not X. Temporal metrics never cross one."""
    edges = np.diff(np.concatenate(([0], (gt != X).astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def fill_abstentions(pred: np.ndarray, runs: Sequence[Span]) -> np.ndarray:
    filled = pred.copy()
    for start, stop in runs:
        window = filled[start:stop]
        known = np.flatnonzero(window != X)
        if not len(known):
            continue
        carry = np.maximum.accumulate(np.where(window != X, np.arange(len(window)), -1))
        carry[carry < 0] = known[0]
        filled[start:stop] = window[carry]
    return filled


def segments(codes: np.ndarray, runs: Sequence[Span]) -> List[Segment]:
    out = []
    for start, stop in runs:
        window = codes[start:stop]
        if not len(window):
            continue
        bounds = np.concatenate(([0], np.flatnonzero(window[1:] != window[:-1]) + 1, [len(window)]))
        out.extend((start + int(bounds[i]), start + int(bounds[i + 1]), int(window[bounds[i]]))
                   for i in range(len(bounds) - 1))
    return out


def possession_spells(codes: np.ndarray, runs: Sequence[Span]) -> List[Segment]:
    return [s for s in segments(codes, runs) if s[2] in (H, A)]


def _overlaps(gt_spells: List[Segment], pred_spells: List[Segment]) -> List[Tuple[float, int, int]]:
    """Same-team overlapping pairs, IoU descending. Both lists are time-sorted, so one sweep."""
    out, first = [], 0
    for i, (g0, g1, gk) in enumerate(gt_spells):
        while first < len(pred_spells) and pred_spells[first][1] <= g0:
            first += 1
        j = first
        while j < len(pred_spells) and pred_spells[j][0] < g1:
            p0, p1, pk = pred_spells[j]
            if pk == gk and p1 > g0:
                out.append(((min(g1, p1) - max(g0, p0)) / float(max(g1, p1) - min(g0, p0)), i, j))
            j += 1
    out.sort(key=lambda t: (-t[0], t[1], t[2]))
    return out


def _matched(overlaps, n_gt: int, threshold: float) -> np.ndarray:
    flags = np.zeros(n_gt, dtype=bool)
    used = set()
    for iou, i, j in overlaps:
        if iou < threshold:
            break
        if flags[i] or j in used:
            continue
        flags[i] = True
        used.add(j)
    return flags


def spell_scores(gt: np.ndarray, pred_filled: np.ndarray, runs: Sequence[Span]) -> Dict:
    gs, ps = possession_spells(gt, runs), possession_spells(pred_filled, runs)
    ov = _overlaps(gs, ps)
    per_iou, f1s = {}, []
    for t in SPELL_IOUS:
        tp = int(_matched(ov, len(gs), t).sum())
        fp, fn = len(ps) - tp, len(gs) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_iou[t] = {"precision": p, "recall": r, "f1": f1}
        f1s.append(f1)
    return {"per_iou": per_iou, "mean_f1": float(np.mean(f1s)), "n_gt": len(gs), "n_pred": len(ps)}


def confusion3(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """3x4 counts. Rows are GT H/A/D, columns prediction H/A/D/X. GT X never appears."""
    cm = np.bincount(gt.astype(np.int64) * 4 + pred.astype(np.int64), minlength=16).reshape(4, 4)
    return cm[:3, :]


def macro_f1(cm3: np.ndarray) -> float:
    f1s = []
    for k in range(3):
        tp = float(cm3[k, k])
        fp = float(cm3[:, k].sum()) - tp
        fn = float(cm3[k, :].sum()) - tp
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s))


def transitions(codes: np.ndarray, runs: Sequence[Span]) -> List[Segment]:
    out = []
    for start, stop in runs:
        w = codes[start:stop]
        out.extend((start + int(i), int(w[i - 1]), int(w[i]))
                   for i in (np.flatnonzero(w[1:] != w[:-1]) + 1).tolist())
    return out


def transition_f1(gt_t, pred_t, tol: int) -> Dict:
    """One-to-one within +/- tol frames, matched on destination label."""
    matched, latency = 0, []
    for label in (H, A, D):
        g = [t[0] for t in gt_t if t[2] == label]
        p = [t[0] for t in pred_t if t[2] == label]
        i = j = 0
        while i < len(g) and j < len(p):
            delta = p[j] - g[i]
            if delta < -tol:
                j += 1
            elif delta > tol:
                i += 1
            else:
                matched += 1
                latency.append(delta)
                i += 1
                j += 1
    prec = matched / len(pred_t) if pred_t else 0.0
    rec = matched / len(gt_t) if gt_t else 0.0
    return {"f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "precision": prec,
            "recall": rec, "mean_latency": float(np.mean(latency)) if latency else None}


def bootstrap_spell(gt: np.ndarray, pred_filled: np.ndarray, runs, n_draws: int = 1000,
                    seed: int = 0) -> Tuple[float, float]:
    """5-95% interval resampling whole possessions, the way the organisers do it."""
    gs, ps = possession_spells(gt, runs), possession_spells(pred_filled, runs)
    if not gs:
        return 0.0, 0.0
    ov = _overlaps(gs, ps)
    flags = np.vstack([_matched(ov, len(gs), t) for t in SPELL_IOUS])
    rng = np.random.RandomState(seed)
    scores = np.empty(n_draws)
    n_gt, n_pred = len(gs), len(ps)
    for d in range(n_draws):
        picks = rng.randint(0, n_gt, n_gt)
        f1s = []
        for row in flags:
            tp = int(row[picks].sum())
            fp = max(0, n_pred - tp)
            f1s.append(2.0 * tp / (2 * tp + fp + n_gt - tp) if tp else 0.0)
        scores[d] = np.mean(f1s)
    low, high = np.percentile(scores, (5, 95))
    return float(low), float(high)


def evaluate(gt: np.ndarray, pred: np.ndarray, bootstrap: int = 0) -> Dict:
    """Every number the harness reports for one match. Only GT-graded frames are scored."""
    if len(gt) != len(pred):
        raise ValueError(f"length mismatch: gt {len(gt)} vs pred {len(pred)}")
    runs = valid_runs(gt)
    cm3 = confusion3(gt, pred)
    filled = fill_abstentions(pred, runs)
    spells = spell_scores(gt, filled, runs)
    gt_t, pred_t = transitions(gt, runs), transitions(filled, runs)
    seg_pred = possession_spells(filled, runs)
    seg_gt = possession_spells(gt, runs)
    alive_tp = float(cm3[:2, :2].sum())
    alive_fp = float(cm3[2, :2].sum())
    alive_fn = float(cm3[:2, 2:].sum())
    result = {
        "spell_f1": spells["mean_f1"],
        "macro_f1": macro_f1(cm3),
        "spell_per_iou": {t: v["f1"] for t, v in spells["per_iou"].items()},
        "n_gt_spells": spells["n_gt"],
        "n_pred_spells": spells["n_pred"],
        "mean_gt_spell_s": float(np.mean([b - a for a, b, _ in seg_gt])) / 25 if seg_gt else 0.0,
        "mean_pred_spell_s": float(np.mean([b - a for a, b, _ in seg_pred])) / 25 if seg_pred else 0.0,
        "alive_dead_f1": 2 * alive_tp / (2 * alive_tp + alive_fp + alive_fn) if alive_tp else 0.0,
        "team_acc": float(cm3[H, H] + cm3[A, A]) / max(float(cm3[:2, :2].sum()), 1.0),
        "n_graded": int(cm3.sum()),
        "n_abstain_graded": int(cm3[:, X].sum()),
    }
    for tol in TAUS:
        result[f"transition_f1_{tol}"] = transition_f1(gt_t, pred_t, tol)["f1"]
    result["latency_25"] = transition_f1(gt_t, pred_t, 25)["mean_latency"]
    if bootstrap:
        result["spell_ci"] = bootstrap_spell(gt, filled, runs, bootstrap)
    return result
