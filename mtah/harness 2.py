"""Leave-one-match-out validation on open matches. This is how versions get chosen.

For each held-out match a fresh predictor is trained on the other matches, then predicts on a
degraded copy of the held-out match, which has its labels removed so a predictor cannot peek. Train
and test use different degradation seeds.

Never pick a version from the competition's public 5 minutes. A gap under the noise band here is a
tie too, whatever the sign.
"""

import dataclasses
import json
import os
import time
import zlib
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from . import datasets as ds
from . import metrics as m
from .degrade import DegradeConfig, degrade, load_stats, random_config
from .labels import controlled
from .match import Match
from .provenance import manifest

TEST_SEED_OFFSET = 1000
# Fold DFL's touch-level flips. Justified in research/STRATEGY_PHASE1.md and scripts/inspect_labels.py.
DFL_MIN_CONTROL_S = 1.0


def harmonise(match: Match, min_control_s: float = DFL_MIN_CONTROL_S) -> Match:
    """Competition-style labels. Metrica's are derived from events and need no change."""
    if match.source != "dfl" or not min_control_s:
        return match
    return dataclasses.replace(match, labels=controlled(match.labels, min_control_s, match.fps))


def prepared(match_id: str, cfg: Optional[DegradeConfig], seed: int, stats: dict) -> Match:
    match = harmonise(ds.load_open(match_id))
    return degrade(match, cfg, seed, stats) if cfg is not None else match


def training_matches(match_ids: Sequence[str], seed: int, copies: int, stats: dict):
    """Yield each match several times at randomly drawn tracking quality. One in memory at a time."""
    for match_id in match_ids:
        clean = harmonise(ds.load_open(match_id))
        for copy in range(copies):
            rng = np.random.default_rng([seed, copy, zlib.crc32(match_id.encode())])
            yield degrade(clean, random_config(rng), seed * 100 + copy, stats)


def without_labels(match: Match) -> Match:
    return dataclasses.replace(match, labels=None)


def run_lomo(make_predictor: Callable[[], object], match_ids: Sequence[str],
             cfg: Optional[DegradeConfig] = DegradeConfig(), seed: int = 0,
             bootstrap: int = 0, needs_training: bool = True) -> List[Dict]:
    """One result row per held-out match."""
    stats = load_stats()
    rows = []
    for held_out in match_ids:
        predictor = make_predictor()
        if needs_training:
            train = [prepared(i, cfg, seed, stats) for i in match_ids if i != held_out]
            predictor.fit(train)
            del train
        else:
            predictor.fit([])
        test = prepared(held_out, cfg, seed + TEST_SEED_OFFSET, stats)
        pred = predictor.predict(without_labels(test))
        if pred.shape != test.labels.shape:
            raise ValueError(f"{predictor.name}: predicted {pred.shape}, expected {test.labels.shape}")
        rows.append({"match": held_out, "name": predictor.name,
                     **m.evaluate(test.labels, pred, bootstrap)})
    return rows


def summarise(rows: Iterable[Dict]) -> Dict:
    rows = list(rows)
    spell = np.array([r["spell_f1"] for r in rows])
    return {"name": rows[0]["name"], "matches": len(rows),
            "spell_f1": float(spell.mean()), "spell_f1_sd": float(spell.std(ddof=1)) if len(rows) > 1 else 0.0,
            "macro_f1": float(np.mean([r["macro_f1"] for r in rows])),
            "poss_ratio": float(np.mean([r["n_pred_spells"] / max(r["n_gt_spells"], 1) for r in rows])),
            "transition_f1_25": float(np.mean([r["transition_f1_25"] for r in rows]))}


def compare(a_rows: List[Dict], b_rows: List[Dict]) -> Dict:
    """Paired comparison of B against A on the same matches."""
    a = {r["match"]: r["spell_f1"] for r in a_rows}
    b = {r["match"]: r["spell_f1"] for r in b_rows}
    shared = sorted(set(a) & set(b))
    diff = np.array([b[k] - a[k] for k in shared])
    mean = float(diff.mean())
    return {"mean_gain": mean, "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()),
            "matches": len(shared),
            "verdict": "real" if abs(mean) >= m.SPELL_NOISE_BAND and
                       ((diff > 0).sum() >= 0.75 * len(diff) or (diff < 0).sum() >= 0.75 * len(diff))
                       else "within noise"}


def save_run(rows: List[Dict], tag: str, folder: str = "runs") -> str:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}.json")
    with open(path, "w") as handle:
        json.dump({"manifest": manifest(), "rows": rows}, handle, indent=1, default=float)
    return path
