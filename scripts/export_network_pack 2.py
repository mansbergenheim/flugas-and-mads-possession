"""Write the open matches as files a sequence network can train on. Open data only.

The networks (colab/) see exactly what the trees see: the same matches, the same damaged copies
(three draws of three copies each, as harness.training_matches makes them), the same 198 inputs
(features.build_features(extra=True)) at the same moments, the same labels and sample weights, and
the same held-out test copies. Only the order in time is new to them.

Files, under runs/network_pack:
  meta.json            input names, how the inputs change when the teams are swapped, the matches
  train_<match>.npz    x0..x8 (nine damaged copies, float32 [n, 198]), y, w, half, frames
  test_<match>.npz     x, y, w, half, frames: the standard test copy every held-out score uses
  sc_<match>.npz       the same for SkillCorner's working half (the reserve half is never loaded)
  stress_<match>.npz   with --stress: x0..x18, one per stress condition, for the three stress matches

Nothing here reads the competition match: only the open loaders are used.

    python3 scripts/export_network_pack.py            # about 7 minutes, 1.2 GB
    python3 scripts/export_network_pack.py --stress   # adds the stress copies, 0.6 GB

MTAH_PACK_NAME=<name> writes to runs/<name> instead, and --only takes a few matches: for dry runs.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from mtah import datasets as ds, harness  # noqa: E402
from mtah.config import SPELL_WEIGHT, TRAIN_COPIES  # noqa: E402
from mtah.degrade import DegradeConfig, degrade, load_stats  # noqa: E402
from mtah.features import build_features, swap_teams  # noqa: E402
from mtah.match import A, D, H, X  # noqa: E402
from mtah.model import spell_lengths  # noqa: E402
from mtah.provenance import manifest  # noqa: E402
from stress_test import BASE, CONDITIONS, HELD_OUT, blank_ball  # noqa: E402

DRAWS = (0, 1, 2)
OUT = os.path.join("runs", os.environ.get("MTAH_PACK_NAME", "network_pack"))
SWAP_TOLERANCE = 1e-4


def inputs(match):
    values, names, frames, half = build_features(match, extra=True)
    return values, names, frames.astype(np.int32), half.astype(np.int8)


def targets(match, frames):
    """The label at each sampled moment and the weight the trees give it. X (not graded) weighs nothing."""
    y = match.labels[frames].astype(np.int8)
    w = ((spell_lengths(match.labels)[frames] / match.fps).clip(1.0, 60.0) ** -SPELL_WEIGHT).astype(np.float32)
    w[y == X] = 0.0
    return y, w


def swap_plan(x, swapped):
    """Where each input comes from when home and away are exchanged: a source column, a sign, an offset.

    Tried in an order that keeps a column where it is whenever that is exact.
    """
    nan, nan_swapped = np.isnan(x), np.isnan(swapped)
    source, sign, offset, worst = [], [], [], 0.0
    for j in range(x.shape[1]):
        best = None
        for k in [j] + [k for k in range(x.shape[1]) if k != j]:
            if not np.array_equal(nan[:, k], nan_swapped[:, j]):
                continue
            for s, o in ((1.0, 0.0), (-1.0, 0.0), (-1.0, 1.0)):
                gap = float(np.nanmax(np.abs(s * x[:, k] + o - swapped[:, j]))) if (~nan_swapped[:, j]).any() else 0.0
                if best is None or gap < best[0]:
                    best = (gap, k, s, o)
            if best is not None and best[0] == 0.0:
                break
        if best is None or best[0] > SWAP_TOLERANCE:
            raise ValueError(f"input {j} has no counterpart when the teams are swapped")
        worst = max(worst, best[0])
        source.append(int(best[1]))
        sign.append(float(best[2]))
        offset.append(float(best[3]))
    return {"source": source, "sign": sign, "offset": offset, "largest_gap": worst}


def apply_swap(x, plan):
    return (x[:, plan["source"]] * np.asarray(plan["sign"], dtype=np.float32)
            + np.asarray(plan["offset"], dtype=np.float32)).astype(np.float32)


def swap_gap(x, swapped, plan):
    mine = apply_swap(x, plan)
    if not np.array_equal(np.isnan(mine), np.isnan(swapped)):
        return float("inf")
    return float(np.nanmax(np.abs(mine - swapped)))


def save(name, **arrays):
    path = os.path.join(OUT, name)
    np.savez_compressed(path, **arrays)
    return path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stress", action="store_true", help="write the stress copies only (after a normal run)")
    parser.add_argument("--only", nargs="*", help="open matches to write, to test the script")
    args = parser.parse_args()
    os.chdir(ROOT)
    os.makedirs(OUT, exist_ok=True)
    stats = load_stats()
    ids = list(args.only or ds.available())
    meta_path = os.path.join(OUT, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as handle:
            meta = json.load(handle)
    started = time.time()
    samples, gaps, skillcorner, stress = {}, [], [], []

    if not args.stress:
        # How the inputs move when the teams are swapped, worked out on one match from each source.
        ends = [harness.prepared(i, DegradeConfig(), harness.TEST_SEED_OFFSET, stats) for i in (ids[0], ids[-1])]
        plan = swap_plan(np.vstack([inputs(m)[0] for m in ends]), np.vstack([inputs(swap_teams(m))[0] for m in ends]))
        del ends
        names = None
        for match_id in ids:
            match = harness.prepared(match_id, DegradeConfig(), harness.TEST_SEED_OFFSET, stats)
            x, got, frames, half = inputs(match)
            names = names or got
            if got != names:
                raise ValueError(f"{match_id}: the inputs differ from the first match's")
            gaps.append(swap_gap(x, inputs(swap_teams(match))[0], plan))
            earlier = os.path.join("runs", "oof", "v5", f"{match_id}.npz")
            if os.path.exists(earlier):
                with np.load(earlier, allow_pickle=False) as z:
                    if not (np.array_equal(z["frames"], frames) and np.array_equal(z["half"], half)):
                        raise ValueError(f"{match_id}: not sampled at the moments the trees' held-out predictions use")
            y, w = targets(match, frames)
            save(f"test_{match_id}.npz", x=x, y=y, w=w, half=half, frames=frames)
            samples[match_id] = int(len(frames))

            copies = {}
            for draw in DRAWS:
                for copy, damaged in enumerate(harness.training_matches([match_id], draw, TRAIN_COPIES, stats)):
                    xc, got, frames_c, _ = inputs(damaged)
                    if got != names or not np.array_equal(frames_c, frames):
                        raise ValueError(f"{match_id}: a training copy is not sampled like the test copy")
                    copies[f"x{draw * TRAIN_COPIES + copy}"] = xc
            save(f"train_{match_id}.npz", y=y, w=w, half=half, frames=frames, **copies)
            print(f"{match_id}: {len(frames)} moments, {len(copies)} training copies, swap gap {gaps[-1]:.1e} "
                  f"({time.time() - started:.0f}s)", flush=True)

        skillcorner = [] if args.only else list(ds.SKILLCORNER_DEV)
        for match_id in skillcorner:
            match = harness.prepared(match_id, DegradeConfig(), harness.TEST_SEED_OFFSET, stats)
            x, got, frames, half = inputs(match)
            if got != names:
                raise ValueError(f"{match_id}: the inputs differ from the open matches'")
            gaps.append(swap_gap(x, inputs(swap_teams(match))[0], plan))
            y, w = targets(match, frames)
            save(f"sc_{match_id}.npz", x=x, y=y, w=w, half=half, frames=frames)
            samples[match_id] = int(len(frames))
            print(f"{match_id}: {len(frames)} moments at {match.fps} frames a second, swap gap {gaps[-1]:.1e} "
                  f"({time.time() - started:.0f}s)", flush=True)
        if max(gaps) > SWAP_TOLERANCE:
            raise ValueError(f"the swap rule is off by {max(gaps):.1e} on some copy")
        meta.update(names=names, swap={**plan, "largest_gap": max(gaps)},
                    open_matches=sorted(set(meta.get("open_matches", [])) | set(ids)),
                    skillcorner=sorted(set(meta.get("skillcorner", [])) | set(skillcorner)))
    else:
        if "names" not in meta:
            sys.exit("run this once without --stress first")
        stress = [h for h in HELD_OUT if h in ids]
        for held in stress:
            clean = harness.harmonise(ds.load_open(held))
            copies, shared = {}, None
            for index, (name, cfg) in enumerate(CONDITIONS.items()):
                test = degrade(clean, BASE, 1000, stats) if cfg == "no_ball" else degrade(clean, cfg, 1000, stats)
                x, got, frames, half = inputs(blank_ball(test) if cfg == "no_ball" else test)
                if got != meta["names"]:
                    raise ValueError(f"{held}, {name}: the inputs differ from the open matches'")
                earlier = os.path.join("runs", "stress_v6_parts", name.replace(" ", "_").replace("%", "pct") + f"__{held}.npz")
                if os.path.exists(earlier):
                    with np.load(earlier, allow_pickle=False) as z:
                        if not np.array_equal(z["frames"], frames):
                            raise ValueError(f"{held}, {name}: not sampled like the trees' stress predictions")
                if shared is None:
                    shared = dict(zip(("y", "w"), targets(test, frames)), half=half, frames=frames)
                elif not np.array_equal(shared["frames"], frames):
                    raise ValueError(f"{held}, {name}: not sampled like the other conditions")
                copies[f"x{index}"] = x
            save(f"stress_{held}.npz", **shared, **copies)
            samples[held + " (stress)"] = int(len(shared["frames"]))
            print(f"stress copies of {held}: {len(copies)} conditions ({time.time() - started:.0f}s)", flush=True)
        meta.update(stress_held_out=sorted(set(meta.get("stress_held_out", [])) | set(stress)),
                    stress_conditions=list(CONDITIONS))

    files = {}
    for name in sorted(os.listdir(OUT)):
        if name.endswith(".npz"):
            path = os.path.join(OUT, name)
            files[name] = {"bytes": os.path.getsize(path), "sha256": sha256(path)}
    meta.update(manifest=manifest(draws=list(DRAWS), copies=TRAIN_COPIES, spell_weight=SPELL_WEIGHT),
                labels={"H": H, "A": A, "D": D, "X": X}, dfl=list(ds.DFL_IDS), metrica=list(ds.METRICA_IDS),
                copies_per_match=len(DRAWS) * TRAIN_COPIES, samples={**meta.get("samples", {}), **samples}, files=files)
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=1, default=str)
    total = sum(f["bytes"] for f in files.values())
    print(f"\n{len(files)} files, {total / 1e9:.2f} GB, under {OUT}. Done in {time.time() - started:.0f}s.")
