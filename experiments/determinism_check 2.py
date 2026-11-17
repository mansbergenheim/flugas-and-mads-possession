"""Would another computer write exactly the same file? Checked on open data only.

The rules ask for a repo that regenerates the exact same CSV. The organisers will probably rerun it
on a machine with a different chip (Intel or AMD rather than Apple) and a different number of cores.
This can't test another chip, but it can test the two things around it:

  1. threads and hash seed: the same small training run with 1, 3 and all cores, and two Python hash
     seeds, must give bit-identical probabilities. If not, the core count alone changes the file.
  2. last-bit sensitivity: a different chip can round a few maths functions differently in the last
     bit. Here every second feature value is moved by one float32 step, far more often than a chip
     difference would, in training and at prediction. Counted: frames whose label changes on two
     held-out open matches. Also counted with the original model and only the test features moved,
     and with only the probabilities moved by one part in 10^12 before decoding.
  3. realistic size (run with --part 3): on the small run from part 1, only one training feature
     value in 10,000 moved by one step, or the sample weights of 1% of rows moved by one step. The
     second stands in for the rare last-bit differences between maths libraries inside training.

Part 1 takes a few minutes, part 2 about 20, part 3 about 5. Nothing here changes the model.

    python3 experiments/determinism_check.py
    python3 experiments/determinism_check.py --part 3
"""

import hashlib
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

SMALL_TRAIN = ("J03WPY", "METRICA1")
SMALL_TEST = "J03WOH"
HELD_OUT = ("J03WMX", "METRICA2")


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def child() -> None:
    """One small training run; prints hashes of the probabilities and the labels."""
    from mtah import harness
    from mtah.config import DECODER
    from mtah.degrade import DegradeConfig, load_stats
    from mtah.model import LearnedModel
    stats = load_stats()
    model = LearnedModel(decoder=DECODER)
    model.fit(harness.training_matches(list(SMALL_TRAIN), 0, 1, stats))
    test = harness.without_labels(harness.prepared(SMALL_TEST, DegradeConfig(), harness.TEST_SEED_OFFSET, stats))
    prob, _, _ = model.probabilities(test)
    print(digest(prob), digest(model.predict(test)))


def part_one() -> bool:
    runs = {"all cores": {}, "1 thread": {"OMP_NUM_THREADS": "1"}, "3 threads": {"OMP_NUM_THREADS": "3"},
            "hash seed 0": {"PYTHONHASHSEED": "0"}, "hash seed 12345": {"PYTHONHASHSEED": "12345"}}
    seen = {}
    for name, extra in runs.items():
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **extra}
        out = subprocess.run([sys.executable, __file__, "--child"], env=env, cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
        seen[name] = tuple(out[-2:])
        print(f"  {name:<16} probabilities {seen[name][0]}  labels {seen[name][1]}", flush=True)
    same = len(set(seen.values())) == 1
    print("  identical" if same else "  DIFFERENT: the core count or hash seed changes the output", flush=True)
    return same


class Nudge:
    """Moves a share of feature values (every second by default) by one float32 step, up or down."""

    def __init__(self, build, seed: int, share: float = 0.5):
        self.build = build
        self.rng = np.random.default_rng(seed)
        self.share = share

    def __call__(self, match):
        values, names, frames, half = self.build(match)
        pick = self.rng.random(values.shape) < self.share
        way = np.where(self.rng.random(values.shape) < 0.5, np.float32(np.inf), np.float32(-np.inf))
        values = np.where(pick, np.nextafter(values, way), values).astype(np.float32)
        return values, names, frames, half


def part_two() -> None:
    from mtah import datasets as ds, harness, metrics as m, model as model_module
    from mtah.config import DECODER, TRAIN_COPIES
    from mtah.decoder import decode_samples, expand
    from mtah.degrade import DegradeConfig, load_stats
    from mtah.model import LearnedModel

    stats = load_stats()
    ids = [i for i in ds.available() if i not in HELD_OUT]
    tests = {i: harness.prepared(i, DegradeConfig(), harness.TEST_SEED_OFFSET, stats) for i in HELD_OUT}
    original = model_module.build_features

    def labels(model, test, scale=None):
        prob, frames, half = model.probabilities(harness.without_labels(test))
        if scale is not None:
            prob = prob * scale(prob.shape)
        return expand(decode_samples(prob, half, DECODER), frames, half, test.halves, test.n_frames)

    def report(name, base, other):
        for i in HELD_OUT:
            changed = int((base[i] != other[i]).sum())
            gap = m.evaluate(tests[i].labels, other[i])["spell_f1"] - m.evaluate(tests[i].labels, base[i])["spell_f1"]
            print(f"  {name:<34} {i:<9} {changed:>6} of {len(base[i])} frames changed, spellF1 {gap:+.4f}",
                  flush=True)

    model = LearnedModel(decoder=DECODER)
    model.fit(harness.training_matches(ids, 0, TRAIN_COPIES, stats))
    base = {i: labels(model, t) for i, t in tests.items()}

    rng = np.random.default_rng(7)
    report("probabilities x (1 +- 1e-12)", base,
           {i: labels(model, t, lambda s: 1.0 + 1e-12 * rng.choice((-1.0, 1.0), size=s)) for i, t in tests.items()})

    model_module.build_features = Nudge(original, 11)
    try:
        report("test features nudged", base, {i: labels(model, t) for i, t in tests.items()})
        nudged = LearnedModel(decoder=DECODER)
        nudged.fit(harness.training_matches(ids, 0, TRAIN_COPIES, stats))
        report("training and test features nudged", base, {i: labels(nudged, t) for i, t in tests.items()})
    finally:
        model_module.build_features = original


def part_three() -> None:
    from mtah import harness, model as model_module
    from mtah.config import DECODER
    from mtah.degrade import DegradeConfig, load_stats
    from mtah.model import LearnedModel

    class NudgedWeights(LearnedModel):
        rng = np.random.default_rng(13)

        def _rows(self, match):
            x, y, names, w = super()._rows(match)
            pick = self.rng.random(len(w)) < 0.01
            way = np.where(self.rng.random(len(w)) < 0.5, np.inf, -np.inf)
            return x, y, names, np.where(pick, np.nextafter(w, way), w)

    stats = load_stats()
    train = list(harness.training_matches(list(SMALL_TRAIN), 0, 1, stats))
    test = harness.without_labels(harness.prepared(SMALL_TEST, DegradeConfig(), harness.TEST_SEED_OFFSET, stats))

    def fitted(cls=LearnedModel, build=None):
        original = model_module.build_features
        model_module.build_features = build or original
        try:
            model = cls(decoder=DECODER)
            model.fit(train)
        finally:
            model_module.build_features = original
        return model.predict(test)

    base = fitted()
    moved = fitted(build=Nudge(model_module.build_features, 17, share=1e-4))
    weights = fitted(NudgedWeights)
    for name, other in (("1 in 10,000 feature values", moved), ("weights of 1% of rows", weights)):
        print(f"  {name:<28} {SMALL_TEST}  {int((other != base).sum()):>6} of {len(base)} frames changed",
              flush=True)


if __name__ == "__main__":
    if "--child" in sys.argv:
        child()
    elif "--part" in sys.argv and sys.argv[sys.argv.index("--part") + 1] == "3":
        os.chdir(ROOT)
        print("\n3. realistic size (small training run, training only)", flush=True)
        part_three()
    else:
        os.chdir(ROOT)
        print("1. threads and hash seed (small training run, open data)", flush=True)
        part_one()
        print("\n2. last-bit sensitivity (trained without the two test matches)", flush=True)
        part_two()
