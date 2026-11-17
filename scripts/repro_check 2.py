"""Does this machine build the same model as ours? Open data only, so anyone can run and share it.

Builds the model exactly as make_submission.py does (same open files, checked and rebuilt, same
seeds, the same shipped TCN), then predicts one open match with the standard test treatment and
prints fingerprints of what the model answers. Compare them with the expected values in the README.
If they match, this machine builds the submitted model bit for bit, and make_submission.py should
write the same CSV.

The two halves of the model are fingerprinted separately as well, so that a difference can be traced:
the trees are trained here, the TCN is only read from its file and run in numpy.

    python3 scripts/repro_check.py
"""

import hashlib
import os
import platform
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mtah import candidate, datasets as ds, harness  # noqa: E402
from mtah.config import DECODER  # noqa: E402
from mtah.decoder import decode_samples, expand  # noqa: E402
from mtah.degrade import DegradeConfig, load_stats  # noqa: E402

CHECK_MATCH = "METRICA2"


def fingerprint(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


if __name__ == "__main__":
    problems = ds.check_open_files()
    if problems:
        sys.exit("open data differs from the pinned files, run scripts/download_open_data.sh again:\n  "
                 + "\n  ".join(problems))
    try:
        candidate.load_tcn()
    except (OSError, ValueError) as error:
        sys.exit(str(error))
    stats = load_stats()
    ids = ds.available()
    print(f"rebuilding {len(ids)} open matches from the raw files", flush=True)
    for match_id in ids:
        ds.load_open(match_id, refresh=True)
    print(f"training on {len(ids)} open matches", flush=True)
    model = candidate.build(ids, stats)

    test = harness.without_labels(harness.prepared(CHECK_MATCH, DegradeConfig(), harness.TEST_SEED_OFFSET, stats))
    trees, tcn, frames, half = model.parts(test)
    prob = (trees + tcn) / 2.0
    labels = expand(decode_samples(prob, half, DECODER), frames, half, test.halves, test.n_frames)
    import scipy
    import sklearn
    print(f"machine {platform.machine()}, {platform.system()}, Python {platform.python_version()}, numpy "
          f"{np.__version__}, scipy {scipy.__version__}, scikit-learn {sklearn.__version__}")
    print(f"test copy     {fingerprint(np.nan_to_num(test.ball))}")
    print(f"trees         {fingerprint(trees)}")
    print(f"tcn           {fingerprint(tcn)}")
    print(f"probabilities {fingerprint(prob)}")
    print(f"labels        {fingerprint(labels)}")
