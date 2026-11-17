"""What produced a result: source, settings, environment and inputs.

Every saved run carries a manifest, so a score can be traced to the exact code that made it. Commit
before a run you mean to keep: `dirty: true` means the source differed from the commit, and then only
source_sha256 identifies it. Manifests hold no match data, so they are safe to commit.
"""

import dataclasses
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Dict, Optional

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE = ("mtah/**/*.py", "mtah/*.json", "scripts/*.py", "scripts/*.sh", "requirements.txt")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hash() -> str:
    """One hash over every source file and its name."""
    digest = hashlib.sha256()
    paths = {p for pattern in SOURCE for p in glob.glob(os.path.join(PROJECT, pattern), recursive=True)}
    for path in sorted(paths):
        digest.update(os.path.relpath(path, PROJECT).encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def git_state() -> Optional[Dict]:
    def git(*args):
        return subprocess.run(["git", "-C", PROJECT, *args], capture_output=True, text=True, check=True).stdout
    try:
        return {"commit": git("rev-parse", "HEAD").strip(),
                "dirty": bool(git("status", "--porcelain", "--", "mtah", "scripts", "requirements.txt").strip())}
    except (OSError, subprocess.CalledProcessError):
        return None


def settings() -> Dict:
    from . import candidate, config, harness
    from .degrade import DegradeConfig
    from .model import TreeConfig
    return {"decoder": dataclasses.asdict(config.DECODER),
            "tag_weights": dataclasses.asdict(config.TAG_WEIGHTS),
            "spell_weight": config.SPELL_WEIGHT, "train_copies": config.TRAIN_COPIES,
            "tree_draws": config.TREE_DRAWS, "tcn_sha256": candidate.TCN_SHA256,
            "trees": dataclasses.asdict(TreeConfig()), "degrade": dataclasses.asdict(DegradeConfig()),
            "dfl_min_control_s": harness.DFL_MIN_CONTROL_S, "test_seed_offset": harness.TEST_SEED_OFFSET}


def open_data() -> Dict[str, str]:
    from . import datasets as ds
    return {os.path.basename(p)[:-4]: sha256_file(p) for p in sorted(glob.glob(os.path.join(ds.CACHE, "*.npz")))}


def environment() -> Dict:
    import numpy
    import scipy
    import sklearn
    return {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "numpy": numpy.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}


def manifest(**extra) -> Dict:
    """Call at the start of a run, so later edits to the source can't change what it records."""
    return {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "command": sys.argv,
            "git": git_state(), "source_sha256": source_hash(), "settings": settings(),
            "environment": environment(), "open_data_sha256": open_data(), **extra}


def save_manifest(path: str, record: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        json.dump(record, handle, indent=1, default=float)
