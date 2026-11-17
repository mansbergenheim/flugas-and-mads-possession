"""Run a script with open-match labels closer to the organisers' written definition of possession.

Applies to training and test labels alike, so a model built this way is compared with the current one
by experiments/compare_definition_labels.py, which scores both on both sets of labels.

    python3 experiments/with_definition_labels.py scripts/train_oof.py --tag v5_def
"""

import dataclasses
import os
import runpy
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from mtah import harness  # noqa: E402
from short_possessions import follow_definition  # noqa: E402

current = harness.harmonise


def harmonise(match, *args, **kwargs):
    match = current(match, *args, **kwargs)
    return dataclasses.replace(match, labels=follow_definition(match))


harness.harmonise = harmonise
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
