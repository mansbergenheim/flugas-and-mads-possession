#!/usr/bin/env python3
"""Turn the commentary chain's output into the small cue files the submission is rebuilt from. Run on the Mac.

    MTAH_DATA_DIR=~/mtah-data python3 scripts/export_cues.py \
        --mentions ~/mtah-data/commentary/possession_mentions.csv --events EVENTS.csv

Writes cues/ in this repository (or --out):

  possession_cues.csv   who has the ball: tactical frame, broadcast second, side (home/away), role,
                        the rule that made the cue ("model": the local model said on/wins/loses the
                        ball; "words": a ball-action word next to the name), transcript segment
  stoppage_cues.csv     what happened: tactical frame, broadcast second, label, status, basis, segment
  manifest.json         where they came from: the SHA-256 of every input (transcript, line-ups,
                        extractor outputs, alignment, tracking metadata), each extractor's own record
                        (model, model digest, Ollama version, prompt and script hashes, settings), the
                        rules applied here, the side check, the counts at every step, and the code
                        commit. No names, no commentary text, no paths.

Only cues said outside replays and inside the lined-up broadcast are kept, exactly as
make_submission.py --mentions/--events would place them. make_submission.py --cues cues/ then gives
the same file without the Mac-only steps. scripts/rebuild_cues.sh runs the whole chain from the match
files and compares its result with these files.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from mtah import commentary as cm  # noqa: E402
from mtah.loaders import gated  # noqa: E402
from mtah.provenance import environment, git_state, sha256_file, source_hash  # noqa: E402

HOME = os.path.expanduser("~")
DROP = {"transcript"}  # an extractor manifest field holding a local path, not a hash


def tidy(value):
    """Paths with the home folder hidden."""
    return value.replace(HOME, "~") if isinstance(value, str) else value


def extractor_record(csv_path: str) -> dict:
    path = os.path.splitext(csv_path)[0] + ".manifest.json"
    if not os.path.exists(path):
        sys.exit(f"the extractor's manifest was not found beside its output: {path}")
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    if record.get("csv_sha256") and record["csv_sha256"] != sha256_file(csv_path):
        sys.exit(f"{csv_path} is not the file its manifest describes")
    return {k: v for k, v in record.items() if k not in DROP}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mentions", required=True, help="scripts/extract_possession_mentions.py output")
    ap.add_argument("--events", required=True, help="scripts/extract_commentary_events.py output")
    ap.add_argument("--out", default=os.path.join(ROOT, "cues"))
    args = ap.parse_args()
    data = gated.data_dir()
    alignment = os.path.join(data, "alignment", "broadcast_alignment.npz")
    lineups = os.path.join(data, "lineups", "lineups.json")
    meta = gated.asset("atd_meta")
    for path in (args.mentions, args.events, alignment, lineups, meta):
        if not os.path.exists(path):
            sys.exit(f"not found: {tidy(path)}")

    poss, stop, info = cm.placed_cues(args.mentions, args.events, alignment)
    sides = cm.side_check(lineups, meta)
    cm.write_cues(args.out, poss, stop)
    files = {name: sha256_file(os.path.join(args.out, name)) for name in (cm.POSSESSION_CUES, cm.STOPPAGE_CUES)}
    record = {
        "made_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "made_by": "scripts/export_cues.py",
        "git": git_state(), "source_sha256": source_hash(), "environment": environment(),
        "files": files,
        "inputs_sha256": {"mentions": sha256_file(args.mentions), "events": sha256_file(args.events),
                          "alignment": sha256_file(alignment), "lineups": sha256_file(lineups),
                          "tracking_metadata": sha256_file(meta)},
        "extractors": {"possession_mentions": extractor_record(args.mentions),
                       "commentary_events": extractor_record(args.events)},
        "rules": {"kept_model_roles": list(cm.TEAM_ROLES), "action_words": cm.ACTION.pattern,
                  "action_window_chars": [cm.ACTION_BEFORE, cm.ACTION_AFTER],
                  "dropped": "mentions said during a broadcast replay or outside the lined-up broadcast"},
        "side_check": sides,
        "counts": info,
        "human_involvement": ("none in making these files: every step is a program; no cue was added, "
                              "removed or corrected by hand"),
    }
    with open(os.path.join(args.out, cm.CUE_MANIFEST), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, sort_keys=True)
        f.write("\n")
    c = info["cues"]
    print(f"wrote {tidy(args.out)}: {info['mentions']['kept']} possession cues ({c['model']} from the model, "
          f"{c['words']} from action words, before placing), {info['events']['kept']} stoppage reports; "
          f"side check {sides['verdict']}")
    for name, digest in files.items():
        print(f"  {name}  sha256 {digest[:16]}")
    print("These files are about the real match: commit them only to the submission repository.")
