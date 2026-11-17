"""Build the competition CSV in one command. Run this yourself, not through an AI tool.

Builds the submitted model, predicts on the competition match, writes the file and checks it. The
model is three tree models, trained here on the open matches, and the TCN shipped in models/tcn.npz,
with their probabilities averaged in equal shares (mtah/candidate.py). Output is deterministic, and
prints counts only.

    MTAH_DATA_DIR=~/mtah-data python3 scripts/make_submission.py --team <team_name>

With the commentary (off unless both files are given; see mtah/commentary.py):

    ... make_submission.py --team <team_name> --mentions MENTIONS.csv --events EVENTS.csv

or, on any computer, from the committed cue files those make (scripts/export_cues.py), same result:

    ... make_submission.py --team <team_name> --cues cues/

Only the sure who-has-the-ball cues are used (the model said on the ball, or a ball-action word is
next to the name), and only if the line-up's home team is the tracking file's H team and the cues
pass the gate (they must agree with the model where the model is sure); otherwise the file is built
without them, and the printout says so. Dead-ball reports from the events file are always used.
"""

import argparse
from dataclasses import replace
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mtah import candidate, commentary, datasets as ds, submission as sub  # noqa: E402
from mtah.config import DECODER, TAG_WEIGHTS, TREE_DRAWS  # noqa: E402
from mtah.decoder import decode_samples, expand  # noqa: E402
from mtah.degrade import load_stats  # noqa: E402
from mtah.loaders import gated  # noqa: E402
from mtah.match import X  # noqa: E402
from mtah.provenance import manifest, open_data, save_manifest, sha256_file  # noqa: E402
from mtah.smart_tagging import halves_without_kickoff, tag_prior  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", required=True, help="file is named after this, as the rules require")
    parser.add_argument("--out", default="submissions")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="open matches to leave out of the trees' training (the TCN's weights are fixed)")
    parser.add_argument("--no-tags", action="store_true", help="ignore Smart Tagging")
    parser.add_argument("--mentions", default=None,
                        help="who-has-the-ball commentary (scripts/extract_possession_mentions.py); needs --events")
    parser.add_argument("--events", default=None,
                        help="commentary events (scripts/extract_commentary_events.py); needs --mentions")
    parser.add_argument("--team-weight", type=float, default=commentary.CommentaryWeights().team,
                        help="commentary: nudge per sample towards the team named on the ball (default %(default)s)")
    parser.add_argument("--dead-weight", type=float, default=commentary.CommentaryWeights().dead,
                        help="commentary: nudge per sample towards dead ball around a reported stoppage (default %(default)s)")
    parser.add_argument("--cues", default=None,
                        help="the committed commentary cues (scripts/export_cues.py), instead of --mentions/--events")
    args = parser.parse_args()
    if bool(args.mentions) != bool(args.events):
        sys.exit("give both --mentions and --events, or neither")
    if args.cues and args.mentions:
        sys.exit("give --cues or --mentions/--events, not both")
    if args.cues:
        for name in (commentary.POSSESSION_CUES, commentary.STOPPAGE_CUES, commentary.CUE_MANIFEST):
            if not os.path.exists(os.path.join(args.cues, name)):
                sys.exit(f"not found: {os.path.join(args.cues, name)}")

    record = manifest(exclude=args.exclude, tags_used=not args.no_tags)

    # Check the inputs before spending minutes on training.
    match = gated.load_atd_match()
    tags = None
    if not args.no_tags:
        tags = gated.load_smart_tagging()
        missing = halves_without_kickoff(tags, match.halves)
        if missing:
            sys.exit(f"no Kick Off (Start) tag in {len(missing)} half(s): check the Smart Tagging file")
    alignment = None
    if args.mentions:
        alignment = os.path.join(gated.data_dir(), "alignment", "broadcast_alignment.npz")
        lineups = os.path.join(gated.data_dir(), "lineups", "lineups.json")
        for path in (args.mentions, args.events, alignment, lineups):
            if not os.path.exists(path):
                sys.exit(f"not found: {path}")
    # The network's weights ship with the repository. Anything but the pinned file stops the build here.
    try:
        tcn = candidate.load_tcn()
    except (OSError, ValueError) as error:
        sys.exit(str(error))

    # A missing or changed open file, or a cache left over from older loader code, would quietly
    # train a different model than a fresh clone does. So check every file, then rebuild the caches.
    problems = ds.check_open_files()
    if problems:
        sys.exit("open data differs from the pinned files, run scripts/download_open_data.sh again:\n  "
                 + "\n  ".join(problems))
    record.update(open_files_checked=True)
    stats = load_stats()
    ids = [i for i in ds.available() if i not in args.exclude]
    print(f"rebuilding {len(ids)} open matches from the raw files", flush=True)
    for match_id in ids:
        ds.load_open(match_id, refresh=True)
    record.update(open_data_sha256=open_data())
    print(f"training {TREE_DRAWS} tree models on {len(ids)} open matches", flush=True)
    model = candidate.Candidate(candidate.train_trees(ids, stats), tcn)
    record.update(model=model.name, tree_draws=TREE_DRAWS, tcn_sha256=candidate.TCN_SHA256)

    print("predicting", flush=True)
    prob, frames, half = model.probabilities(match)
    prior = None if tags is None else tag_prior(tags, frames, half, match.halves, TAG_WEIGHTS)
    if args.mentions or args.cues:
        weights = replace(commentary.CommentaryWeights(), team=args.team_weight, dead=args.dead_weight)
        if args.cues:
            poss, stop, cue_manifest = commentary.read_cues(args.cues)
            said = commentary.masks(poss, stop, frames, half, match.halves, weights)
            check = commentary.gate_from_cues(said, prob, cue_manifest)
            source = {"cues_sha256": {name: sha256_file(os.path.join(args.cues, name)) for name in
                                      (commentary.POSSESSION_CUES, commentary.STOPPAGE_CUES, commentary.CUE_MANIFEST)},
                      "cues": cue_manifest["counts"]["cues"]}
        else:
            said, info = commentary.load(args.mentions, args.events, alignment, frames, half, match.halves, weights)
            check = commentary.team_gate(said, prob, lineups, gated.asset("atd_meta"))
            source = {"mentions_sha256": sha256_file(args.mentions), "events_sha256": sha256_file(args.events),
                      "cues": info["cues"]}
        if check["verdict"] != "on":
            said["team"][:] = False         # who-has-the-ball evidence off; dead-ball evidence stays
        extra = commentary.prior_from(said, weights)
        prior = extra if prior is None else prior + extra
        record.update(commentary={**source, "gate": check["verdict"], "sides": check["sides"],
                                  "gate_samples": check["samples"], "weights": vars(weights)})
        print(f"commentary: who has the ball {'USED' if check['verdict'] == 'on' else 'NOT used (gate: ' + check['verdict'] + ')'}; "
              f"dead-ball reports used (strengths: team {weights.team:g}, dead {weights.dead:g})", flush=True)
    path = decode_samples(prob, half, DECODER, prior)
    codes = sub.force_no_abstain(expand(path, frames, half, match.halves, match.n_frames, default=X))

    problems = sub.check_submission(codes)
    if problems:
        sys.exit("not written:\n  " + "\n  ".join(problems))
    os.makedirs(args.out, exist_ok=True)
    target = os.path.join(args.out, f"{args.team}.csv")
    sub.write_submission(target, codes)
    problems = sub.check_file(target, codes)
    if problems:
        sys.exit("written badly:\n  " + "\n  ".join(problems))
    digest = sha256_file(target)
    record.update(submission_sha256=digest, counts=sub.summary(codes))
    save_manifest(os.path.join(args.out, f"{args.team}.manifest.json"), record)

    print(f"written {target}  sha256 {digest[:16]}  (model: {model.name})")
    for key, value in sub.summary(codes).items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    print("Now run the organisers' validator on this exact file.")
