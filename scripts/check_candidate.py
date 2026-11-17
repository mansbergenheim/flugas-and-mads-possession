"""Check the candidate on the real match: the trees and the TCN, averaged. Run this yourself, not through an AI tool.

Prints words only, never a measurement, so what it prints can be repeated to anyone. About 5 minutes.

    MTAH_DATA_DIR=~/mtah-data python3 scripts/check_candidate.py

The candidate is built from saved models, all trained on the 10 open matches: the three tree models in
runs/models/v6_candidate.pkl and the TCN in runs/network_results/tcn/all/. The saved v5 goes through
the same steps as the yardstick.

1. FILE     The candidate's labels for the whole match, made the way make_submission.py makes v5's (same
            decoder setting, same use of the tags), against our own checks. The file is written to
            <data folder>/candidate_check/, outside this workspace. It is a test file, not the submission.
2. MEMBERS  How often the trees and the TCN name a different class at the same moment. On open matches
            they had not seen it is 5 to 8 moments in 100. On SkillCorner, where the networks are at
            their weakest, it is 13 to 21. Much more disagreement than at home means the network is on
            unfamiliar ground.
3. LABELS   How many frames the candidate labels differently from v5.
4. OUTPUT   How sure the candidate is of itself, the questions check_real_match.py asks of v5.
5. PUBLIC   The candidate against v5 on the public five minutes, with our copy of the scorer.

Read 5 for one thing only: is something broken? One window cannot rank two versions. On the open
matches the candidate is 0.064 ahead of v5 over whole matches, and still behind v5 in one five-minute
window in seven. So the line says NORMAL unless the candidate is further behind v5 than in 19 of 20
windows on the open matches, and it says the same whether the candidate is ahead or a little behind.
Never choose between versions on this window.

With --score, part 5 also prints the two scores themselves, spellF1 on the public five minutes for the
candidate and for v5, marked "[a number]". Those two lines are measurements of the real match: keep
them on this computer. The rest stays words only.

    MTAH_DATA_DIR=~/mtah-data python3 scripts/check_candidate.py --score

For the organisers' own figure, point their scorer at the folder with the two test files:

    mkdir -p ~/mtah-data/candidate_public && python3 ../official-repository/evaluation/run_evaluation.py --results ~/mtah-data/candidate_check --out ~/mtah-data/candidate_public

The reference values come from open matches only: mtah/check_candidate.json, written by
scripts/fit_candidate_ranges.py.
"""

import argparse
import csv
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, ROOT)
from check_real_match import OUTPUT_NAMES, kickoff_blocks, output_figures  # noqa: E402
from compare_tracking import position  # noqa: E402
from mtah import metrics as m, network, submission as sub  # noqa: E402
from mtah.config import DECODER, TAG_WEIGHTS  # noqa: E402
from mtah.decoder import decode_samples, expand  # noqa: E402
from mtah.loaders import gated  # noqa: E402
from mtah.match import A, D, H, X  # noqa: E402
from mtah.smart_tagging import halves_without_kickoff, tag_prior  # noqa: E402

RANGES = os.path.join(ROOT, "mtah", "check_candidate.json")
V5 = os.path.join(ROOT, "runs", "models", "v5_final_model.pkl")
TREES = os.path.join(ROOT, "runs", "models", "v6_candidate.pkl")
TCN = os.path.join(ROOT, "runs", "network_results", "tcn", "all", "network_model_tcn.npz")
PUBLIC = os.path.join(ROOT, "..", "official-repository", "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
FILE_NAME = "candidate_trees_tcn.csv"
YARDSTICK_NAME = "yardstick_v5.csv"
NUMBER = "[a number]"
SWAP_MARGIN = 0.10     # spellF1: a file with home and away exchanged scores far more than this higher when put right
NEAR_NOTHING = 0.10    # spellF1: below this, almost no possession is matched


def between(value: float, home, away) -> str:
    """Where a share sits against the open matches (low, quarter, quarter, high) and against SkillCorner."""
    if value < home[0]:
        return "less than on any open match"
    if value <= home[3]:
        return "as on the open matches (" + position(value, *home).replace("inside, ", "") + ")"
    if value < away[0]:
        nearer = "the open matches" if value - home[3] < away[0] - value else "SkillCorner"
        return f"MORE than on any open match, less than on any SkillCorner match, nearer {nearer}"
    return "MORE than on any open match: as on SkillCorner" if value <= away[3] else "MORE than on any SkillCorner match"


def read_public(path: str, n_frames: int) -> np.ndarray:
    """The organisers' public labels, one row per frame. Anything that is not H, A or D is not graded."""
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))[1:]
    if len(rows) != n_frames:
        raise ValueError(f"the public label file has {len(rows)} rows, the match {n_frames} frames")
    lookup = {"H": H, "A": A, "D": D}
    return np.array([lookup.get(row[1].strip().upper(), X) for row in rows], dtype=np.int8)


def exchanged(codes: np.ndarray) -> np.ndarray:
    out = codes.copy()
    out[codes == H], out[codes == A] = A, H
    return out


def public_lines(path: str, candidate: np.ndarray, yardstick: np.ndarray, window: dict, show: bool = False):
    """The lines of part 5. With show, the two scores themselves come first, marked as numbers."""
    truth = read_public(path, len(candidate))
    if not (truth != X).any():
        return [("The public labels", "none found in the file")]

    def score(codes):
        return m.evaluate(truth, codes)["spell_f1"]

    mine, old = score(candidate), score(yardstick)
    lines = []
    if show:
        lines += [(f"spellF1, the candidate  {NUMBER}", f"{mine:.3f}"), (f"spellF1, v5  {NUMBER}", f"{old:.3f}")]
    for name, value, codes in (("the candidate", mine, candidate), ("v5", old, yardstick)):
        if score(exchanged(codes)) > value + SWAP_MARGIN:
            lines.append((f"Home and away, {name}", "BROKEN: it scores clearly higher with home and away exchanged"))
        elif value < NEAR_NOTHING:
            lines.append((f"Home and away, {name}", "BROKEN: almost nothing matches, either way round"))
        else:
            lines.append((f"Home and away, {name}", "the right way round"))
    gap = mine - old
    if gap >= window["p05"]:
        words = "NORMAL: not further behind v5 than one window allows (it may be ahead)"
    elif gap >= window["p01"]:
        words = "LOW: further behind v5 than in 19 of 20 windows on the open matches"
    else:
        words = "VERY LOW: further behind v5 than in 99 of 100 windows on the open matches"
    lines.append(("The candidate against v5", words))
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", help="folder for the test file; default <data folder>/candidate_check")
    parser.add_argument("--public", default=PUBLIC, help="the organisers' public label file")
    parser.add_argument("--no-tags", action="store_true", help="ignore Smart Tagging")
    parser.add_argument("--score", action="store_true",
                        help="also print spellF1 on the public window for the candidate and v5. These are numbers "
                             "from the real match: keep them on this computer")
    args = parser.parse_args()
    with open(RANGES) as handle:
        ranges = json.load(handle)
    for path in (V5, TREES, TCN):
        if not os.path.exists(path):
            sys.exit(f"missing {os.path.relpath(path, ROOT)}: this check needs the saved models")

    match = gated.load_atd_match()
    tags = None
    if not args.no_tags:
        try:
            tags = gated.load_smart_tagging()
            if halves_without_kickoff(tags, match.halves):
                tags = None
        except (OSError, ValueError):
            tags = None
    with open(V5, "rb") as handle:
        v5 = pickle.load(handle)
    with open(TREES, "rb") as handle:
        trees = pickle.load(handle)["ensemble"].models
    tcn = network.load(TCN)
    if tcn["config"]["names"] != trees[0].names:
        sys.exit("the trees and the TCN were trained on different inputs")

    print("predicting: the trees", flush=True)
    values, swapped, frames, half = trees[0].views(match)
    tree_prob = sum(model.probabilities_from(values, swapped) for model in trees) / float(len(trees))
    print("predicting: the TCN", flush=True)
    tcn_prob = network.probabilities(tcn, values, half)
    print("predicting: v5", flush=True)
    v5_prob, v5_frames, v5_half = v5.probabilities(match)
    if not (np.array_equal(frames, v5_frames) and np.array_equal(half, v5_half)):
        sys.exit("v5 and the candidate are not sampled at the same moments")
    prob = (tree_prob + tcn_prob) / 2.0

    def final(p):
        """The labels as make_submission.py makes them."""
        prior = None if tags is None else tag_prior(tags, frames, half, match.halves, TAG_WEIGHTS)
        path = decode_samples(p, half, DECODER, prior)
        return sub.force_no_abstain(expand(path, frames, half, match.halves, match.n_frames, default=X))

    def plain(p):
        """The labels without the tags, as on the open matches the references come from."""
        return expand(decode_samples(p, half, DECODER), frames, half, match.halves, match.n_frames)

    candidate, yardstick = final(prob), final(v5_prob)
    play = kickoff_blocks(match) if tags is not None else None
    blocks = play or match.halves
    moments = np.zeros(len(frames), dtype=bool)
    inside = np.zeros(match.n_frames, dtype=bool)
    for start, stop in blocks:
        moments |= (frames >= start) & (frames < stop)
        inside[start:stop] = True

    print(f"\nCandidate: the trees ({len(trees)} saved models) and the TCN ({len(tcn['seeds'])} saved networks), "
          "averaged. Yardstick: the saved v5.")
    print("Tags: " + ("used, as in make_submission.py" if tags is not None else
                      "NOT used (switched off, unreadable, or no kickoff tag in a half)"))

    print("\n1. FILE")
    problems = sub.check_submission(candidate)
    out = args.out or os.path.join(gated.data_dir(), "candidate_check")
    if problems:
        print(f"  {'Our own checks':<42} FAILED, nothing written: " + "; ".join(problems))
    else:
        os.makedirs(out, exist_ok=True)
        target = os.path.join(out, FILE_NAME)
        sub.write_submission(target, candidate)
        problems = sub.check_file(target, candidate)
        print(f"  {'Our own checks':<42} " + ("passed" if not problems else "written badly: " + "; ".join(problems)))
        print(f"  {'Written to':<42} {target}")
        print(f"  {'':<42} a test file, not the submission; it stays on this computer")
        if not sub.check_submission(yardstick):
            sub.write_submission(os.path.join(out, YARDSTICK_NAME), yardstick)
            print(f"  {'Beside it':<42} {YARDSTICK_NAME}: the saved v5's labels, made in the same run")

    print("\n2. MEMBERS")
    differ = float((tree_prob[moments].argmax(1) != tcn_prob[moments].argmax(1)).mean())
    print(f"  {'The trees and the TCN disagree':<42} " + between(differ, ranges["members"]["open"],
                                                                 ranges["members"]["skillcorner"]))

    print("\n3. LABELS")
    changed = float((plain(prob)[inside] != plain(v5_prob)[inside]).mean())
    print(f"  {'Frames labelled differently from v5':<42} " + between(changed, ranges["labels"]["open"],
                                                                    ranges["labels"]["skillcorner"]))

    print("\n4. OUTPUT" + ("" if play else "  (no tags, so the wait before the second-half kickoff is included)"))
    seen = np.isfinite(match.ball[frames, 0])
    figures = output_figures(prob, frames, half, seen, match.halves, match.n_frames, play)
    for key, label in OUTPUT_NAMES.items():
        verdict = position(figures[key], *ranges["output"][key]) if np.isfinite(figures[key]) else "not measurable"
        print(f"  {label:<42} {verdict.replace('trained on', 'seen on open matches')}")

    print("\n5. PUBLIC WINDOW  (is something broken? one window cannot rank two versions)")
    shown = False
    if not os.path.exists(args.public):
        print(f"  skipped: no public label file at {args.public}")
    else:
        try:
            for label, words in public_lines(args.public, candidate, yardstick, ranges["window"], args.score):
                print(f"  {label:<42} {words}")
                shown = shown or NUMBER in label
        except (ValueError, IndexError) as error:
            print(f"  skipped: {error}")

    if shown:
        window = ranges["window"]
        print(f"\nThe two lines marked {NUMBER} come from the real match and our copy of the scorer. Keep them on this\n"
              "computer: say them to an AI tool in words, never as numbers. Everything else above is words only.\n"
              f"How to read them: on the open matches the candidate is {window['mean']:.3f} ahead of v5 over whole matches "
              f"and still\nbehind v5 in {round(window['below_v5'] * window['windows'])} of {window['windows']} "
              "five-minute windows. Never choose between versions on this window.")
    else:
        print("\nEverything above is words only.")
