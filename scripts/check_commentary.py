#!/usr/bin/env python3
"""Check the commentary on the real match before it goes into the submission. Run this yourself.

    python3 scripts/check_commentary.py --mentions MENTIONS.csv --events EVENTS.csv

Prints words and shares only: nothing frame by frame, no names, no labels, and it never looks at the
public five minutes' answers. About two minutes: the candidate is built from the saved models
(the three trees in runs/models/v6_candidate.pkl or models/v6_candidate.pkl, and models/tcn.npz),
exactly the model make_submission.py builds.

1 PLACED    how many reports land on the tactical timeline, outside it, or during a replay
2 COVERAGE  how much of the match gets a team nudge or a dead-ball nudge
0 CUES      the who-has-the-ball mentions kept: the model's on/wins/loses-ball answers and any
            mention with a ball-action word next to the name (mtah/commentary.py sure_cues)
3 SIDES     is the line-up graphics' home team the tracking file's H team? By name and shirt numbers.
  GATE      where the model is sure who has the ball, how often the commentary names the same team.
            "on" (at least 70%, over 25 sure moments if the sides agree, 200 if not): the team
            evidence may be used. "noise" or "swapped": it is switched off. Fixed before the real match was looked at; see mtah/commentary.py.
4 EFFECT    how many frames change with the commentary, and of what kind
"""
import argparse
from dataclasses import replace
import os
import pickle
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from mtah import candidate, commentary  # noqa: E402
from mtah.config import DECODER, TAG_WEIGHTS  # noqa: E402
from mtah.decoder import decode_samples, expand  # noqa: E402
from mtah.loaders import gated  # noqa: E402
from mtah.match import A, D, H, X  # noqa: E402
from mtah.smart_tagging import halves_without_kickoff, tag_prior  # noqa: E402

TREES = [os.path.join(ROOT, "runs", "models", "v6_candidate.pkl"), os.path.join(ROOT, "models", "v6_candidate.pkl")]


def share(x: float) -> str:
    return f"{100 * x:.0f}%" if x >= 0.01 or x == 0 else "under 1%"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mentions", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--team-weight", type=float, default=commentary.CommentaryWeights().team,
                    help="nudge per sample towards the team named on the ball (default %(default)s)")
    ap.add_argument("--dead-weight", type=float, default=commentary.CommentaryWeights().dead,
                    help="nudge per sample towards dead ball around a reported stoppage (default %(default)s)")
    args = ap.parse_args()
    trees_path = next((p for p in TREES if os.path.exists(p)), None)
    if trees_path is None:
        sys.exit("the saved tree models were not found (runs/models/v6_candidate.pkl or models/v6_candidate.pkl)")
    alignment = os.path.join(gated.data_dir(), "alignment", "broadcast_alignment.npz")
    with open(trees_path, "rb") as f:
        trees = pickle.load(f)["ensemble"].models
    model = candidate.Candidate(trees, candidate.load_tcn())
    match = gated.load_atd_match()
    tags = gated.load_smart_tagging()
    if halves_without_kickoff(tags, match.halves):
        tags = None
    print("predicting with the candidate (trees + TCN)", flush=True)
    prob, frames, half = model.probabilities(match)
    weights = replace(commentary.CommentaryWeights(), team=args.team_weight, dead=args.dead_weight)
    said, info = commentary.load(args.mentions, args.events, alignment, frames, half, match.halves, weights)

    print(f"            nudge strengths: team {weights.team:g}, dead ball {weights.dead:g} per sample")
    c = info["cues"]
    print(f"0 CUES      who-has-the-ball cues kept: {c['model']} the model said on/wins/loses the ball, "
          f"{c['words']} a ball-action word next to the name; {c['dropped']} other mentions dropped")
    for k, name in enumerate(("mentions", "events")):
        i = info[name]
        print(f"{'1 PLACED' if not k else '':<12}{name:<9} {i['kept']} of {i['rows']} on the tactical timeline; "
              f"{i['outside']} outside the lined-up broadcast; {i['replay']} said during a replay (dropped)")
    print(f"2 COVERAGE  team nudge on {share(said['team'].any(axis=1).mean())} of the moments; dead-ball nudge on "
          f"{share(said['dead'].mean())}; live-play nudge on {share(said['alive'].mean())}")
    lineups = os.path.join(gated.data_dir(), "lineups", "lineups.json")
    check = commentary.team_gate(said, prob, lineups, gated.asset("atd_meta"))
    sides = check["sides"]
    print(f"3 SIDES     line-up home team vs the tracking file's H team: by name {sides['names']}, "
          f"by shirt numbers {sides['numbers']} -> {sides['verdict']}"
          + ("" if sides["verdict"] == "agrees" else " (the gate needs the full 200 sure moments)"))
    agree = "has no sure moments to compare" if not check["samples"] else f"agrees {share(check['agree'])}"
    meaning = {"on": "ON: the team evidence may be used",
               "noise": "OFF: about as often wrong as right, so it would only add noise",
               "swapped": ("OFF: the line-up's home team looks like the tracking file's AWAY team: investigate"
                           if sides["verdict"] == "swapped" else
                           "OFF: it mostly names the other team; something is swapped (line-ups or sides): investigate"),
               "too little": "OFF: too few sure moments with a named team to judge"}[check["verdict"]]
    print(f"  GATE      where the model is sure ({commentary.SURE:.0%}+), the commentary {agree} "
          f"over {check['samples']} moments -> {meaning}")

    base_prior = None if tags is None else tag_prior(tags, frames, half, match.halves, TAG_WEIGHTS)
    if check["verdict"] != "on":
        said["team"][:] = False
    extra = commentary.prior_from(said, weights)
    with_prior = extra if base_prior is None else base_prior + extra

    def labels(prior):
        return expand(decode_samples(prob, half, DECODER, prior), frames, half, match.halves, match.n_frames, default=X)

    before, after = labels(base_prior), labels(with_prior)
    graded = np.zeros(match.n_frames, dtype=bool)
    for a, b in match.halves:
        graded[a:b] = True
    changed = graded & (before != after)
    team_swap = changed & np.isin(before, (H, A)) & np.isin(after, (H, A))
    to_dead = changed & (after == D)
    from_dead = changed & (before == D)
    print(f"4 EFFECT    {changed.sum()} of {graded.sum()} graded frames change ({100 * changed.sum() / graded.sum():.2f}%), "
          f"in {int((np.diff(changed.astype(np.int8)) == 1).sum())} stretches: "
          f"{share(team_swap.sum() / max(changed.sum(), 1))} of those swap team, "
          f"{share(to_dead.sum() / max(changed.sum(), 1))} become dead ball, "
          f"{share(from_dead.sum() / max(changed.sum(), 1))} leave dead ball")
    print("VERDICT     " + ("the commentary can go in: make_submission.py --mentions ... --events ..."
                            if check["verdict"] == "on" else
                            "only the dead-ball reports would be used; the team evidence stays off"))
