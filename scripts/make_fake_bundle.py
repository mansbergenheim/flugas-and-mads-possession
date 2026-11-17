"""Write an open match as a fake competition data bundle, to dry-run the submission path.

Same folders, file formats, frame counts and half layout as the real bundle, including the dead
frames before the second-half kickoff. Tags are simulated. Contains no competition data.

It also writes a made-up broadcast (mtah/fakecast.py) in the broadcast folder: tracking on its own
timeline and a homography file that says which frames are wide shots. fake_broadcast_truth.npz holds
how the two timelines really line up, to check scripts/align_broadcast.py against.

    python3 scripts/make_fake_bundle.py J03WR9 /some/folder/outside/the/workspace
"""

import os
import sys
from xml.sax.saxutils import escape

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mtah import harness, submission as sub  # noqa: E402
from mtah.degrade import DegradeConfig, load_stats  # noqa: E402
from mtah.fakecast import fake_broadcast  # noqa: E402
from mtah.loaders import gated  # noqa: E402
from mtah.match import D, H, Match, X  # noqa: E402
from mtah.smart_tagging import simulate_tags  # noqa: E402

KICKOFFS = (11185, 113742)
GRADED_END = (90652, 190198)


def layout(match):
    """Open match placed on the competition's video timeline."""
    n, slots = gated.N_FRAMES, match.players.shape[1]
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    players = np.full((n, slots, 2), np.nan, dtype=np.float32)
    labels = np.full(n, X, dtype=np.int8)
    for (start, stop), kick, end in zip(match.halves, KICKOFFS, GRADED_END):
        length = min(stop - start, end - kick)
        ball[kick:kick + length] = match.ball[start:start + length]
        players[kick:kick + length] = match.players[start:start + length]
        labels[kick:kick + length] = match.labels[start:start + length]
    labels[labels == X] = X
    block = slice(110808, KICKOFFS[1])
    players[block] = players[KICKOFFS[1]]
    labels[block] = D
    return ball, players, labels


def to_unit(xy):
    out = np.empty_like(xy)
    out[..., 0] = xy[..., 0] / gated.PITCH[0] + 0.5
    out[..., 1] = 0.5 - xy[..., 1] / gated.PITCH[1]
    return out


def group(x, y):
    return "NaN,NaN,NaN" if x != x else f"{x:.5f},{y:.5f},0.00"


if __name__ == "__main__":
    match_id, root = sys.argv[1], sys.argv[2]
    gated.data_dir(root)  # same rule as the real data: never inside the workspace
    x = harness.prepared(match_id, DegradeConfig(), 2000, load_stats())
    ball, players, labels = layout(x)
    folder = os.path.join(root, "tactical")
    os.makedirs(folder, exist_ok=True)
    stem = os.path.join(folder, f"{gated.GAME}_tactical")

    ids = "".join(f'<Player id="p{i}" teamId="{"T1" if t == 0 else "T2"}"/>' for i, t in enumerate(x.team))
    with open(stem + "_atd.xml", "w") as f:
        f.write(f'<main><Teams><Team id="T1"/><Team id="T2"/></Teams><Players>{ids}</Players></main>')

    ub, up = to_unit(ball), to_unit(players)
    with open(stem + "_atd.txt", "w") as f:
        for i in range(gated.N_FRAMES):
            row = ";".join(group(a, b) for a, b in up[i].tolist())
            f.write(f"{i + 1}:{row}:{group(*ub[i].tolist())}\n")

    tags = simulate_tags(labels, np.random.default_rng(0))
    for kick, name in zip(KICKOFFS, ("1st Half", "2nd Half")):
        tags.append({"code": "SET PIECES", "start": kick - 88, "stop": kick + 37,
                     "tags": {"Team": "Barcelona", "Half": name, "Type": "Kick Off (Start)"}})
    with open(stem + "_smart_tagging.xml", "w") as f:
        f.write("<file><ALL_INSTANCES>")
        for i, t in enumerate(tags):
            parts = "".join(f"<label><group>{escape(k)}</group><text>{escape(v)}</text></label>"
                            for k, v in t["tags"].items())
            f.write(f"<instance><ID>{i}</ID><start>{t['start'] / 25}</start><end>{t['stop'] / 25}</end>"
                    f"<code>{escape(t['code'])}</code>{parts}</instance>")
        f.write("</ALL_INSTANCES></file>")

    sub.write_submission(os.path.join(root, "fake_gt.csv"), labels)

    # The broadcast: made from the undegraded match on the same video timeline, so that its tracking
    # errors are not the tactical recording's errors.
    clean = harness.prepared(match_id, None, 0, load_stats())
    cball, cplayers, clabels = layout(clean)
    spans = [(kick, kick + min(stop - start, end - kick)) for (start, stop), kick, end in zip(clean.halves, KICKOFFS, GRADED_END)]
    cast = fake_broadcast(Match(match_id, clean.source, 25, gated.PITCH, cball, cplayers, clean.team, spans, clabels))
    extra = gated.N_BROADCAST_FRAMES - len(cast.live)
    if extra < 0:
        sys.exit("the made-up broadcast is longer than the real one: pick a shorter match")
    shown = np.concatenate((cast.players, np.full((extra,) + cast.players.shape[1:], np.nan, dtype=np.float32)))
    shown_ball = np.concatenate((cast.ball, np.full((extra, 2), np.nan, dtype=np.float32)))
    wide = np.concatenate((cast.valid, np.zeros(extra, dtype=bool)))
    folder = os.path.join(root, "broadcast")
    os.makedirs(folder, exist_ok=True)
    stem = os.path.join(folder, f"{gated.GAME}_broadcast")
    ids = "".join(f'<Player id="p{i}" teamId="{"T1" if t == 0 else "T2"}"/>' for i, t in enumerate(cast.team))
    with open(stem + "_atd.xml", "w") as f:
        f.write(f'<main><Teams><Team id="T1"/><Team id="T2"/></Teams><Players>{ids}</Players></main>')
    ub, up = to_unit(shown_ball), to_unit(shown)
    with open(stem + "_atd.txt", "w") as f:
        for i in range(gated.N_BROADCAST_FRAMES):
            row = ";".join(group(a, b) for a, b in up[i].tolist())
            f.write(f"{i + 1}:{row}:{group(*ub[i].tolist())}\n")
    with open(stem + "_homography.csv", "w") as f:
        f.write("frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n")
        for i in range(gated.N_BROADCAST_FRAMES):
            f.write(f"{i},1,0,0,0,1,0,0,0,1,0.3,1\n" if wide[i] else f"{i},NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,0\n")
    np.savez_compressed(os.path.join(root, "fake_broadcast_truth.npz"),
                        live=np.concatenate((cast.live, np.full(extra, -1, dtype=np.int32))),
                        state=np.concatenate((cast.state, np.zeros(extra, dtype=np.uint8))))
    print(f"fake bundle for {match_id} in {root}: {len(tags)} tags, {int((labels != X).sum())} graded frames")
