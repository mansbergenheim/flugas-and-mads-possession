# Possession pipeline

A trained possession model, the one-command submission script, and the harness that validates it
**without touching the competition match**.

## What the submission builds since 12 November

`make_submission.py` builds **the trees and the TCN, averaged** (`mtah/candidate.py`):

- **The trees.** Three tree models with 198 inputs: v5's 129 and 69 more about how the evidence
  changes over time. Each is trained on its own random draw of the damaged training copies, and the
  three are averaged.
- **The TCN.** A small network over the same inputs that reads about six seconds either side of each
  moment. It was trained on Google Colab from the same ten open matches, and its weights ship in
  `models/tcn.npz`.
- **The mix.** The plain mean of the two sets of probabilities, then v5's decoder with v5's settings.
  Nothing about the mix is fitted.

It replaced v5 by a rule fixed before the networks had a result: at least +0.015 on the 10 open
matches with 8 of 10 up, and no loss in four other tests. spellF1, every match scored by models that
never trained on it:

| | 10 open matches | Stricter labels | Stress table | Unseen label source | SkillCorner's working half |
|---|---|---|---|---|---|
| v5 | 0.634 | 0.688 | 0.616 | 0.588 | 0.340 |
| The trees | 0.646 | 0.697 | 0.628 | 0.601 | 0.351 |
| The TCN alone | 0.707 | 0.718 | 0.687 | 0.662 | 0.349 |
| **The trees and the TCN, averaged** | **0.699** | **0.732** | **0.681** | **0.646** | **0.370** |

It is ahead of v5 on every match in each of these tests, and on all 57 copies of the stress table
(`experiments/candidate_against_v5.py`). Seventeen other ways of using the trained models were tried
and none is clearly better (`experiments/compare_networks.py`, `experiments/combine_rules.py`).

A dry run on 12 November put an open match through `make_submission.py` in the competition's file
formats. The file it wrote, with the trees trained afresh, is identical byte for byte to the one made
from the saved models that every test above used. So the script builds the model that was tested.

The rest of this README was written for v5, which is commit `e90d19e`. Its numbers are right for v5,
and it is being rewritten for the new model. Still to do for the new model: the fingerprints on a
second machine, and the one check on SkillCorner's reserve half.

## Results

Ten matches, each scored by a model that never trained on it, with decoder settings chosen on the
other matches' held-out predictions:

| Model | spellF1 | DFL | Metrica | macroF1 |
|---|---|---|---|---|
| Organisers' baseline (`nearest:50`) | 0.239 | | | 0.566 |
| Hand-built rules (`RuleModel`) | 0.312 | 0.302 | 0.334 | 0.700 |
| **Learned model** | **0.630** | 0.632 | 0.625 | 0.882 |

Every match scores between 0.60 and 0.66. A dry run through `make_submission.py` on a held-out match
in the competition's file formats scored 0.689 (5-95% interval 0.650 to 0.734). The previous version's
run from a fresh `git clone` produced a byte-identical file. Repeat that check on this version once it
is committed.

**These aren't competition scores.** They come from degraded open matches under neutral test
conditions, and open-match labels aren't the competition's labels. On the real match the organisers
publish spellF1 0.2433 and macroF1 0.672 for this baseline, against 0.239 and 0.566 here. That's a
comparison only: no setting is fitted to it. A plausible real range is 0.45 to 0.60, which is an
estimate, not a measurement. It leans on the real ball being usable: see the stress table.

### How far to trust these numbers

- **Its quality on the real match is unknown.** On 7 Nov it ran end to end on the real files: they
  loaded, and the output passed the organisers' validator and a sanity check on the public 5 minutes.
  That shows the pipeline works, not how good the labels are. This is the largest unknown by far.
- **Decoder selection isn't strictly nested.** The models behind the held-out predictions trained on
  the scored match, so a little can leak. On these matches the whole grid spans only 0.623 to 0.634,
  and the gap between the best setting and the reported score is 0.004. That describes this grid on
  this data. It isn't a guarantee for other matches.
- **Features were chosen on these matches.** They were screened on 3 of the 10, after studying misses
  on all 10. The gain held on the other 7 (+0.058, against +0.068 on the 3), but it isn't a clean test.
- **An earlier version used published scores.** Two degrader settings were fitted to them, and that
  test data guided the choice of features, weighting and decoder. That's removed: the decoder was
  re-picked and the weighting re-checked without it (0 gives 0.606; 0.25, 0.5 and 0.75 give 0.635,
  0.638 and 0.641). The feature design dates from that period. Disclose this in your document.
- **The training ranges were widened after measuring the real tracking.** `compare_tracking.py` showed
  the real tracking was cleaner than every training copy on most measurements, so the ranges in
  `random_config` now reach cleaner tracking, and most of those measurements now sit inside them. That
  comparison uses no answers and no scores, and the test conditions were moved towards it too. Disclose
  this in your document.
- **The wider range is a small gain, under the organisers' noise band.** On identical test data, each
  range with its own decoder pick, it scored 0.630 against 0.615, with 8 matches up and 2 down. A paired
  check puts the chance of that arising by luck at about 4%, though it's the same 10 matches and one
  seed. The gain is nearly all DFL (0.612 to 0.632). Metrica barely moved (0.622 to 0.625), and Metrica's data is probably
  the nearer match to the competition's. It stays because it covers the tracking the model will actually see,
  not because of that score. `experiments/with_old_ranges.py` reruns anything on the old range.
- **Every result traces to a run.** `runs/oof/v5/manifest.json` records the commit, source hash,
  settings, package versions, open-data hashes and a hash of each match's predictions. v5 ran before
  its code was committed, so the manifest is marked dirty and its source hash is the link. The scores
  are in `runs/oof/v5/sweep_prob.json`, the stress test in `runs/stress_test.json`, and the earlier
  version's reproduction check in `runs/final_checks_log.txt`.

### What lifted short possessions

The first version found 21% of possessions under 4 s, mostly with the ball in plain view. Each row
below was screened on 3 held-out matches, then the kept changes were confirmed on all 10, where all 10
improved (0.578 to 0.640, mean +0.061). These ran under the earlier test conditions, so they compare
with each other and not with the table above. The code for rejected rows was removed before this
folder had git, so only their logs remain, in `runs/dev_log_*.txt`. The two event-label rows are the exception: they ran
under the current test conditions, and `experiments/dfl_event_labels.patch` restores their code.

| Change | spellF1 on 3 matches | Kept? |
|---|---|---|
| First version | 0.589 | |
| Touch sequence, short windows, block-averaged ball | 0.634 | Yes |
| Plus weighting each sample by its possession's length, strength 0.5 | 0.661 | Yes |
| Weighting at strength 0.75, best decoder | 0.664 | No, a tie |
| Weighting at strength 1, best decoder | 0.656 | No, too many possessions |
| Kick detection | 0.634 with or without | No |
| Taking out DFL's label delay (train DFL, test Metrica) | 0.608 vs 0.603 | No, a tie |
| Larger trees / 5 training copies | 0.669 / 0.668 vs 0.661 | No, ties costing 1.5 to 2 times the training |
| DFL labels rebuilt from events (train DFL, test Metrica) | 0.581 vs 0.612 | No, worse |
| DFL's timing, events only removing unsupported short spells (same test) | 0.617 vs 0.612 | No, a tie |

Share of true possessions found, averaged over the metric's IoU levels, all 10 matches. "After" is on
the same test data as "Before". "Now" is the final model under the current test conditions.

| Length | Share of possessions | Before | After | Now |
|---|---|---|---|---|
| Under 1 s | 6% | 1% | 4% | 6% |
| 1 to 2 s | 16% | 13% | 22% | 21% |
| 2 to 4 s | 22% | 33% | 48% | 46% |
| 4 to 8 s | 22% | 60% | 70% | 68% |
| 8 to 16 s | 18% | 75% | 82% | 81% |
| 16 s and over | 16% | 84% | 86% | 86% |

### If the real tracking is worse than assumed

`scripts/stress_test.py`, one problem at a time on 3 held-out matches. Both columns use the same test
data, so they compare directly. The left one came from `experiments/with_old_ranges.py`:

| Condition | Earlier narrow range | Current |
|---|---|---|
| As trained | 0.644 | 0.656 |
| Ball always visible (ceiling for any ball detector) | 0.692 | 0.712 |
| Ball noise 1.5 m | 0.616 | 0.612 |
| Ball noise 3 m | 0.588 | 0.569 |
| 15% false ball detections | 0.621 | 0.645 |
| Player noise 2 m | 0.603 | 0.604 |
| 1 m white noise | 0.635 | 0.632 |
| 15% tracks on the wrong team | 0.601 | 0.607 |
| Many more blackouts | 0.620 | 0.631 |
| Ball visible in most stoppages | 0.653 | 0.668 |
| Ball hidden in most stoppages | 0.631 | 0.666 |
| No ball at all | 0.479 | 0.511 |
| Several problems at once | 0.553 | 0.546 |
| Everything much worse | 0.448 | 0.446 |

The model leans on the ball. A perfect ball is worth +0.056, and no ball at all costs 0.145. The wider
range gives up a little under heavy ball noise, 0.019 at worst, and gains elsewhere. The real ball
measures smooth, but smooth isn't the same as accurate: a ball that is steadily in the wrong place
would pass that check and still cost what these rows show.

## How it works

1. **Features at 5 Hz** (`mtah/features.py`): ball evidence when present, team shape and movement
   always, memory of who was near the ball either side of a blackout, and the sequence of which team
   played the ball last, before that, and next. Speed is measured over 1.2 s on block-averaged
   positions and also as a rank within the half, so noise can't hide stoppages.
2. **Frame model** (`mtah/model.py`): gradient-boosted trees giving P(H), P(A), P(D). Trained on open
   matches at randomly drawn tracking quality, home and away swapped too, and averaged over both views.
   Samples in short possessions weigh more, because the metric counts possessions, not frames.
3. **Decoder** (`mtah/decoder.py`): Viterbi with a cost per change, so stray touches don't become
   possessions.
4. **Smart Tagging** (`mtah/smart_tagging.py`): a small prior in the decoder, and the kickoff tags mark
   the dead frames before each half with no hard-coded frame numbers. Overlapping tags count once,
   because they come from one automatic inference and share their errors. In simulation it adds 0.003
   to 0.007 and doesn't hurt with bad tags. Its real value is unknown.

## Quick start

Built on Python 3.14.3 on an M2 Mac with 8 GB. Use Python 3.14 and the pinned packages, so a rebuild
matches ours.

```bash
pip install -r requirements.txt
```

```bash
python3 -m unittest tests.test_metrics tests.test_pipeline tests.test_skillcorner
```

`data_open/` is git-ignored. On a new machine, fetch it (about 2.8 GB). Both sources are pinned to the
versions the model was trained on, and every file is checked against `mtah/open_data_sha256.json`:

```bash
sh scripts/download_open_data.sh
```

The SkillCorner checks also need its open matches (about 2 GB, MIT licence, credit SkillCorner). The
submission doesn't:

```bash
sh scripts/download_skillcorner.sh
```

Three tests need `../official-repository` and `../research/verification` beside this folder, and skip
without them.

This folder is a local git repository with no remote yet. Push it to your private GitHub repo.

## Run these once yourselves

Claude couldn't run the organisers' code here, so this direct comparison is still open:

```bash
MTAH_OFFICIAL_REPO=../official-repository python3 -m unittest tests.test_official_parity
```

## Making a submission

With the match data on your machine, in your own terminal, not through an AI tool. It trains three
tree models, reads the TCN from `models/tcn.npz`, and prints counts only. It takes just under half an
hour on the build Mac:

```bash
MTAH_DATA_DIR=~/mtah-data python3 scripts/make_submission.py --team YOUR_TEAM_NAME --out ~/mtah-data/submissions
```

It checks the inputs first and stops if the files don't match `DATA.md`: every frame exactly once, the
right slots, believable ball and player counts, and a kickoff tag in each half. It also stops if any
open file is missing or differs from the pinned version, then rebuilds the open matches from the raw
files, so an old cache can't change the result. Those messages give
counts and names only, so they are safe to share. Commit before you run: the file is written with a
`YOUR_TEAM_NAME.manifest.json` beside it recording the commit, settings and hash.

Then check the exact file with the organisers' validator:

```bash
python3 ../official-repository/evaluation/validate_submission.py ~/mtah-data/submissions/YOUR_TEAM_NAME.csv
```

Score it on the public 5 minutes **as a pipeline check only, never to choose between versions**:

```bash
mkdir -p ~/mtah-data/public_check && python3 ../official-repository/evaluation/run_evaluation.py --results ~/mtah-data/submissions --out ~/mtah-data/public_check
```

Keep `--out` on both commands. Without it, the submission and the scorer's frame-level reports land
inside this workspace, where AI tools can read them.

A spellF1 near zero there means something is broken, most likely home and away swapped. Don't read
anything else into that score, and don't change the model because of it: that's tuning on the window.

Measure the real tracking too:

```bash
(cd ~/mtah-data && MTAH_DATA_DIR=~/mtah-data python3 "$OLDPWD/scripts/atd_summary.py")
```

It writes aggregates to `runs/gated/atd_summary.json` and prints nothing else. Then compare the real
tracking with the training copies, which prints words only and writes nothing:

```bash
MTAH_DATA_DIR=~/mtah-data python3 scripts/compare_tracking.py
```

Done on 7 Nov. What's left outside the range is ball and player wobble, on the clean side and close to
the noise floor, so they aren't worth chasing.

## Three more checks of the real match

`compare_tracking.py` compares 11 measurements of the tracking. These go further, and like it they use
no answers, print words only and write nothing. Run them in your own terminal, about 10 minutes:

```bash
MTAH_DATA_DIR=~/mtah-data python3 scripts/check_real_match.py
```

1. **Ball.** How much of the visible ball is a perfectly straight, evenly spaced line? A ball followed
   by cameras never is (0% in the DFL matches). Metrica's own sample games are straight lines for 43
   to 63% of the moving ball, and the organisers' public baseline shows almost no holes between half a
   second and five seconds, so short losses are probably bridged. The simulator doesn't do that.
2. **Inputs.** For each of the model's inputs, is the real match inside what the 30 practice copies
   cover? Only inputs far outside are listed. A faithful fake match lists none; a fake with the home
   goalkeeper tracked as an away player lists 14.
3. **Output.** Is the model as sure of itself on the real match as on open matches it had not seen,
   with the ball in view and hidden? It uses `runs/models/v5_final_model.pkl` if present, else trains.

Run on 10 Nov: no straight lines in the ball, no input far outside, and a model that is less sure
than in practice with the ball in view and more sure with it hidden. A follow-up looks at why, also
in words only and in about a minute:

```bash
MTAH_DATA_DIR=~/mtah-data python3 scripts/check_real_match.py --only followup
```

It lists every input that is outside the practice copies and which way, splits the ball-in-view time
by what is happening at the ball (at one player's feet, two players at it, travelling fast, loose) and
says how sure the model is in each, asks whether the ball turns where the players are, and repeats the
straight-line test in the camera picture using the homography file. A match that is simply more
contested shows up as more time in the hard situations. A ball that misleads shows up as doubt in the
easy ones.

The reference values come from open matches only (`mtah/check_ranges.json` and
`mtah/check_followup.json`, rebuilt by `scripts/fit_check_ranges.py` and
`scripts/fit_check_followup.py`). What to do with the answers is in
`../research/OPTIMISATION_REVIEW_2026-11-10.md`.

## Checking the candidate on the real match

The submitted model is the plain average of the trees (the extra inputs, three draws averaged) and
the TCN. This check builds it from saved models, so without the half hour of training, and runs it on
the real match beside the saved v5, in your own terminal, in about a minute. It prints words only:

```bash
MTAH_DATA_DIR=~/mtah-data python3 scripts/check_candidate.py
```

1. **File.** The candidate's labels, made the way `make_submission.py` makes v5's, against our own
   checks. A test file is written to `~/mtah-data/candidate_check/`, outside this workspace.
2. **Members.** How often the trees and the TCN name a different class at the same moment. On open
   matches they had not seen it is 5 to 8 moments in 100, and on SkillCorner 13 to 21.
3. **Labels.** How many frames the candidate labels differently from v5 (3.5 to 5% on the open
   matches, 6 to 9% on SkillCorner).
4. **Output.** How sure the candidate is of itself, the questions asked of v5 above.
5. **Public window.** The candidate against v5 on the public five minutes, with our copy of the
   scorer. It answers one thing: is something broken? On the open matches the candidate is 0.064
   ahead of v5 over whole matches and still behind v5 in one five-minute window in seven (26 of 184).
   So the line says NORMAL unless the candidate is further behind v5 than in 19 of 20 windows there,
   and it says the same whether the candidate is ahead or a little behind. **Never choose between
   versions on this window.**

With `--score` added, part 5 also prints the two scores themselves: spellF1 on the public five
minutes for the candidate and for v5, from our copy of the scorer, each marked `[a number]`. Those two
lines are measurements of the real match. Keep them on this computer and out of AI tools. The saved
v5's labels are written beside the candidate's as `yardstick_v5.csv`, so the organisers' scorer can
be pointed at that folder for their own figure, as in "Making a submission" above.

To follow the model against the pictures, `scripts/show_spells.py` lists what a labels file says on
the public five minutes as possessions with their times: the match clock, the place in the tracking
timeline, who has the ball and for how long. With `--answers` it shows the organisers' public labels
beside ours. Its output is about the real match, so it stays on your computer, and nothing seen there
may change the model. Watching the match to understand is allowed by the rules; labelling by hand is
not.

The reference values come from open matches only (`mtah/check_candidate.json`, rebuilt by
`scripts/fit_candidate_ranges.py`). A fake bundle made from an open match the models were trained on
gives the answers one would expect: less disagreement than on any open match, a model surer than on
any open match, and NORMAL. With home and away exchanged in the labels it says BROKEN.

## Lining the broadcast up with the tactical recording

Not part of the submission. It is groundwork for using the broadcast, which DATA.md says is on its
own timeline with no mapping given.

    python3 scripts/align_broadcast.py

Run it yourself on the real match. It reads the two sets of tracking and the broadcast's homography
file, never a video, and takes about half a minute. It prints words only, so the printout can be
pasted anywhere. It writes `<data folder>/alignment/broadcast_alignment.npz`, which is about the real
match and stays on that computer: for each tactical frame, the broadcast frame that shows the same
moment, and whether the broadcast was then on its live wide shot, on a replay, on a wide shot with
too few players to tell, or on something with no pitch transform (a close-up, a crowd shot).

How it works (`mtah/align.py`): while the broadcast shows the live wide view, its tracking places the
players it sees on the pitch, and the tactical tracking places everybody on every frame. Short clips
of the broadcast are matched against every tactical moment and vote for the shift between the two
timelines. The broadcast runs in real time between cuts, so true shifts repeat and wrong ones do not.
Each piece is then fitted to the frame. The aligner also finds which way round the broadcast draws
the pitch, whether the two files number the teams the same way, and undoes a pitch that the
broadcast draws slightly stretched or off centre.

What has been tested: no open match comes with a broadcast, so one is made up from each
(`mtah/fakecast.py`): a camera that shows the pitch around the ball, close-ups and replays at
stoppages, the half-time cut out, its own tracking errors. On six such set-ups from four suppliers'
matches, drawn the same way round, turned and mirrored (`experiments/align_check.py`):

| | |
|---|---|
| Line-ups found | 6 of 6, each half as one piece |
| Frames given exactly the right shift | 100% |
| Truly live wide frames recognised as live | 85% to 100% |
| Frames called live that are live | 99.9% to 100% |
| Replays not taken for live | 95% to 100% |
| Time | about 23 seconds |

With far fewer wide shots, noisier broadcast tracking and a pitch drawn 4% too long and 2 m off, the
shift was still exact. The whole path was also run on a fake bundle in the competition's file
formats (`scripts/make_fake_bundle.py` now writes a made-up broadcast too) and matched the known
truth on every frame.

What this does not show: that the real broadcast behaves like the made-up ones. The first real
evidence is the words the script prints on the real match. Nor does it show what a director's cuts
or the broadcast's ball are worth to the score: the made-up director follows rules written here.

## Checking another machine builds the same file

The rules ask for a repo that regenerates the exact same CSV, and the organisers will probably rerun
it on an Intel or AMD machine with Linux. Ours was built on an Apple chip. So before submitting, a
teammate with a Linux PC, or Windows with WSL (Ubuntu), runs two checks on the same commit.

**1. Open data only, safe to share anywhere.** About half an hour after the download (27 minutes on
the build Mac):

```bash
git clone <your repo URL> mtah && cd mtah && python3 -m venv .venv && . .venv/bin/activate
```

```bash
pip install -r requirements.txt && sh scripts/download_open_data.sh && python3 scripts/repro_check.py
```

The last five lines must match the build Mac's (12 November, the trees and the TCN):

```text
test copy     8a1175fa39ad97cc
trees         dc8b1af3e77046ae
tcn           f602821ad3381bd7
probabilities b4048da20b2b3da1
labels        5bb1373d7775153e
```

If one differs, the first line that differs says where. `test copy` is the degrader. `trees` is the
features or the training of the tree models. `tcn` is the weights file or the numpy arithmetic that
runs it. `probabilities` is the mean of the two, and `labels` is the decoder. These lines come from
open data only, so paste them to anyone, Claude included. The full output of the build Mac's run is
in `runs/repro_check_candidate.txt`.

For v5, at commit `e90d19e`, the lines were `test copy 8a1175fa39ad97cc`, `probabilities
f616b947cd112398` and `labels caaa7aa3cea176b1`.

**2. The real file, in your own terminal.** Run `make_submission.py` with the same team name as on the
Mac, then compare the `sha256` it prints with the Mac's. Compare the two hashes yourselves, by eye.

What's already known, from `experiments/determinism_check.py` (`runs/determinism_check.txt`):

- The number of cores and Python's hash seed change nothing: 1, 3 and 8 threads give identical bits.
- Prediction and decoding are robust. Moving every second test feature by the smallest possible step
  changed 10 to 25 frames in 145,000, and moving the probabilities by one part in 10^12 changed none.
- Training is not. Moving just one training feature value in 10,000 by that step changed 1.8% of the
  frames, though not the score (within 0.006). So a single value computed differently on another chip
  can change the file. Moving the sample weights the same way changed nothing.
- So the calculations that feed the training features directly now give the same bits on any chip:
  distances use `sqrt(x*x + y*y)` rather than the system's `hypot`, and the noise draws use exact
  fused multiply-adds and Python's maths library. On this Mac that changed nothing (same fingerprints
  before and after). A few steps still work in float64 with library maths (scipy's noise filter,
  numpy's rare tail draws), but they're rounded to float32 before use, which absorbs a last-bit
  difference almost every time. scikit-learn's own maths is left, which the weight test suggests
  doesn't matter.

If check 1 differs on the teammate's machine even so, the fallback is to ship the trained model, since
prediction is robust. Ask before doing that: it changes what "regenerates" means for the organisers.

## Changing the model

Commit first, so the run's manifest points at exact code. Train held-out models, about 50 minutes,
then explore decoder settings in about 2 minutes:

```bash
python3 scripts/train_oof.py --tag v6
```

```bash
python3 scripts/sweep_decoder.py --tag v6
```

Then check robustness, about 6 minutes:

```bash
python3 scripts/stress_test.py
```

The 0.04 noise band is the organisers' figure for two scores on one match, so `harness.compare` is a
strict default, not a statistical test. For a paired comparison, also look at how many matches moved
and by how much. Judge a change on the stress table as well as the first row: the first version here scored 0.625 as trained
but fell to 0.445 with 1 m of noise. Settings used by the submission live in `mtah/config.py`.

## The v6 candidate (not adopted as a whole)

This section is the record of the v6 test of 10 and 11 November, when `make_submission.py` built v5.
Two of its three parts, the extra inputs and the three models averaged, are now the tree half of the
submitted model. Its smoother is not used.

The candidate added three things to v5, all off by default, and with them off the v5 fingerprints
were unchanged (`runs/v6_repro_v5.txt`):

- **Extra inputs** (`build_features(extra=True)`): the evidence a moment earlier and later, whether the
  ball moves with a player or past him, and the closest approach at the full frame rate.
- **Three models averaged** (`model.AveragedModel`): one model is one random draw, and draws differ.
- **A learned smoother** (`mtah/smoother.py`): a second model over the first one's probabilities in
  time. It rounds them to four decimals first, so it rebuilds the same on any chip.

The rule was fixed on 10 Nov, before any 10-match result: v6 replaces v5 only if the mean gain on the
10 open matches is at least 0.015 with at least 8 up, no stress row is more than 0.01 below v5, and the
lead over the rules on SkillCorner's working half is not smaller. To run it, about five hours:

```bash
python3 scripts/train_oof.py --tag v6_d0 --extra --draw 0 && python3 scripts/train_oof.py --tag v6_d1 --extra --draw 1 && python3 scripts/train_oof.py --tag v6_d2 --extra --draw 2
```

```bash
python3 experiments/evaluate_v6.py && python3 experiments/stress_v6.py && python3 experiments/skillcorner_v6.py
```

**Result: not adopted.** Three parts passed and one failed.

| Part of the rule | Result | |
|---|---|---|
| Mean gain of at least 0.015 on the 10 open matches | 0.655 against 0.630, +0.025 | Passed |
| At least 8 of 10 matches up | 10 up | Passed |
| No stress row more than 0.01 below v5 | 18 of 19 rows above v5, but "very harsh" 0.422 against 0.446 | **Failed** |
| Lead over the rules on SkillCorner's working half not smaller | +0.087 against +0.085 | Passed, a tie in practice |

**Why, piece by piece.** The candidate was decoded with a setting re-picked for it, so the first table
mixes the models with the decoder. The follow-ups below give every piece v5's own decoder setting.
Gains are against v5:

| | Smoother alone, on v5 | Extra inputs, 3 models averaged, no smoother | Whole candidate |
|---|---|---|---|
| 10 open matches | +0.013, 9 up, 1 down | +0.012, 9 up, 1 down | +0.027, 10 up |
| Looser labels, DFL's raw flags, 7 matches | +0.012, 6 up, 1 down | +0.009, 6 up, 1 down | +0.027, 7 up |
| Stress table, 19 rows | mean +0.004, 11 up, 8 down, worst −0.013 | mean +0.013, 17 up, 2 down, worst −0.002 | mean +0.016, 18 up, 1 down, worst −0.021 |
| A label source it never saw (DFL to Metrica and back) | −0.007, 3 up, 7 down | +0.012, 8 up, 2 down | −0.001, 5 up, 5 down |
| Stricter labels, single touches under 4 s folded | −0.008, 3 up, 7 down | +0.009, 8 up, 2 down | −0.002, 5 up, 5 down |
| SkillCorner's working half | +0.001, 6 up, 4 down | +0.011, 10 up | +0.004, 7 up, 3 down |

So the candidate is two halves. The extra inputs with three models averaged are worth about 0.01
everywhere we can test, and their worst row in any of these tables is 0.002 below v5. That does not
depend on the decoder setting: given the same setting as v5, they are ahead at all 175 settings of the
grid on the 10 open matches and on SkillCorner's working half. The smoother adds another 0.015 to 0.02
on labels like ours or looser under mild damage, and gives it back whenever conditions differ from
practice: heavy damage, a label source it has not seen, stricter labels, SkillCorner. It is the cause of the failed stress row.
The re-picked decoder setting is not: with v5's setting the candidate is still 0.021 below v5 there.
The half without the smoother is under the 0.015 bar, so by the rule it is not adopted either.
Both of its parts matter: three v5 models averaged with no new inputs gain 0.004 on the 10 open matches
and 0.012 on SkillCorner, and the extra inputs add the rest on the open matches and nothing on
SkillCorner.

| Script | What it answers |
|---|---|
| `experiments/v6_same_decoder.py` | The 10-match table with v5's decoder setting for every piece |
| `experiments/stress_v6_parts.py` | The stress table by piece, with both decoder settings |
| `experiments/skillcorner_v6_parts.py` | SkillCorner's working half by piece, with both decoder settings |
| `experiments/source_held_out_v6.py` | Trained on DFL and tested on Metrica, and the reverse |
| `experiments/v6_label_styles.py` | The same predictions scored against looser and stricter labels |
| `experiments/v6_no_smoother_grid.py` | v5 against the candidate without its smoother at every decoder setting |
| `experiments/v6_averaging_only.py` | A control: three v5 models averaged, with no new inputs |

Until 12 November `make_submission.py` built v5 only. Since then it builds the half of this candidate
without the smoother, averaged with the TCN: see the top of this README.

## Layout

| Path | What it does |
|---|---|
| `mtah/match.py` | The shared match format: metres, centre-spot origin, label codes |
| `mtah/metrics.py` | spellF1, macroF1 and more, pinned to official output |
| `mtah/submission.py` | Writes and checks the CSV; `force_no_abstain` removes graded `X` |
| `mtah/loaders/dfl.py`, `metrica.py` | Open matches. Metrica labels are derived from events |
| `mtah/loaders/skillcorner.py` | SkillCorner's open matches, for testing only. Not in the submitted model's training |
| `mtah/loaders/gated.py` | The competition match. Refuses data stored inside this workspace |
| `mtah/labels.py` | Folds touch-level flips out of DFL labels |
| `mtah/degrade.py` | The degrader |
| `mtah/harness.py` | Leave-one-match-out validation, randomised training sets, paired comparison |
| `mtah/features.py`, `model.py`, `decoder.py` | The tree models and the decoder |
| `mtah/candidate.py` | The submitted model: three tree models and the TCN, their probabilities averaged |
| `mtah/align.py`, `mtah/fakecast.py` | Lining the broadcast up with the tactical recording, and the made-up broadcast it is tested on. Not used by the submission |
| `mtah/network.py`, `models/tcn.npz` | The TCN: prediction in plain numpy, and the trained weights with their pinned hash. See `models/README.md` |
| `colab/`, `scripts/export_network_pack.py`, `scripts/import_network_results.py` | How the TCN was trained on Colab from the open matches, and how its test predictions came back. A BiLSTM trained the same way was not adopted. See `colab/README.md` |
| `mtah/smart_tagging.py` | Tag prior and simulated tags for sizing it |
| `mtah/config.py` | The settings the submission uses |
| `mtah/provenance.py` | The manifest saved with every run |
| `scripts/make_fake_bundle.py` | Writes an open match as a fake competition bundle, for dry runs |
| `scripts/dev_split.py` | Quick screen of a change on 3 held-out matches in 5 conditions |
| `scripts/diagnose_spells.py` | Which possessions were missed, by length, and why |
| `scripts/compare_tracking.py` | Words-only check of the real tracking against what the model trained on |
| `scripts/fit_tracking_ranges.py` | Measures the training copies for that check. Open data only |
| `scripts/repro_check.py` | Fingerprints of the model built from open data, to compare machines |
| `mtah/open_data_sha256.json` | SHA-256 of every open file the model trains on |
| `scripts/` | Also download, sweeps, stress test, ATD summary |
| `experiments/` | Kept out of the model: the old training range, the DFL event-label patch, the short-possession breakdown, the learning curve, the SkillCorner checks, the tests of the v6 candidate |

## Rules this code is built around

- **Match data stays out of AI tools.** Keep it outside this workspace and run gated code yourself.
- **Never choose between versions on the public 5 minutes.** Choose here.
- **Don't copy public-window statistics into a model.** Learn priors from the open matches.
- **Credit the data in your document:** DFL with Bassek et al. 2025 (CC BY 4.0), and Metrica Sports.
  Explain the label derivation, the 1.0 s filter, that an earlier version used published scores, and
  that the training ranges were widened after a words-only comparison with the real tracking.
- `.gitignore` blocks data, predictions and CSVs, but check your history before making the repo public.

## Known limits

- **Many missed short possessions may not count in the competition.** On open matches 38% of
  possessions under 4 s show a single touch or none, and the organisers' definition says a touch
  without control doesn't change possession. Our labels keep them. Giving those spells back to the
  previous team lifts the same predictions from 0.634 to 0.688, and the predicted count from 0.80 to
  1.01 of the true count. That's a what-if, not a better model: the public window holds a 0.76 s
  possession, so the competition's labels aren't fully strict either. See
  `experiments/short_possessions.py`, which sorts by rules of thumb on 2D tracking.
- **Training on those stricter labels didn't help, so the labels stay as they are.** On all 10 matches
  a model trained on them scored 0.695 against the current model's 0.691 on the stricter labels (6 up,
  4 down), and 0.615 against 0.630 on the current labels (1 up, 9 down). The rule, set before the
  result, was to adopt only if the gain clearly beat the loss. So this gap changes how to read the
  score, not what to train on. See `experiments/compare_definition_labels.py`.
- **The misses that do show control are mostly the model's, not the tracking's.** Perfect tracking
  rescues about 1 in 5 of them (`experiments/clean_tracking_check.py`). It also lifts 3 matches from
  0.656 to 0.744 overall, nearly all from the ball, so tracking matters, but not mainly for these.
- **It still misses very short possessions.** It predicts 0.82 of the true count and finds about 17%
  of those under 2 s. Part of this is DFL's labels: on clean tracking the team nearest the ball
  matches them only 53 to 65% of the time in possessions under 2 s, against 69 to 81% for labels
  rebuilt from events. But training on the rebuilt labels made the model worse on Metrica, probably
  because their timing is less consistent. That explanation is untested.
- **DFL labels run 0.25 to 0.7 s behind the tracking**, and Metrica's about 0.1 s or less. Correcting
  this made no measurable difference, so the code leaves it, but say so in your document.
- **More matches of the same kind would barely help.** Training on 3, 6 and all 9 other matches
  scores 0.611, 0.625 and 0.630, about +0.009 per doubling, so doubling the open data would add about
  that, under the 0.01 bar set beforehand. See `experiments/learning_curve.py`.
- **Part of the lead is learned labelling style, which may not carry to the competition.** On 10 of
  SkillCorner's open matches (labels from its algorithms) the model scores 0.340 against 0.255 for
  the rules and 0.199 for the baseline: ahead on all 10, but by 0.085 instead of 0.32. Its sparse
  broadcast tracking explains little of that: our own matches cut to the 10 players nearest the ball
  and rebuilt from 10 fps still keep a 0.26 lead. So most of the drop is SkillCorner's labels or
  league. The competition's labels are a style we can't see, so this is a risk to the real score, not
  a measurement of it. See `experiments/skillcorner_check.py` and `experiments/broadcast_check.py`.
  Adding those 10 matches to training didn't help on our own: 0.622 against 0.630, 3 matches up and
  7 down (`runs/skillcorner_train.txt`), so the submitted model doesn't use them. Nor did they help on
  a labelling style the model hadn't seen: trained on DFL and tested on Metrica, or the reverse,
  adding them cost 0.015 and 0.006 (`experiments/source_held_out.py`). That run also eases the worry a
  little: trained on DFL alone, the model scores 0.629 on Metrica, as well as when it also sees
  Metrica (0.625). Trained on Metrica's 3 matches alone it scores 0.577 on DFL, against 0.611 in the
  learning curve for any 3 of the other matches, a cost of about 0.03 for a style it never saw. So
  between these two styles the lead mostly carries over, and
  SkillCorner's algorithmic labels may be the odd one out. That is two styles, not proof.
  SkillCorner's other 10 matches (`datasets.SKILLCORNER_RESERVE`) are untouched, kept for one final check.
- **It depends on ball quality.** With no ball at all it scores 0.511.
- **How often stoppages hide the ball is a guess.** It's a neutral 0.5 at test time. Only the ground
  truth could settle it, so the model trains across the whole range.
- Blackouts ignore where play is, apart from the stoppage link. Real losses cluster in crowded areas.
- Smart Tagging weights were sized on simulated tags only.
- Labels on the open matches aren't the competition's labels. They're close, not identical, and any
  systematic timing difference won't show up here.
- Byte-identical output is confirmed on one machine, including from a fresh clone with its caches
  rebuilt. Training is sensitive to last-bit differences, so an Intel or AMD rebuild still needs
  checking (see "Checking another machine builds the same file").
