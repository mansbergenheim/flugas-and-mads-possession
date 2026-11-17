# Flugas and Mads · Initial Challenge: frame-level possession, FC Barcelona v Athletic Club

> **DRAFT, 17 November 2026.** Before submitting: fill every `TODO`, delete this box.

## Summary

Each frame is labelled by a model trained only on **other matches**: seven Bundesliga matches from
the public DFL/IDSSE dataset and three Metrica Sports sample matches. Their clean tracking is damaged
to look like the supplied automatic tracking, so that what the model learns carries over. The
pipeline has three stages:

1. **Evidence.** Two models give P(home), P(away), P(dead) five times a second from the tactical
   tracking: three gradient-boosted tree models and a small temporal convolutional network (TCN).
   Their probabilities are averaged in equal shares.
2. **Outside evidence.** Small, bounded nudges from the supplied Smart Tagging and from the broadcast
   commentary, both as soft evidence, never as labels.
3. **Decision.** A decoder picks one home/away/dead path per half, with a cost for every change.

No label from the supplied match was used for training, tuning, selection or validation. The output
is deterministic, and `regenerate.py` rebuilds the submitted file from the data bundle with one
command.

## 1. Approach

The metric counts possessions, not frames. A frame classifier on its own flickers and invents
possessions; heavy smoothing merges real ones. So we separate **evidence** (a probability per moment)
from **decision** (one path per half that maximises the summed log-probability minus a cost per
change: home↔away 2.5, live→dead 6.0, dead→live 4.0). A single deflection can't pay for two changes;
sustained evidence of a new team in control can.

**Why learned models.** Under our test protocol (§5), hand-built rules over the same inputs score
spellF1 0.312; a single tree model 0.634; the submitted average of trees and TCN 0.699.

**Why two kinds of model.** The tree models see each moment through 198 summary inputs (team
geometry, ball contact, speeds, how the evidence changes over the surrounding seconds). The TCN reads
the same inputs as a sequence, about six seconds either side, and is much better at short
possessions. The TCN alone scores higher on our main test, but falls behind the trees on a different
labelling style (SkillCorner). The plain average holds up on every test, and was chosen by a rule
fixed before the networks were trained (§5).

**Key assumptions.** The supplied tracking resembles our damaged training copies; the organisers'
labels are closer to our training labels' convention than to one we have not seen; the tactical
recording is the reference timeline.

## 2. Assets and how they are used

| Asset | Use |
|---|---|
| Tactical ATD (`_atd.xml`, `_atd.txt`) | The model's input. Converted to metres on a 105.3 × 68.0 m pitch; all inputs are per team (nearest player, counts, centroids, spreads, speeds), so no track slot is treated as a person. Missing values stay missing (the trees handle them natively; the TCN gets a missing flag per input). No correction or interpolation |
| Tactical Smart Tagging | Soft evidence for the decoder (§3) |
| Broadcast video and its audio | The commentary, turned into cues by local programs (§3.3) |
| Broadcast ATD | Only to line the broadcast up with the tactical recording, so commentary can be placed in time and anything said during a replay dropped |
| Organisers' constants | The graded half spans published in DATA.md; frames outside them are written as `X` |

**Tried and dropped:**

| Asset or idea | Why dropped |
|---|---|
| Referee whistle in the audio | Tried on open recordings (RoboCup, SoccerNet, synthetic mixes) and then on the two supplied soundtracks: no whistle line was shared between them. Nothing from it is used; the code is in the repository marked closed |
| Broadcast ball position | Not useful once lined up: too often missing or off the wide shot |
| Director's cuts as stoppage evidence | Measured on the lined-up broadcast; not added to the model |
| A smoother on the tree probabilities | Failed its pre-set rule on open matches |
| A second network (BiLSTM) | Passed, but no better than the TCN; the simplest passing combination was kept |
| SkillCorner data for training | Lowered the score on the other sources; kept for evaluation only |

## 3. Possession rules

### 3.1 What the models learn

| Situation | How it is handled |
|---|---|
| Controlled turnover | Learned from labelled turnovers; the decoder switches team when the summed evidence pays the turnover cost |
| Deflections, fleeting touches, aerial duels | A one- or two-sample lean towards the other team can't pay for switching away and back |
| Loose balls | The previous team is kept until evidence of the new team in control is sustained, matching the rule that a loose ball stays with the last team in control |
| Ball not observed | Inputs that need no ball (team shape, depth, collective movement) plus the ball evidence just before and after a gap; training copies have realistic gaps, so the models learn what to do in them |
| Stoppages | Slow collective movement (also as a rank within the half, which doesn't depend on the noise level), the ball off the pitch, and the outside evidence below |
| Before each kick-off | A strong nudge towards dead before each half's `Kick Off (Start)` tag, so no frame constants are needed |
| Abstention | None on graded frames: any `X` left there gets the label the scorer's forward fill would give |

### 3.2 Smart Tagging

Phase tags +0.1 per sample (5 a second, in log-probability) towards the phase's team and −0.05 against
dead; set-piece windows +0.3 towards dead, then +0.2 towards the named team. Overlapping tags are not
summed, since they come from one automatic system and share its errors. Sized on open matches with
deliberately imperfect simulated tags (+0.003 to +0.007 spellF1).

### 3.3 The broadcast commentary

The organisers confirmed in writing that the commentary may be used as input, and that the cue times
it produces may be committed to the repository. `TODO: confirm whether their answer also covered
player names and the on-screen line-up graphics (email question 4b).` Every step is a program running
on our own computer: no audio, transcript or derived file went to any hosted service.

1. **Transcript.** `TODO: speech-to-text model, version, command, and that it ran locally.`
2. **Line-ups**, read from the broadcast's line-up graphics: Apple's on-device text recognition finds
   the graphics, and a local vision-language model (Qwen 3.5 4B through Ollama 0.40.16, digest
   `7f3251fa6878…`) reads the names; a name is kept only if the model reads it in at least half of the
   graphic's frames *and* the text recognition also sees it. One away player was not read.
3. **Who has the ball.** Code finds every line-up name in the transcript, with strict rules: a name
   that could belong to either team is never used, and no team names or nicknames are used. The same
   local model says what each named player is doing. The team comes from the line-up, never from the
   model.
4. **What happened.** The same local model reads the transcript in one-minute windows and reports
   stoppages (fouls, offsides, ball out, set pieces, goals, substitutions) as happened, denied
   ("no offside, play on") or only speculated.
5. **Sure cues only.** On the real transcript the model called almost every mention "not live play",
   so only mentions with evidence of who has the ball are kept: the model said on, winning or losing
   the ball, or a ball-action word ("passes", "shoots", "header"…) stands next to the name. Words
   where the player may be on either side of a duel (tackle, foul, challenge) are excluded.
6. **Placing in time.** The broadcast is lined up with the tactical recording from the tracking alone;
   anything said during a replay or outside the lined-up broadcast is dropped.
7. **Checks before use.** A *side check* confirms the line-up's home team is the team our labels call
   home (by team name and, with a clear margin, shirt numbers). A *gate* then requires the cues to
   agree with the model at least 70% of the time at moments where the model is at least 90% sure;
   otherwise the team cues are switched off. Neither check uses any label.
8. **Effect.** Each cue nudges its team over the 2 s before the name is said
   (1.0 per sample, in log-probability); a reported stoppage nudges dead over 1 s before to 4 s after
   (0.3); a denied foul or offside nudges against dead (0.05). Each kind of evidence
   counts once per moment, and two teams named at once cancel.

On the real match: 305 name mentions in 132 minutes of commentary; 15 sure cues, 14 placed; 144
stoppage reports, 137 placed. The side check agreed; the gate saw the cues agree with the model at all
44 sure moments it could compare, so the team cues are used. The commentary changes 245 of 158,857
graded frames at the standard strengths (0.15%) and 305 at the strong ones (0.19%), 90% of them
towards dead ball. The submitted file uses the strong setting.

**How the cues are reproduced.** The language-model steps can give different answers on other
hardware, and the text recognition is macOS-only, so the chain's end product is committed in `cues/`:
tactical frame, side or event type, and the rule that made each cue, with no names and no commentary
text. `cues/manifest.json` records the SHA-256 of every input (transcript, line-ups, extractor outputs,
alignment, tracking metadata), each extractor's model digest, Ollama version, prompt and script hashes
and settings, and the counts at every step. `scripts/rebuild_cues.sh` reruns the whole chain from the
data bundle on a Mac and reports, step by step, whether each result is byte-identical to what the
manifest recorded; this is how anyone can check the cues were made by programs and not by hand.

**Strengths and settings.** The submitted strengths (team 1.0, dead 0.3) and the 2 s window come from
simulations on the ten open matches. With player names every 6 s, team strength 1.0 gained +0.015
spellF1 when one name in ten points at the wrong team (10 of 10 matches up) and +0.005 when one in four
does, the same as 0.5; 2.0 lost badly at one in four and was ruled out. Our cues are few and agreed with
the model at every sure moment, which is the mostly-right case. A dead-ball strength of 0.3 was slightly
ahead of 0.15 in most rows of the stoppage simulation (+0.0003 to +0.0008 against +0.0001 to +0.0004). No
strength was chosen on the supplied match.

## 4. Training data

| Dataset | Source and version | Licence | Matches | Who labelled, and how | Amount |
|---|---|---|---|---|---|
| DFL / IDSSE (Bassek et al. 2025) | <https://huggingface.co/datasets/pysport/idsse-data>, revision `a715a38` | CC BY 4.0 | 7 (2022/23 Bundesliga and 2. Bundesliga) | The data provider: per-frame `BallStatus` and `BallPossession` in the position files. Possessions under 1.0 s between two of the other team's are folded, since the provider's flag changes on touches the competition doesn't count | 1,002,644 frames (11.1 h), 2,707 possessions |
| Metrica Sports sample data | <https://github.com/metrica-sports/sample-data>, commit `e706dd5` | Public sample data | 3 anonymised matches | Metrica's hand-collected events, turned into per-frame labels by fixed rules following Metrica's event definitions | 424,880 frames (4.7 h), 1,041 possessions |
| SkillCorner open data | <https://github.com/SkillCorner/opendata>, commit `4340d27` | MIT | 10 of 20 A-League matches | SkillCorner's algorithmic possession groups | **Evaluation only**; the other 10 untouched |

All 31 open files are pinned by SHA-256 (`mtah/open_data_sha256.json`) and checked before every build.

**Damaging the training tracking** (`mtah/degrade.py`): ball gaps whose lengths follow the ball-missing
pattern of the organisers' public baseline output (a property of the tracking, not of labels), thinned
to the 59.3% visible share published in DATA.md; correlated position noise; false ball detections;
players split into anonymous tracklets over 46 slots with occasional wrong-team assignment; stoppage
gaps whose edges never line up with the true boundary. All drawn from wide ranges. Each match is used
in nine damaged copies (three per tree model), each also with home and away exchanged; samples are
weighted by (possession length in seconds, clipped to 1–60)^−0.5.

**The models.**
- *Trees:* three scikit-learn 1.8.0 `HistGradientBoostingClassifier`s (200 iterations, learning rate
  0.07, 31 leaves, at least 300 samples per leaf, L2 1.0, seeds 0–2), each on its own draw of the
  damaged copies, trained from scratch when the submission is built.
- *TCN:* three networks of about 124,000 weights each (64 channels, dilations 1-2-4-8), trained on the
  same ten matches and inputs on Google Colab with open data only (`colab/train_tcn.ipynb`); training
  length found with one match held back, temperature 1.1 fitted on it. GPU training doesn't repeat
  to the last bit, so the weights ship as `models/tcn.npz` (SHA-256 pinned) and prediction runs in
  plain numpy, rounded to four decimals, so every machine gets the same numbers.

No pretrained models, no hosted APIs, no pseudo-labels from the supplied match.

## 5. Development and evaluation

**Protocol.** Leave one match out over the ten open matches: each match is scored by models that never
trained on it, on a damaged copy with different seeds. Scores use our reimplementation of the official
metric, tested against the organisers' published results.

| | 10 open matches | Stricter labels | Stress table | Unseen label source | SkillCorner |
|---|---|---|---|---|---|
| Organisers' baseline (`smooth50`) | 0.239 | | | | 0.199 |
| Hand-built rules | 0.312 | | | | 0.255 |
| One tree model (our 8 Nov version) | 0.634 | 0.688 | 0.616 | 0.588 | 0.340 |
| Three tree models | 0.646 | 0.697 | 0.628 | 0.601 | 0.351 |
| TCN alone | 0.707 | 0.718 | 0.687 | 0.662 | 0.349 |
| **Submitted: trees + TCN averaged** | **0.699** | **0.732** | **0.681** | **0.646** | **0.370** |

*Stricter labels*: brief touches given back to the team in control. *Stress table*: 19 kinds of extra
damage (from "ball always visible" to "no ball at all"), 57 copies. *Unseen label source*: trained on
one source (DFL or Metrica), scored on the other. The submitted model was chosen by a rule written
before the networks were trained: at least +0.015 on the ten matches with 8 of 10 up, and no loss in
the four other tests; among passing combinations, the simplest. It is ahead of the 8 Nov version on
every match of every test and on all 57 stress copies. Eleven other ways of combining the trained
models were tried; none was clearly better. The decoder settings are the 8 Nov version's; re-picking
them for the new model gained 0.002 at home and lost 0.005 on SkillCorner, so they were kept.

**Disclosures about the process.**
- Input families were screened on 3 of the 10 matches after studying errors on all 10; the gain held
  on the other 7, but that isn't a clean test.
- An early version fitted two damage settings to the organisers' published baseline scores, which
  derive from the private ground truth. We removed that; the decoder and weighting were re-picked
  without it, but the input design dates from that period.
- After a first run, summary measures of the supplied tracking were compared with our training copies
  (no labels, no scores): the supplied tracking was cleaner, so the damage ranges were widened towards
  cleaner tracking (0.615 → 0.630 on open matches, within noise).
- **The commentary design was changed after its first run on the supplied match.** The first run's
  counts showed the language model called 299 of 305 mentions "not live", and the gate had nothing to
  compare. A team member then read 20 random transcript lines with the model's answers, to judge the
  extractor; in 14 of them they couldn't tell. That led to keeping only sure cues and to the side
  check. These were design decisions; no cue was added, removed or corrected by hand, and nothing was
  compared with any label. A team member also looked at the line-up file read from the broadcast to
  see it worked; nothing in it was corrected. We also looked at how many frames the two cue strengths
  change before choosing; the choice itself rests on the open-match simulations above.
- **The public five-minute ground truth was used only as a pipeline check** (file length, frame
  alignment, home/away orientation). Its score was never used to train, tune or select anything.

**Expected performance: unknown.** Open-match scores are not competition scores.

## 6. Limitations

- **The labelling convention may differ from the organisers'.** Our largest risk: on SkillCorner's
  algorithmic labels every model scores about half of what it scores at home.
- **Short possessions:** about a quarter of true possessions under 2 s are found on open matches.
- **Dependence on the ball:** the stress table runs from 0.551 under very harsh damage up to its best
  with the ball always visible.
- **The commentary adds little:** few sure cues, mostly where the model was already sure; its value is
  as a guard, not a large gain.
- **The cue chain needs a Mac** (Apple's text recognition) and the same model digest to repeat exactly.
- **With more time:** a labelled source with the competition's convention; a speech-to-text with word
  timings; a cross-platform text recognition step.

## 7. Reproduction

- Repository: <https://github.com/mansbergenheim/flugas-and-mads-possession>, commit `TODO`
- One command, from a fresh clone, Python 3.14 (built with 3.14.3):
  `python3.14 regenerate.py --data /path/to/bundle` (Windows: `py -3.14 regenerate.py --data …`).
  It makes a private environment with the pinned packages (numpy 2.4.4, scipy 1.17.1, scikit-learn
  1.8.0), downloads and checks the open data (about 2.6 GB), runs the tests, compares the model's
  fingerprints with the build machine's, builds the file with the settings in `submission.json`, runs
  the organisers' own validator, and compares the file's SHA-256 with the recorded one. About an hour
  on a laptop CPU; no GPU.
- The build step alone: `MTAH_DATA_DIR=/path/to/bundle python3 scripts/make_submission.py --team
  Flugas_and_Mads --cues cues/ --team-weight 1.0 --dead-weight 0.3`
- Built on macOS (Apple M2, 8 GB). Rebuilt on `TODO: Windows/x86 result`.
- Final file: `Flugas_and_Mads.csv`, SHA-256
  `211d9f84e378f0027984649e086bd99082562f29cc810d65ad9a71b09b252dd8`. Each build also writes a manifest with the code
  commit, settings, package versions and input hashes.
- Checking the commentary cues (Mac, Ollama, about two hours):
  `MTAH_DATA_DIR=/path/to/bundle bash scripts/rebuild_cues.sh TRANSCRIPT.srt`

## 8. Use of AI tools

An AI coding assistant (Claude, Anthropic) helped write and test code and documentation, on public data
and invented examples. In line with the data terms, no supplied asset, extract, transcript or output
computed from them was shown to any hosted AI service; runs on the supplied data were made by team
members on their own computers, and only short summaries in words and counts were discussed. The
language and vision models that read the commentary and the line-up graphics ran locally (Ollama).

## 9. Credits

DFL / IDSSE data: Deutsche Fußball Liga; Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025), *An
integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data, CC BY 4.0.
Metrica Sports sample data. SkillCorner open data (MIT, evaluation only). NumPy, SciPy, scikit-learn,
PyTorch (training on Colab only), Ollama and Qwen 3.5.
