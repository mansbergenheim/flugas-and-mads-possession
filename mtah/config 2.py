"""The settings the submission uses. Chosen on open matches only; see scripts/sweep_decoder.py."""

from .decoder import DecoderConfig
from .smart_tagging import TagWeights

# Best of the grid in scripts/sweep_decoder.py on held-out predictions from 10 open matches. The whole
# grid spans only 0.011, so these are not delicate. See runs/oof/v5/sweep_prob.json.
DECODER = DecoderConfig(temperature=1.0, turnover=2.5, to_dead=6.0, from_dead=4.0)
TAG_WEIGHTS = TagWeights()

# The metric counts possessions, not frames, so left alone a model spends its effort on long ones.
# Each sample is weighted by (seconds in its run) ** -SPELL_WEIGHT: 0 treats every sample alike, 1
# every possession alike. Between 0.5 and 0.75 the score is flat; 1 makes too many possessions.
SPELL_WEIGHT = 0.5

# Each training match is used this many times at randomly drawn tracking quality.
TRAIN_COPIES = 3

# The submission averages this many tree models, each with its own draw of those copies and its own
# tree seed, and then averages them with the TCN in equal shares (mtah/candidate.py). One model is one
# draw, and draws differ by more than most changes we test.
TREE_DRAWS = 3
