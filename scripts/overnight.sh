#!/bin/bash
# The real-match runs, one after the other, unattended. Run this yourself before bed:
#
#   cd "<repo>" && caffeinate -i bash scripts/overnight.sh "YOUR_TEAM_NAME"
#
# 1. who-has-the-ball mentions from the commentary (local Qwen, about an hour)
# 2. the words-only commentary check (the gate)
# 3. the submission WITHOUT commentary (the candidate as it stands, about 30 minutes)
# 4. the submission WITH commentary (about 30 minutes; team evidence only if the gate says on)
#
# Everything stays in <data folder>/overnight/. The log holds words and counts only: paste it to
# Claude in the morning. Nothing is uploaded or committed; which file to submit is your decision.
set -u
cd "$(dirname "$0")/.."
export MTAH_DATA_DIR="${MTAH_DATA_DIR:-$HOME/mtah-data}"
TEAM="${1:?give the team name, in quotes}"
SRT="${SRT:-../private-output/commentary/BAR-ATH_broadcast.srt}"
EVENTS="${EVENTS:-../private-output/commentary/commentary_events.csv}"
OUT="$MTAH_DATA_DIR/overnight"
MENTIONS="$MTAH_DATA_DIR/commentary/possession_mentions.csv"
mkdir -p "$OUT" "$MTAH_DATA_DIR/commentary"
rm -f "$OUT/DONE"
LOG="$OUT/log.txt"
ERR="$OUT/errors.txt"
: > "$ERR"

step() {
    echo
    echo "== $1  ($(date '+%H:%M'))"
}

{
    echo "overnight run, started $(date '+%Y-%m-%d %H:%M')"
    step "1 POSSESSION MENTIONS"
    python3 scripts/extract_possession_mentions.py "$SRT" -o "$MENTIONS" 2>>"$ERR" || echo "FAILED: see errors.txt"
    step "2 COMMENTARY CHECK"
    python3 scripts/check_commentary.py --mentions "$MENTIONS" --events "$EVENTS" 2>>"$ERR" || echo "FAILED: see errors.txt"
    step "3 SUBMISSION WITHOUT COMMENTARY"
    python3 scripts/make_submission.py --team "$TEAM" --out "$OUT/without_commentary" 2>>"$ERR" || echo "FAILED: see errors.txt"
    step "4 SUBMISSION WITH COMMENTARY"
    python3 scripts/make_submission.py --team "$TEAM" --out "$OUT/with_commentary" \
        --mentions "$MENTIONS" --events "$EVENTS" 2>>"$ERR" || echo "FAILED: see errors.txt"
    step "FINISHED"
} 2>&1 | tee "$LOG"
touch "$OUT/DONE"
