#!/bin/bash
# Regenerate the committed commentary cues (cues/) from the match files, on a Mac, and check they come
# out byte for byte the same. This is the evidence that every cue was made by a program and none by hand.
#
#   MTAH_DATA_DIR=~/mtah-data bash scripts/rebuild_cues.sh TRANSCRIPT.srt
#
# Needs: macOS (Apple's text recognition, built from scripts/ocr_video_text.swift with swiftc), ffmpeg,
# and Ollama 0.40.16 serving qwen3.5:4b with digest 7f3251fa6878... (ollama pull qwen3.5:4b). About two
# hours, nearly all of it the local language model. Nothing leaves the computer.
#
# Everything is rebuilt in a fresh folder beside the data (the match files are linked, not copied), so
# the files the submission was made from are never overwritten. Each step's result is compared with the
# hash cues/manifest.json recorded for it, so a difference is found at the step where it starts:
#
#   0 transcript   the speech-to-text of the broadcast commentary, made with OpenAI Whisper run locally:
#                    whisper BAR-ATH_broadcast.mp4 --model small --language English --output_format srt
#                  (openai-whisper 20250625, PyTorch 2.11.0, on CPU, Windows)
#                  Whisper may answer differently on other hardware, so the transcript is given here and
#                  checked against the hash the cues were made from; every step after it must repeat
#   1 alignment    scripts/align_broadcast.py: the broadcast lined up with the tactical recording
#   2 line-ups     scripts/read_lineups_vision.py: Apple text recognition + qwen3.5:4b on the line-up graphics
#   3 mentions     scripts/extract_possession_mentions.py: player names by code, their role by qwen3.5:4b
#   4 events       scripts/extract_commentary_events.py: what happened, by qwen3.5:4b
#   5 cues         scripts/export_cues.py: the sure cues, placed on the tactical timeline
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${MTAH_DATA_DIR:?set MTAH_DATA_DIR to the folder holding tactical/ and broadcast/}"
SRT="${1:?give the transcript (.srt) of the broadcast commentary}"
REBUILD="${REBUILD:-$DATA/../mtah-rebuild-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$REBUILD/commentary"
ln -s "$(cd "$DATA" && pwd)/tactical" "$REBUILD/tactical"
ln -s "$(cd "$DATA" && pwd)/broadcast" "$REBUILD/broadcast"
export MTAH_DATA_DIR="$REBUILD"
echo "rebuilding in $REBUILD"

expect() {  # expect STEP FILE KEY: compare a file's SHA-256 with the one cues/manifest.json recorded
    python3 - "$1" "$2" "$3" <<'EOF'
import hashlib, json, sys
step, path, key = sys.argv[1:]
m = json.load(open("cues/manifest.json"))
want = m
for k in key.split("/"):
    want = want[k]
got = hashlib.sha256(open(path, "rb").read()).hexdigest()
print(f"{step:<14} {'SAME' if got == want else 'DIFFERENT'}  {got[:16]}  (recorded {want[:16]})")
EOF
}

# Whisper on Windows writes Windows line endings and git stores them as Unix ones; the text is the same.
# The chain was run on the Unix-ending copy, so the transcript is normalised the same way first.
python3 -c "import sys; d=open(sys.argv[1],'rb').read().replace(b'\r\n', b'\n'); open(sys.argv[2],'wb').write(d)" \
    "$SRT" "$REBUILD/commentary/transcript.srt"
SRT="$REBUILD/commentary/transcript.srt"
expect "0 transcript" "$SRT" extractors/possession_mentions/transcript_sha256
python3 scripts/align_broadcast.py --data "$REBUILD"
expect "1 alignment" "$REBUILD/alignment/broadcast_alignment.npz" inputs_sha256/alignment
python3 scripts/read_lineups_vision.py --data "$REBUILD"
expect "2 line-ups" "$REBUILD/lineups/lineups.json" inputs_sha256/lineups
python3 scripts/extract_possession_mentions.py "$SRT" --data "$REBUILD" -o "$REBUILD/commentary/possession_mentions.csv"
expect "3 mentions" "$REBUILD/commentary/possession_mentions.csv" inputs_sha256/mentions
python3 scripts/extract_commentary_events.py "$SRT" -o "$REBUILD/commentary/commentary_events.csv"
expect "4 events" "$REBUILD/commentary/commentary_events.csv" inputs_sha256/events
python3 scripts/export_cues.py --mentions "$REBUILD/commentary/possession_mentions.csv" \
    --events "$REBUILD/commentary/commentary_events.csv" --out "$REBUILD/cues"
expect "5 cues" "$REBUILD/cues/possession_cues.csv" files/possession_cues.csv
expect "5 cues" "$REBUILD/cues/stoppage_cues.csv" files/stoppage_cues.csv
echo "If every line says SAME, the committed cues are exactly what the programs make from the match files."
