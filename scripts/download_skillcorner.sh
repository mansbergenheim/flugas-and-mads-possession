#!/bin/sh
# SkillCorner open data: 20 A-League 2024/25 matches of broadcast tracking, about 1.8 GB, into
# data_open/skillcorner/ (git-ignored). MIT licence; SkillCorner asks to be credited.
# Pinned to one commit, so the files are the same on every machine.
set -eu
cd "$(dirname "$0")/.."
COMMIT=4340d274572876239c154c90bc507a9b3250a656
RAW=https://raw.githubusercontent.com/SkillCorner/opendata/$COMMIT/data
LFS=https://media.githubusercontent.com/media/SkillCorner/opendata/$COMMIT/data
IDS="1874553 1886347 1899585 1925299 1927964 1953632 1959846 1986691 1996435 1996436
     2006229 2006363 2007448 2007721 2010085 2011166 2013725 2015213 2016236 2017461"
mkdir -p data_open/skillcorner
curl -sSL --fail --retry 3 -o data_open/skillcorner/matches.json "$RAW/matches.json"
for id in $IDS; do
  d="data_open/skillcorner/$id"
  mkdir -p "$d"
  for f in match.json phases_of_play.csv dynamic_events.csv; do
    curl -sSL --fail --retry 3 -o "$d/${id}_$f" "$RAW/matches/$id/${id}_$f"
  done
  if [ ! -s "$d/${id}_tracking_extrapolated.jsonl" ]; then
    curl -sSL --fail --retry 3 -o "$d/${id}_tracking_extrapolated.jsonl.part" \
      "$LFS/matches/$id/${id}_tracking_extrapolated.jsonl"
    mv "$d/${id}_tracking_extrapolated.jsonl.part" "$d/${id}_tracking_extrapolated.jsonl"
  fi
  echo "$id done"
done
echo all done
