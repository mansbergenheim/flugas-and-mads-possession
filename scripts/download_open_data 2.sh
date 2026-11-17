#!/bin/sh
# Open data about other matches, about 2.8 GB, into data_open/ (git-ignored).
# DFL/IDSSE: CC BY 4.0, credit DFL and Bassek et al. 2025. Metrica: acknowledge the source.
# Both are pinned to the versions the model was trained on, and every file is checked against
# mtah/open_data_sha256.json at the end, so a later change upstream can't change the submission.
set -eu
cd "$(dirname "$0")/.."
mkdir -p data_open/dfl data_open/metrica
M=https://raw.githubusercontent.com/metrica-sports/sample-data/e706dd506b360d69d9d123d5b8026e7294b13996
for f in documentation/events-definitions.pdf \
  data/Sample_Game_1/Sample_Game_1_RawEventsData.csv \
  data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv \
  data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv \
  data/Sample_Game_2/Sample_Game_2_RawEventsData.csv \
  data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv \
  data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv \
  data/Sample_Game_3/Sample_Game_3_events.json \
  data/Sample_Game_3/Sample_Game_3_metadata.xml \
  data/Sample_Game_3/Sample_Game_3_tracking.txt; do
  curl -sSL --fail --retry 3 -o "data_open/metrica/$(basename "$f")" "$M/$f"
done
B=https://huggingface.co/datasets/pysport/idsse-data/resolve/a715a38dfbaf5f58e431727c2b78d174101a703c
for id in 000002_DFL-MAT-J03WPY 000002_DFL-MAT-J03WOH 000002_DFL-MAT-J03WOY 000002_DFL-MAT-J03WQQ \
          000002_DFL-MAT-J03WR9 000001_DFL-MAT-J03WMX 000001_DFL-MAT-J03WN1; do
  for k in 02_01_matchinformation 03_02_events_raw 04_03_positions_raw_observed; do
    n="DFL_${k}_DFL-COM-$id.xml"
    curl -sSL --fail --retry 3 -o "data_open/dfl/$n" "$B/$n"
  done
done
python3 -c "
import sys
sys.path.insert(0, '.')
from mtah import datasets
problems = datasets.check_open_files()
print('\n'.join(problems) if problems else 'all 31 open files match the pinned versions')
sys.exit(1 if problems else 0)
"
