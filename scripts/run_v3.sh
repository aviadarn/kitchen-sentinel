#!/usr/bin/env bash
# v3 on a rented GPU box, after scripts/box_setup.sh: every Oops clip's footage before its failure (cut 3 s
# early) as YouTube normal negatives, then one feature pass and CV trained without and with them.
# Uploaded from the Mac beforehand: data/{candidates,labels,pasbench_metadata}.csv and data/norm/epic/.
# The box rebuilds the rest itself (the Mac's uplink is ~1 MB/s); $BADAS_CKPT points at the BADAS weights.
set -euo pipefail
cd /workspace/kitchen-sentinel
source .box_env
W=${WORKERS:-16}
uv run python -m ks.data.prepare oops-videos --out data/oops &
uv run python -m ks.data.prepare charades --out data/norm/charades --work data/tmp --workers "$W" &
wait
uv run python -m ks.data.prepare oops-normalize --candidates data/candidates.csv --out data/norm/oops --workers "$W"
uv run python -m ks.data.build --candidates data/candidates.csv --labels data/labels.csv \
  --negatives data/norm/epic/negatives.csv data/norm/charades/negatives.csv --oops-dir data/norm/oops \
  --pasbench-meta data/pasbench_metadata.csv --prefail all --prefail-margin 3.0 --prefail-dir data/norm/oops_prefail \
  --out data/manifest.csv
uv run python -m ks.data.prepare oops-prefail --manifest data/manifest.csv --candidates data/candidates.csv \
  --out data/norm/oops_prefail --workers "$W"
uv run python -c "from huggingface_hub import snapshot_download as d; \
d('facebook/vjepa2-vitl-fpc16-256-ssv2'); d('google/siglip-so400m-patch14-384')"
export HF_HUB_OFFLINE=1
uv run python -m ks.features --manifest data/manifest.csv --out features --workers 8 2>&1 \
  | grep --line-buffered -v -E "UNEXPECTED|Loading weights" > features.log
# A clip ffmpeg cannot read leaves its windows undone; ks.eval.cv drops undone windows, so note the count
# and go on (v3: one 2.2 s prefail clip, 1 window of 63,979) rather than stopping before the CV.
grep "^done;" features.log | tail -1
uv run python -m ks.eval.cv --features features --out reports/v3_without_youtube --train-without oops > cv_without.log 2>&1
uv run python -m ks.eval.cv --features features --out reports/v3_with_youtube > cv_with.log 2>&1
echo V3_DONE >> cv_with.log
