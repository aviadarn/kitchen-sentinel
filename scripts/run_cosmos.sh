#!/usr/bin/env bash
# Cosmos3 Reasoner zero-shot on the v3 windows, on a rented GPU box after scripts/box_setup.sh (24 GB is
# enough: the Reasoner alone is ~17 GB in bf16). Uploaded from the Mac beforehand: data/candidates.csv,
# data/manifest_v3.csv, data/norm/epic/, and features_v3/index.csv as features_cosmos/index.csv, so
# ks.features refuses to run unless the box rebuilds exactly the v3 windows. Then, on the Mac:
#   python -m ks.eval.cv --features features_cosmos --scores reports/v3_with_youtube/scores.csv \
#       --add-zero-shot cosmos --out reports/v3_with_youtube
set -euo pipefail
cd /workspace/kitchen-sentinel
source .box_env
W=${WORKERS:-16}
uv run python -m ks.data.prepare oops-videos --out data/oops &
oops=$!
uv run python -m ks.data.prepare charades --out data/norm/charades --work data/tmp --workers "$W" &
charades=$!
# Reasoner tower only: the video generator's VAE and the audio tokenizer are not needed
uv run python -c "from huggingface_hub import snapshot_download as d; \
d('nvidia/Cosmos3-Nano', ignore_patterns=['vae/*', 'sound_tokenizer/*', 'assets/*', 'images/*'])" &
weights=$!
# A bare `wait` returns 0 even when a job failed (a reset Charades download once slipped through);
# waiting on each pid lets set -e stop the run.
wait $oops
wait $charades
wait $weights
uv run python -m ks.data.prepare oops-normalize --candidates data/candidates.csv --out data/norm/oops --workers "$W"
uv run python -m ks.data.prepare oops-prefail --manifest data/manifest_v3.csv --candidates data/candidates.csv \
  --out data/norm/oops_prefail --workers "$W"
uv run python -m ks.features --manifest data/manifest_v3.csv --out features_cosmos --models cosmos \
  --batch "${BATCH:-16}" --workers 8 2>&1 | grep --line-buffered -v -E "UNEXPECTED|Loading weights" > cosmos.log
grep "^done;" cosmos.log | tail -1
echo COSMOS_DONE >> cosmos.log
