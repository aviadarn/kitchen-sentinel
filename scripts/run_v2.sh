#!/usr/bin/env bash
# v2 on a rented GPU box, after scripts/box_setup.sh: same-source negatives (Oops clips before their failure)
# in the manifest; one feature pass, then CV trained without them (pilot recipe) and with them.
set -euo pipefail
cd /workspace/kitchen-sentinel
source .box_env
uv run python -c "from huggingface_hub import snapshot_download as d; \
d('facebook/vjepa2-vitl-fpc16-256-ssv2'); d('google/siglip-so400m-patch14-384')"
export HF_HUB_OFFLINE=1
# Rerun until every video has arrived (uploads can still be running) and every window is done.
for attempt in 1 2 3 4 5 6; do
  uv run python -m ks.features --manifest data/manifest.csv --out features --workers "${WORKERS:-8}" 2>&1 \
    | grep --line-buffered -v -E "UNEXPECTED|Loading weights" >> features.log || true
  if tail -1 features.log | grep -q "^done; 0 windows"; then break; fi
  sleep 120
done
tail -1 features.log | grep -q "^done; 0 windows" || { echo FEATURES_INCOMPLETE >> cv_with.log; exit 1; }
uv run python -m ks.eval.cv --features features --out reports/v2_without_youtube --train-without oops > cv_without.log 2>&1
uv run python -m ks.eval.cv --features features --out reports/v2_with_youtube > cv_with.log 2>&1
echo V2_DONE >> cv_with.log
