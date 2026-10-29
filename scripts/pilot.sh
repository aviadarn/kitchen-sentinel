#!/usr/bin/env bash
# Phase 3 pilot on a rented GPU box, after scripts/box_setup.sh. Expects data/manifest.csv, data/norm/ and
# the BADAS-Open checkpoint at $BADAS_CKPT (copied from a machine that accepted its HF terms).
set -euo pipefail
cd /workspace/kitchen-sentinel
source .box_env
uv run python -c "from huggingface_hub import snapshot_download as d; \
d('facebook/vjepa2-vitl-fpc16-256-ssv2'); d('google/siglip-so400m-patch14-384')"
export HF_HUB_OFFLINE=1
uv run python -m ks.features --manifest data/manifest.csv --out features --workers "${WORKERS:-8}" 2>&1 \
  | grep --line-buffered -v -E "UNEXPECTED|Loading weights" > features.log
uv run python -m ks.eval.cv --features features --out reports/pilot > cv.log 2>&1
echo PILOT_DONE >> cv.log
