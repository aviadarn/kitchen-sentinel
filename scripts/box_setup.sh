#!/usr/bin/env bash
# Bootstrap a rented GPU box (vast.ai PyTorch image) after the repo is rsynced to /workspace/kitchen-sentinel.
set -euo pipefail
apt-get update -qq && apt-get install -y -qq ffmpeg rsync pigz >/dev/null
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
cd /workspace/kitchen-sentinel
uv sync -q --extra model
# Each dataloader/decode worker re-imports the numeric stack; one BLAS thread per core per worker has
# exhausted a container's pids limit before and hung the run silently.
cat > .box_env <<'ENV'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source "$HOME/.local/bin/env"
ENV
echo "ready: $(uv run python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))')"
