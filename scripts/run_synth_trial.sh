#!/usr/bin/env bash
# Synthesis trial on a rented GPU box, after scripts/box_setup.sh: fetch the source Charades videos,
# generate the matched pairs, and add 8 fps / 256 px copies for the feature pipeline.
# Extra arguments go to ks.synth.wan_pairs (e.g. --limit 1 --steps 30 for a timing run).
set -euo pipefail
cd /workspace/kitchen-sentinel
source .box_env
uv sync -q --extra model --extra gen
mkdir -p data/charades_480 data/tmp
zip=data/tmp/Charades_v1_480.zip
if [ ! -f "$zip" ]; then
  curl -sSfL -o "$zip.part" https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip
  mv "$zip.part" "$zip"
fi
uv run python -c "
import zipfile, pandas as pd
from pathlib import Path
with zipfile.ZipFile('$zip') as z:
    for v in pd.read_csv('configs/synth_trial.csv').video_id:
        vid = v.split(':')[-1]
        Path('data/charades_480', vid + '.mp4').write_bytes(z.read('Charades_v1_480/' + vid + '.mp4'))
"
uv run python -c "from huggingface_hub import snapshot_download as d; d('Wan-AI/Wan2.2-TI2V-5B-Diffusers')"
uv run python -m ks.synth.wan_pairs --jobs configs/synth_trial.csv --videos data/charades_480 --out data/synth/trial "$@"
uv run python -c "
from pathlib import Path
from ks.data.normalize import normalize_video
out = Path('data/synth/trial_norm'); out.mkdir(parents=True, exist_ok=True)
for p in sorted(Path('data/synth/trial').glob('p*_*.mp4')):
    normalize_video(p, out / p.name)
"
echo SYNTH_DONE
