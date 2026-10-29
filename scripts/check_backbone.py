"""Phase 0 check: the real V-JEPA2 backbone loads and emits the token grid the pipeline assumes.

With --badas, also loads the gated BADAS-Open checkpoint into our Anticipator and reports any key
mismatch (needs the HF terms accepted at https://huggingface.co/nexar-ai/BADAS-Open).

    uv run python scripts/check_backbone.py [--badas]
"""

import argparse
import time

import torch
from transformers import VJEPA2Model

from ks.models.vjepa_anticipator import BACKBONE, Anticipator, load_badas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--badas", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    t0 = time.time()
    backbone, info = VJEPA2Model.from_pretrained(BACKBONE, output_loading_info=True)
    print(f"loaded {BACKBONE} in {time.time() - t0:.0f}s, "
          f"{sum(p.numel() for p in backbone.parameters()) / 1e6:.0f}M params")
    for kind in ("missing_keys", "unexpected_keys"):
        keys = info[kind]
        print(f"  {kind}: {len(keys)}" + (f" e.g. {sorted(keys)[:3]}" if keys else ""))

    model = Anticipator(backbone, spatial_pool=2).to(device).eval()
    clip = torch.randn(1, 16, 3, 256, 256, device=device)
    with torch.no_grad():
        tokens = model.backbone(pixel_values_videos=clip, skip_predictor=True).last_hidden_state
        t0 = time.time()
        pooled = model.encode(clip)
        logits = model.head(pooled)
        if device == "mps":
            torch.mps.synchronize()
    print(f"device={device} raw tokens {tuple(tokens.shape)} pooled {tuple(pooled.shape)} "
          f"logits {tuple(logits.shape)}; one clip {time.time() - t0:.2f}s")

    if args.badas:
        load_badas(backbone)
        print("BADAS checkpoint: loaded, no missing tensors")


if __name__ == "__main__":
    main()
