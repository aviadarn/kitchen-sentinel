# kitchen-sentinel

Anticipating kitchen accidents (cuts, burns, fire, falling objects, slips) from video, seconds
before they happen. The recipe follows Nexar's [BADAS-Open](https://huggingface.co/nexar-ai/BADAS-Open)
collision-anticipation model, a V-JEPA2 ViT-L video encoder with an attentive-probe head. Here it
is applied to kitchens, with data collection, labeling, training and evaluation in one repo.

Not affiliated with Nexar.

## Status

Phase 0: scaffold, core logic (window labels, metrics, splits, manifest validation) and backbone checks.

## Data rights

This repo never contains video. It publishes manifests (video IDs + timestamps), labels, code and
metrics only. The source datasets are Oops! (unofficial mirror, no license), EPIC-KITCHENS-100
(CC BY-NC 4.0), Charades (non-commercial research) and PaSBench-Video (CC BY-NC 4.0). Use them for
research only.

## Development

```bash
uv sync                    # core (labels, metrics, splits)
uv sync --extra model      # + torch / transformers / av for model work
uv run pytest
```
