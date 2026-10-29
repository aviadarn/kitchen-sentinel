# kitchen-sentinel

Can a video model see a kitchen accident coming? This repo applies the recipe of Nexar's
[BADAS-Open](https://huggingface.co/nexar-ai/BADAS-Open) collision-anticipation model (a frozen
V-JEPA2 ViT-L video encoder, an attention head, 2 s sliding windows) to cuts, burns, fire, falling
objects and slips. It covers the whole loop in one place: data collection, a labeling tool, a feature
cache, training, and an evaluation built to catch shortcuts.

Not affiliated with Nexar.

## Result in brief

- **A video model beats a single frame, but only a little.** On ranking each clip's warning moments
  above its own earlier, calm moments, the frozen V-JEPA2 probe scores hard AP 0.56 against 0.47 for
  SigLIP on the window's last frame (pilot recipe, 5-fold cross-validation, 89 clips).
- **BADAS-Open does not transfer from roads to kitchens**: AUC 0.52, chance level.
- **Part of the signal is a shortcut.** Every accident clip is a YouTube fail compilation, and normal
  footage comes from other datasets. On YouTube footage where nothing goes wrong, the probe and the
  single-frame model raise 19–100 alarms per hour, against 1–2 per hour on normal cooking. Training
  on 11 minutes of such footage halves those false alarms but does not remove them.
- **The limit is data, not the model.** 89 labeled clips, a median warning of 0.43 s between "danger
  visible" and the accident, and 34 clips with no calm footage before the warning at all.

## Results

5-fold cross-validation grouped by YouTube compilation, EPIC participant and Charades actor, so
footage from one source never sits on both sides. Per fold, the alarm threshold tau is set for at most 1
false alarm per hour on the validation fold's normal cooking, then applied to the test fold.

| model | trained on YouTube normal footage | hard AP [95% CI] | caught on time | false alarms/h, normal cooking | false alarms/h, YouTube normal |
|---|---|---|---|---|---|
| class prior | – | 0.23 | 0% | 0.00 | 0.0 |
| BADAS-Open zero-shot | – | 0.24 [0.18, 0.34] | 0% | 1.22 | 11.2 |
| SigLIP last frame + LR | no | 0.47 [0.39, 0.64] | 41% | 0.95 | 100.5 |
| SigLIP last frame + LR | yes | 0.43 [0.38, 0.68] | 33% | 2.24 | 55.8 |
| V-JEPA2 mean token + LR | no | 0.45 [0.39, 0.62] | 7% | 0.88 | 11.2 |
| V-JEPA2 mean token + LR | yes | 0.40 [0.33, 0.50] | 5% | 0.41 | 0.0 |
| **V-JEPA2 probe (3 seeds)** | no | **0.56 [0.50, 0.73]** | 28% | 1.16 | 33.5 |
| **V-JEPA2 probe (3 seeds)** | yes | 0.53 [0.45, 0.68] | 26% | 1.81 | 18.6 |

- **Hard AP** ranks the windows inside each labeled warning (alert to accident) against the same
  clip's windows from before the alert. Source, camera and editing are held fixed, so it measures
  anticipation rather than "is this a fail video". 95% CIs come from resampling whole compilations.
- **Caught on time**: share of the 81 accidents (those with at least one full 2 s window before them)
  whose first alarm came no earlier than 0.5 s before the labeled alert. Earlier alarms count as
  early, not caught.
- **YouTube normal**: 125 Oops kitchen clips that were not kept, cut 1.5 s before their failure
  marker. That is only 11 minutes, so those rates rest on a handful of alarms (probe: about 3–6,
  SigLIP: about 10–18).
- Full tables, including the BADAS-style "easy" AP, time-to-accident and calibration:
  `reports/v2_without_youtube/` and `reports/v2_with_youtube/` (`results.md`, `audit.md`).
  `curves.png` shows risk curves for 20 random clips.

### What the audit found

`audit.md` compares each model's scores on calm fail-clip footage (before the alert) with normal
cooking. A model that anticipates should not tell the two apart (AUC 0.5). The single-frame model
does, 93% of the time (88% after the fix); the probe 71% (70%). The single frame's higher catch rate
in the pilot came largely from firing throughout fail videos. The probe fires about 15 times more
often inside warnings than before them, against about 5 times for the single frame.

## Limitations

- One labeler. The planned blind self-agreement recheck was not run.
- Warnings are short (median 0.43 s), so the task sits close to "spot the failure as it starts".
- Too few clips for a fixed test split: every number is out-of-fold over 5 folds, with wide CIs.
- The normal-cooking false-alarm rate holds in-distribution only. On footage that looks like the
  positives but where nothing goes wrong, the probe alarms 10–30 times more often.
- Only frozen features were tried. No fine-tuning, which would likely overfit 89 clips.

## Next steps

1. More and longer accident clips, cut with at least 5 s of footage before the fail.
2. Much more same-source normal footage: about 3,300 Oops clips have at least 4 s before their failure.
3. Only then, fine-tuning (from V-JEPA2 or BADAS-Open weights).

## Pipeline

| step | where | command |
|---|---|---|
| Oops annotations and videos, Charades, EPIC subset, normalized to 8 fps, 256 px | GPU box | `python -m ks.data.prepare {oops-annotations,oops-videos,oops-normalize,charades,epic} --out ...` |
| Kitchen filter (SigLIP zero-shot + titles) | GPU box | `python -m ks.data.filter_kitchen --videos <dir> --ann <dir> --out candidates.csv` |
| Labeling: keep/reject, hazard, alert and event marks, several accidents per clip | Mac, browser | `python -m ks.label.server --candidates data/candidates.csv --videos data/label_clips --labels data/labels.csv` |
| Manifest with 5 group folds and YouTube normal footage | Mac | `python -m ks.data.build --candidates data/candidates.csv --labels data/labels.csv --negatives data/norm/*/negatives.csv --oops-dir data/norm/oops --pasbench-meta data/pasbench_metadata.csv --out data/manifest.csv` |
| Feature cache: V-JEPA2 tokens, BADAS-Open score, SigLIP embedding | GPU box | `python -m ks.features --manifest data/manifest.csv --out features` |
| Cross-validated probe and baselines | GPU box | `python -m ks.eval.cv --features features --out reports/<run> [--train-without oops]` |
| Shortcut audit, outcome by hazard, risk curves | Mac | `python -m ks.eval.audit --features features --run reports/<run>` |

`scripts/box_setup.sh` and `scripts/run_v2.sh` reproduce the GPU part on a rented RTX 4090: about 1 h
of features and 1 h of CV, roughly $1. BADAS-Open is gated on Hugging Face. Accept its terms, then
point `BADAS_CKPT` at a local copy of `weights/badas_open.pth`. Its checkpoint runs V-JEPA2's
predictor 1 s into the future and feeds those tokens to the head; `Anticipator(future_fraction=...)`
reproduces that.

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
