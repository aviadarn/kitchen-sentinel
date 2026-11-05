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
- **Neither does a general video-reasoning model, asked directly.** NVIDIA's Cosmos3-Nano, asked one
  yes/no question per window, reaches hard AP 0.42 (AUC 0.65): above BADAS-Open, below the single
  frame. It stays quiet on calm cooking but raises 37 alarms an hour on YouTube footage where nothing
  goes wrong, and catches 11% of accidents on time.
- **Part of the signal was a shortcut, and real footage fixes it.** Every accident clip is a YouTube
  fail compilation, and normal footage comes from other datasets. On YouTube footage where nothing
  goes wrong, the probe trained the usual way raises 34 alarms an hour (the single frame 162),
  against about 1 on normal cooking. Adding 3 hours of such footage to training (2,092 Oops clips,
  cut 3 s before their failure) brings the probe down to 1.9 an hour while it catches accidents as
  before (26% on time, from 28%).
- **Generated accidents did not help.** Wan 2.2 continued real cooking frames into accidents; on
  review they look generated and lack physical logic (`reports/synth_trial/`).
- **The limit is data, not the model.** 89 labeled clips, a median warning of 0.43 s between "danger
  visible" and the accident, and 34 clips with no calm footage before the warning at all.

## Results

5-fold cross-validation grouped by YouTube compilation, EPIC participant and Charades actor, so
footage from one source never sits on both sides. Per fold, the alarm threshold tau is set for at most 1
false alarm per hour on the validation fold's normal cooking, then applied to the test fold.

| model | trained on YouTube normal footage | hard AP [95% CI] | caught on time | false alarms/h, normal cooking | false alarms/h, YouTube normal |
|---|---|---|---|---|---|
| class prior | – | 0.23 | 0% | 0.00 | 0.0 |
| BADAS-Open zero-shot | – | 0.24 [0.18, 0.34] | 0% | 1.22 | 7.1 |
| Cosmos3-Nano zero-shot (yes/no question) | – | 0.42 [0.30, 0.62] | 11% | 1.43 | 37.4 |
| SigLIP last frame + LR | no | 0.47 [0.39, 0.64] | 41% | 0.95 | 162.1 |
| SigLIP last frame + LR | yes | 0.50 [0.41, 0.71] | 41% | 1.36 | 8.1 |
| V-JEPA2 mean token + LR | no | 0.45 [0.39, 0.62] | 7% | 0.88 | 37.4 |
| V-JEPA2 mean token + LR | yes | 0.39 [0.30, 0.51] | 4% | 0.48 | 1.7 |
| **V-JEPA2 probe (3 seeds)** | no | **0.56 [0.50, 0.73]** | 28% | 1.16 | 34.4 |
| **V-JEPA2 probe (3 seeds)** | yes | 0.54 [0.47, 0.67] | 26% | 1.16 | **1.9** |

- **Hard AP** ranks the windows inside each labeled warning (alert to accident) against the same
  clip's windows from before the alert. Source, camera and editing are held fixed, so it measures
  anticipation rather than "is this a fail video". 95% CIs come from resampling whole compilations.
- **Caught on time**: share of the 81 accidents (those with at least one full 2 s window before them)
  whose first alarm came no earlier than 0.5 s before the labeled alert. Earlier alarms count as
  early, not caught.
- **Cosmos3-Nano zero-shot**: NVIDIA's Cosmos3 Reasoner (Qwen3-VL-8B architecture) sees each window
  as 8 frames at 4 fps and is asked whether an accident is about to happen or starting; the score is
  P(yes) against P(no) for its first answer token, with reasoning off and nothing tuned
  (`ks/models/cosmos_reasoner.py`).
- **YouTube normal**: 2,092 Oops clips that are not labeled accidents, kitchen or not, cut 3 s before
  their failure marker: 178 minutes. Each fold's clips are scored by models that never trained on them.
- Full tables, including the BADAS-style "easy" AP, time-to-accident and calibration:
  `reports/v3_without_youtube/` and `reports/v3_with_youtube/` (`results.md`, `audit.md`); the earlier
  11-minute version is in `reports/v2_*`. `curves.png` shows risk curves for 20 random clips.

### What the audit found

`audit.md` compares each model's scores on calm fail-clip footage (before the alert) with normal
cooking. A model that anticipates should not tell the two apart (AUC 0.5). Trained the usual way, the
single frame does, 93% of the time, and the probe 71%. With the YouTube footage in training the probe
drops to 66% and fires on 0.1% of calm YouTube windows (from 1.8%); the single frame stays at 91%,
so it still recognises kitchen fail clips. The probe fires about 11 times more often inside warnings
than before them; on hard AP its lead over the single frame shrinks to +0.04 [-0.09, +0.12].
Cosmos3-Nano sits between them at 78%: it too scores calm fail-clip footage above cooking, and fires
on 2% of calm YouTube windows.

## Limitations

- One labeler. The planned blind self-agreement recheck was not run.
- Warnings are short (median 0.43 s), so the task sits close to "spot the failure as it starts".
- Too few clips for a fixed test split: every number is out-of-fold over 5 folds, with wide CIs.
- False alarms are measured on Charades, EPIC-KITCHENS and Oops footage. Real kitchens with a fixed
  camera have not been tested.
- Only frozen features were tried. No fine-tuning, which would likely overfit 89 clips.

## Next steps

1. More and longer accident clips, cut with at least 5 s of footage before the fail.
2. Only then, fine-tuning (from V-JEPA2 or BADAS-Open weights).

## Pipeline

| step | where | command |
|---|---|---|
| Oops annotations and videos, Charades, EPIC subset, normalized to 8 fps, 256 px | GPU box | `python -m ks.data.prepare {oops-annotations,oops-videos,oops-normalize,charades,epic} --out ...` |
| Kitchen filter (SigLIP zero-shot + titles) | GPU box | `python -m ks.data.filter_kitchen --videos <dir> --ann <dir> --out candidates.csv` |
| Labeling: keep/reject, hazard, alert and event marks, several accidents per clip | Mac, browser | `python -m ks.label.server --candidates data/candidates.csv --videos data/label_clips --labels data/labels.csv` |
| Manifest with 5 group folds and YouTube normal footage | Mac | `python -m ks.data.build --candidates data/candidates.csv --labels data/labels.csv --negatives data/norm/*/negatives.csv --oops-dir data/norm/oops --pasbench-meta data/pasbench_metadata.csv --out data/manifest.csv` |
| Feature cache: V-JEPA2 tokens, BADAS-Open score, SigLIP embedding | GPU box | `python -m ks.features --manifest data/manifest.csv --out features` |
| Cross-validated probe and baselines | GPU box | `python -m ks.eval.cv --features features --out reports/<run> [--train-without oops]` |
| Cosmos3-Nano zero-shot score, added to a finished run without retraining | GPU box, then Mac | `python -m ks.features --manifest data/manifest.csv --out features_cosmos --models cosmos`, then `python -m ks.eval.cv --features features_cosmos --scores reports/<run>/scores.csv --add-zero-shot cosmos --out reports/<run>` |
| Shortcut audit, outcome by hazard, risk curves | Mac | `python -m ks.eval.audit --features features --run reports/<run>` |
| Example videos: 2 random clips per outcome, risk drawn in as they play, plus an `index.html` gallery | Mac | `python -m ks.demo --features features --run reports/<run> --out demos` |

The example videos contain source footage, so `demos/` stays out of git.

`scripts/box_setup.sh` and `scripts/run_v3.sh` reproduce the GPU part on a rented RTX 4090: about 1 h of downloads, 1 h
of features and 1 h of CV, roughly $1.60. BADAS-Open is gated on Hugging Face. Accept its terms, then
point `BADAS_CKPT` at a local copy of `weights/badas_open.pth`. Its checkpoint runs V-JEPA2's
predictor 1 s into the future and feeds those tokens to the head; `Anticipator(future_fraction=...)`
reproduces that. `scripts/run_cosmos.sh` scores the same windows with Cosmos3-Nano (not gated, ~20 GB
of GPU memory) on a 4090 at about 13 windows/s, 1.4 h; the run here cost $2.83, a repeated Charades
download included.

## Data rights

This repo never contains video. It publishes manifests (video IDs + timestamps), labels, code and
metrics only.
`labels/oops_kitchen_labels.csv` is a snapshot of the hand labels: 666 Oops clips, each kept or
rejected, with alert and accident times and a hazard type per accident (one labeler). The labeling
tool writes to `data/labels.csv`. The source datasets are Oops! (unofficial mirror, no license), EPIC-KITCHENS-100
(CC BY-NC 4.0), Charades (non-commercial research) and PaSBench-Video (CC BY-NC 4.0). Use them for
research only.

## Development

```bash
uv sync                    # core (labels, metrics, splits)
uv sync --extra model      # + torch / transformers / av for model work
uv run pytest
```
