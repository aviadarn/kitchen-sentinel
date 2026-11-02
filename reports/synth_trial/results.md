# Synthetic matched pairs: trial

Wan 2.2 TI2V-5B (diffusers 0.42.1, 40 steps, 1088x832, 121 frames at 24 fps) continued 11 real Charades
kitchen frames twice each: once into an accident, once with nothing happening, same seed. Config:
`configs/synth_trial.csv`. One RTX 4090, 425 s per clip on average, about $1.85 for the trial in all.
Videos are not in the repo (they derive from Charades); `data/synth/trial/index.html` shows them locally.

## By eye (one reviewer)

| pair | hazard | accident clip |
|---|---|---|
| p01 | fire: frying pan flares | convincing flames; the cook's face changes once it flares |
| p02 | scald: pot of boiling water tips | pot ends up on the floor, the tip itself is never shown |
| p03 | dish slips and shatters | camera dives to the floor, the kitchen rearranges, the person vanishes |
| p04 | glass knocked off the counter | breaks, but the viewpoint jumps and the room changes |
| p05 | bottle knocked off the table | chaotic camera, the bottle never visibly falls |
| p06 | jar falls out of the fridge | nothing happens: the prompt's accident is ignored |
| p07 | stack of pots topples | convincing |
| p08 | slip on a wet floor | convincing |
| p09 | mug of hot tea dropped | camera drops to the feet; no mug breaks |
| p10 | towel catches fire on the stove | flame catches, but the electric stove turns into a gas burner |
| p11 | pan flares up | convincing flames; the pan lifts oddly towards her face |

Four of eleven accidents are usable as they are, two partly, five fail. Every safe twin stays ordinary.
Fire and toppling or falling bodies work; small objects breaking and liquids spilling do not, and the
failures share a pattern: the camera swings to where the accident should be and the scene changes.

## Can frozen features tell them apart?

2 s windows, 1 s apart, 44 per kind; logistic regression scored with each pair held out in turn
(`python -m ks.synth.detect`).

| comparison | V-JEPA2 mean token | SigLIP last frame |
|---|---|---|
| generated (both twins) vs. the real source footage | 0.83 | 0.70 |
| safe twin vs. the real source footage | 0.67 | 0.64 |
| accident twin vs. safe twin | 0.81 | 0.90 |

The calm twins sit fairly close to real footage, so most of the generated-vs-real signal comes from the
accident clips, which is where the glitches are. Accident vs. safe is separable, but with 11 pairs this
cannot say whether the features see the danger or the glitches. These are small samples: each AUC
rests on 11 held-out pairs.

## Verdict

Usable for fire and falls, not yet for breaking objects or spills. Before scaling: keep only clips that
pass review, prompt for a fixed camera, and test whether a model trained with them improves on the
real hard set, which is the only measure that counts.
