89 positive clips, 81 accidents with a visible lead-up; 14.7 h of normal cooking; 178 min of YouTube normal footage (Oops clips before their failure). Out-of-fold over 5 folds; tau per fold for <= 1 false alarm/h on its validation fold's normal cooking. Of the accidents, 'on time' = first alarm no earlier than 0.5 s before the labeled alert; 'early' = before that (the clip looks like a fail video, but no danger is visible yet).

| method | hard AP [95% CI] | hard AUC | easy AP @0.25s | easy AP @0.5s | easy AP @1.0s | recall @tau | on time | early | mTTA (s) | FA/h @tau | FA/h YouTube normal | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prior | 0.228 [0.190, 0.271] | 0.500 | 0.001 | 0.001 | 0.001 | 0.00 | 0.00 | 0.00 | nan | 0.00 | 0.0 | 0.272 |
| badas | 0.242 [0.184, 0.342] | 0.522 | 0.002 | 0.002 | 0.002 | 0.00 | 0.00 | 0.00 | nan | 1.22 | 7.1 | 0.572 |
| cosmos | 0.417 [0.303, 0.616] | 0.649 | 0.047 | 0.018 | 0.005 | 0.19 | 0.11 | 0.07 | 2.98 | 1.43 | 37.4 | 0.110 |
| siglip_lr | 0.498 [0.407, 0.705] | 0.722 | 0.217 | 0.147 | 0.097 | 0.59 | 0.41 | 0.19 | 2.10 | 1.36 | 8.1 | 0.227 |
| vjepa_lr | 0.394 [0.298, 0.509] | 0.686 | 0.061 | 0.043 | 0.017 | 0.07 | 0.04 | 0.04 | 2.18 | 0.48 | 1.7 | 0.234 |
| probe (3 seeds) | 0.540 [0.466, 0.674] | 0.732 | 0.231 | 0.166 | 0.052 | 0.33 | 0.26 | 0.07 | 1.39 | 1.16 | 1.9 | 0.180 |

Gate (probe beats single-frame SigLIP on hard AP): delta = +0.042 [-0.092, +0.117]
