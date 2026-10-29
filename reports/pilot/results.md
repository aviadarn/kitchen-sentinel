89 positive clips, 81 accidents with a visible lead-up; 14.7 h of normal cooking. Out-of-fold over 5 folds; tau per fold for <= 1 false alarm/h on its validation fold. Of the accidents, 'on time' = first alarm no earlier than 0.5 s before the labeled alert; 'early' = before that (the clip looks like a fail video, but no danger is visible yet).

| method | hard AP [95% CI] | hard AUC | easy AP @0.25s | easy AP @0.5s | easy AP @1.0s | recall @tau | on time | early | mTTA (s) | FA/h @tau | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prior | 0.228 [0.190, 0.271] | 0.500 | 0.002 | 0.001 | 0.001 | 0.00 | 0.00 | 0.00 | nan | 0.00 | 0.272 |
| badas | 0.242 [0.184, 0.342] | 0.522 | 0.003 | 0.002 | 0.002 | 0.00 | 0.00 | 0.00 | nan | 1.22 | 0.572 |
| siglip_lr | 0.470 [0.387, 0.635] | 0.709 | 0.305 | 0.205 | 0.141 | 0.59 | 0.41 | 0.19 | 1.89 | 0.95 | 0.290 |
| vjepa_lr | 0.445 [0.386, 0.622] | 0.693 | 0.068 | 0.043 | 0.018 | 0.09 | 0.07 | 0.01 | 1.41 | 0.88 | 0.239 |
| probe (3 seeds) | 0.561 [0.495, 0.726] | 0.745 | 0.238 | 0.185 | 0.060 | 0.34 | 0.28 | 0.07 | 1.41 | 1.16 | 0.173 |

Gate (probe beats single-frame SigLIP on hard AP): delta = +0.091 [+0.011, +0.168]
