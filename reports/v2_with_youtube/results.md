89 positive clips, 81 accidents with a visible lead-up; 14.7 h of normal cooking; 11 min of YouTube normal footage (Oops clips before their failure). Out-of-fold over 5 folds; tau per fold for <= 1 false alarm/h on its validation fold's normal cooking. Of the accidents, 'on time' = first alarm no earlier than 0.5 s before the labeled alert; 'early' = before that (the clip looks like a fail video, but no danger is visible yet).

| method | hard AP [95% CI] | hard AUC | easy AP @0.25s | easy AP @0.5s | easy AP @1.0s | recall @tau | on time | early | mTTA (s) | FA/h @tau | FA/h YouTube normal | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prior | 0.228 [0.190, 0.271] | 0.500 | 0.002 | 0.001 | 0.001 | 0.00 | 0.00 | 0.00 | nan | 0.00 | 0.0 | 0.272 |
| badas | 0.242 [0.184, 0.342] | 0.522 | 0.003 | 0.002 | 0.002 | 0.00 | 0.00 | 0.00 | nan | 1.22 | 11.2 | 0.572 |
| siglip_lr | 0.425 [0.380, 0.684] | 0.705 | 0.276 | 0.184 | 0.119 | 0.54 | 0.33 | 0.21 | 2.18 | 2.24 | 55.8 | 0.221 |
| vjepa_lr | 0.397 [0.325, 0.499] | 0.674 | 0.056 | 0.038 | 0.015 | 0.06 | 0.05 | 0.01 | 0.58 | 0.41 | 0.0 | 0.199 |
| probe (3 seeds) | 0.529 [0.453, 0.680] | 0.736 | 0.243 | 0.182 | 0.053 | 0.33 | 0.26 | 0.07 | 1.28 | 1.81 | 18.6 | 0.184 |

Gate (probe beats single-frame SigLIP on hard AP): delta = +0.104 [-0.061, +0.149]
