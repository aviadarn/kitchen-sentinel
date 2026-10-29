## Shortcut audit

pre-alert AUC: pre-alert windows of fail clips vs normal-cooking windows (0.5 = can't tell them apart before danger is visible). Fires: share of windows at or above the fold's alarm threshold.

| method | pre-alert AUC | pre-alert | gap | warning | after | youtube | charades | epic |
|---|---|---|---|---|---|---|---|---|
| badas | 0.682 | 0.000 | 0.000 | 0.000 | 0.000 | 0.006 | 0.000 | 0.000 |
| prior | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| probe | 0.698 | 0.017 | 0.104 | 0.225 | 0.095 | 0.010 | 0.001 | 0.000 |
| siglip_lr | 0.884 | 0.110 | 0.111 | 0.331 | 0.234 | 0.032 | 0.001 | 0.000 |
| vjepa_lr | 0.704 | 0.001 | 0.000 | 0.017 | 0.017 | 0.000 | 0.000 | 0.000 |

## Outcome by hazard (share of accidents; probe averaged over seeds)

| method | hazard | n | on time | early | miss |
|---|---|---|---|---|---|
| probe | appliance | 1 | 0.00 | 0.00 | 1.00 |
| probe | burn_scald | 7 | 0.43 | 0.05 | 0.52 |
| probe | falling_breaking | 42 | 0.33 | 0.06 | 0.60 |
| probe | fire | 7 | 0.19 | 0.29 | 0.52 |
| probe | impact | 3 | 0.00 | 0.00 | 1.00 |
| probe | other | 3 | 0.11 | 0.00 | 0.89 |
| probe | slip_fall | 18 | 0.13 | 0.06 | 0.81 |
| siglip_lr | appliance | 1 | 0.00 | 0.00 | 1.00 |
| siglip_lr | burn_scald | 7 | 0.71 | 0.14 | 0.14 |
| siglip_lr | falling_breaking | 42 | 0.33 | 0.24 | 0.43 |
| siglip_lr | fire | 7 | 0.43 | 0.14 | 0.43 |
| siglip_lr | impact | 3 | 0.33 | 0.00 | 0.67 |
| siglip_lr | other | 3 | 0.00 | 0.33 | 0.67 |
| siglip_lr | slip_fall | 18 | 0.22 | 0.22 | 0.56 |

![risk curves](curves.png)
