"""Evaluation sets built from per-window predictions.

`preds` has one row per window: video_id, group, video_label (1 = positive video), t_end, window_label
(1 / 0 / NaN for windows the training labels drop), score, time_of_alert, time_of_event, duration.
"""

import numpy as np

from ks.eval.metrics import false_alarms_per_hour


def hard_set(preds):
    """Positive windows vs pre-alert windows of the *same* positive videos: source and style held fixed."""
    p = preds[(preds.video_label == 1) & preds.window_label.notna()].sort_values(["video_id", "t_end"])
    return p.window_label.astype(int).to_numpy(), p.score.to_numpy(), p.group.to_numpy()


def easy_set(preds, offset, tolerance=0.13):
    """BADAS/Nexar-style: per positive video, the window ending `offset` s before the event, against
    every window of the negative videos. Inflated by any source shortcut; reported next to hard_set."""
    pos_y, pos_s, pos_g = [], [], []
    for _, v in preds[preds.video_label == 1].groupby("video_id", sort=True):
        target = v.time_of_event.iloc[0] - offset
        gap = (v.t_end - target).abs()
        if gap.min() <= tolerance:
            w = v.loc[gap.idxmin()]
            pos_y.append(1), pos_s.append(w.score), pos_g.append(w.group)
    neg = preds[preds.video_label == 0].sort_values(["video_id", "t_end"])
    return (np.array(pos_y + [0] * len(neg), dtype=int), np.concatenate([pos_s, neg.score.to_numpy()]),
            np.concatenate([pos_g, neg.group.to_numpy()]))


def risk_curves(preds, positive):
    """Positive videos -> (times, probs, event) up to the event; negatives -> (times, probs, duration)."""
    curves = []
    for _, v in preds[preds.video_label == (1 if positive else 0)].groupby("video_id", sort=True):
        v = v.sort_values("t_end")
        if positive:
            event = float(v.time_of_event.iloc[0])
            v = v[v.t_end <= event]
            curves.append((v.t_end.tolist(), v.score.tolist(), event))
        else:
            curves.append((v.t_end.tolist(), v.score.tolist(), float(v.duration.iloc[0])))
    return curves


def choose_tau(negative_curves, max_fa_per_hour, refractory=2.0):
    """Lowest threshold whose false-alarm rate on `negative_curves` stays within budget.

    Alarm counts only fall as tau rises (fewer samples above it can't fit more alarms one refractory
    apart), so a binary search over the observed scores and the values just above them is exact.
    """
    scores = np.unique(np.concatenate([np.asarray(p, dtype=float) for _, p, _ in negative_curves]))
    candidates = np.unique(np.concatenate([scores, np.nextafter(scores, np.inf)]))
    lo, hi = 0, len(candidates) - 1  # candidates[-1] is above every score: zero alarms
    while lo < hi:
        mid = (lo + hi) // 2
        if false_alarms_per_hour(negative_curves, candidates[mid], refractory) <= max_fa_per_hour:
            hi = mid
        else:
            lo = mid + 1
    return float(candidates[lo])
