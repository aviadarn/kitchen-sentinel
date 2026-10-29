"""Evaluation sets built from per-window predictions.

`preds` has one row per window: video_id, group, video_label (1 = positive video), t_end, window_label
(1 / 0 / NaN for windows the training labels drop), score, events, duration. `events` lists the clip's
accidents as (alert, event) pairs or the manifest's JSON; a clip may hold several, and warning time and
the easy set are computed per accident.
"""

import json

import numpy as np

from ks.eval.metrics import false_alarms_per_hour


def hard_set(preds):
    """Positive windows vs pre-alert windows of the *same* positive videos: source and style held fixed."""
    p = preds[(preds.video_label == 1) & preds.window_label.notna()].sort_values(["video_id", "t_end"])
    return p.window_label.astype(int).to_numpy(), p.score.to_numpy(), p.group.to_numpy()


def event_times(events):
    """Sorted accident times from (alert, event) pairs, event dicts, or their JSON."""
    if isinstance(events, str):
        events = json.loads(events)
    return sorted(e["event"] if isinstance(e, dict) else e[1] for e in events)


def easy_set(preds, offset, tolerance=0.13):
    """BADAS/Nexar-style: per accident, the window ending `offset` s before it, against every window of
    the negative videos. Inflated by any source shortcut; reported next to hard_set."""
    pos_y, pos_s, pos_g = [], [], []
    for _, v in preds[preds.video_label == 1].groupby("video_id", sort=True):
        for event in event_times(v.events.iloc[0]):
            gap = (v.t_end - (event - offset)).abs()
            if gap.min() <= tolerance:
                w = v.loc[gap.idxmin()]
                pos_y.append(1), pos_s.append(w.score), pos_g.append(w.group)
    neg = preds[preds.video_label == 0].sort_values(["video_id", "t_end"])
    return (np.array(pos_y + [0] * len(neg), dtype=int), np.concatenate([pos_s, neg.score.to_numpy()]),
            np.concatenate([pos_g, neg.group.to_numpy()]))


def accident_curves(preds):
    """One dict (video_id, times, probs, alert, event, hazard) per accident of the positive videos, over
    the windows after the previous accident up to this one."""
    curves = []
    for video_id, v in preds[preds.video_label == 1].groupby("video_id", sort=True):
        v = v.sort_values("t_end")
        events = v.events.iloc[0]
        events = json.loads(events) if isinstance(events, str) else events
        events = sorted(({"alert": e[0], "event": e[1]} if not isinstance(e, dict) else e for e in events),
                        key=lambda e: e["event"])
        previous = -np.inf
        for e in events:
            seg = v[(v.t_end > previous) & (v.t_end <= e["event"])]
            curves.append(dict(video_id=video_id, times=seg.t_end.tolist(), probs=seg.score.tolist(),
                               alert=float(e["alert"]), event=float(e["event"]), hazard=e.get("hazard")))
            previous = e["event"]
    return curves


def risk_curves(preds, positive):
    """Positive videos -> one (times, probs, event) per accident (see accident_curves); negatives ->
    (times, probs, duration)."""
    if positive:
        return [(c["times"], c["probs"], c["event"]) for c in accident_curves(preds)]
    return [(v.t_end.tolist(), v.score.tolist(), float(v.duration.iloc[0]))
            for _, v in preds[preds.video_label == 0].sort_values("t_end").groupby("video_id", sort=True)]


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
