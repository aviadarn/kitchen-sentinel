"""Anticipation metrics that sklearn doesn't provide (AP/AUC come straight from sklearn).

A risk curve is a pair (times, probs): the model's P(danger) for windows ending at each time.
"""

import math

import numpy as np


def time_to_accident(times, probs, tau, event):
    """Seconds of warning: event minus the first time P >= tau at or before the event; None if missed."""
    for t, p in zip(times, probs):
        if t > event:
            break
        if p >= tau:
            return event - t
    return None


def mean_tta(curves, tau):
    """(mean TTA over detected positives, recall) for curves given as (times, probs, event)."""
    ttas = [time_to_accident(times, probs, tau, event) for times, probs, event in curves]
    detected = [x for x in ttas if x is not None]
    mtta = float(np.mean(detected)) if detected else math.nan
    return mtta, len(detected) / len(curves)


def false_alarms(times, probs, tau, refractory=2.0):
    """Alarms on a negative video: rising edges of P >= tau, at most one per `refractory` seconds.

    A sustained high probability is one alarm. An edge within `refractory` of the last counted alarm
    is merged into it, the way a device would not re-sound a warning it just gave.
    """
    count, last_alarm, was_above = 0, None, False
    for t, p in zip(times, probs):
        above = p >= tau
        if above and not was_above and (last_alarm is None or t - last_alarm >= refractory):
            count += 1
            last_alarm = t
        was_above = above
    return count


def false_alarms_per_hour(videos, tau, refractory=2.0):
    """Pooled alarm rate over negative videos given as (times, probs, duration_seconds)."""
    alarms = sum(false_alarms(times, probs, tau, refractory) for times, probs, _ in videos)
    hours = sum(duration for _, _, duration in videos) / 3600.0
    return alarms / hours


def expected_calibration_error(y, p, n_bins=10):
    """Equal-width-bin ECE: sum over bins of |accuracy - confidence| weighted by bin share."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in np.unique(bins):
        in_bin = bins == b
        ece += abs(y[in_bin].mean() - p[in_bin].mean()) * in_bin.mean()
    return float(ece)


def group_bootstrap_ci(metric, groups, n_boot=1000, seed=0, alpha=0.05):
    """Percentile CI of `metric(row_indices)` resampling whole groups (videos from one compilation
    or participant are correlated, so resampling rows would understate the uncertainty).

    Resamples where the metric is undefined (NaN, e.g. AP with a single class) are skipped.
    """
    groups = np.asarray(groups)
    rows_by_group = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        picks = rng.integers(0, len(rows_by_group), size=len(rows_by_group))
        value = metric(np.concatenate([rows_by_group[i] for i in picks]))
        if not math.isnan(value):
            values.append(value)
    if not values:
        return math.nan, math.nan
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))
