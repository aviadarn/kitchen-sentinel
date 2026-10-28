import math

import numpy as np
import pytest

from ks.eval.metrics import (
    expected_calibration_error,
    false_alarms,
    false_alarms_per_hour,
    group_bootstrap_ci,
    mean_tta,
    time_to_accident,
)

TIMES = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_tta_is_event_minus_first_crossing():
    probs = [0.1, 0.2, 0.6, 0.9, 0.9, 0.9]
    assert time_to_accident(TIMES, probs, tau=0.5, event=6.5) == pytest.approx(2.5)


def test_tta_none_when_never_crossing():
    probs = [0.1, 0.2, 0.6, 0.9, 0.9, 0.9]
    assert time_to_accident(TIMES, probs, tau=0.95, event=6.5) is None


def test_tta_ignores_crossings_after_the_event():
    probs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.9]
    assert time_to_accident(TIMES, probs, tau=0.5, event=6.5) is None


def test_tta_counts_crossing_exactly_at_event_as_zero_warning():
    probs = [0.1, 0.1, 0.1, 0.1, 0.9, 0.9]
    assert time_to_accident(TIMES, probs, tau=0.5, event=6.0) == pytest.approx(0.0)


def test_mean_tta_averages_detected_and_reports_recall():
    curves = [
        (TIMES, [0.1, 0.2, 0.6, 0.9, 0.9, 0.9], 6.5),  # caught, tta 2.5
        (TIMES, [0.1, 0.1, 0.1, 0.1, 0.9, 0.9], 6.5),  # caught, tta 0.5
        (TIMES, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 6.5),  # missed
    ]
    mtta, recall = mean_tta(curves, tau=0.5)
    assert mtta == pytest.approx(1.5)
    assert recall == pytest.approx(2 / 3)


def test_mean_tta_is_nan_when_nothing_detected():
    mtta, recall = mean_tta([(TIMES, [0.0] * 6, 6.5)], tau=0.5)
    assert math.isnan(mtta)
    assert recall == 0.0


# times 0..8, alarm state:   .  A  A  .  A  .  .  .  A
FA_TIMES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
FA_PROBS = [0.0, 0.9, 0.9, 0.0, 0.9, 0.0, 0.0, 0.0, 0.9]


def test_false_alarms_counts_rising_edges():
    # Sustained high probability is one alarm, not one per sample.
    assert false_alarms(FA_TIMES, FA_PROBS, tau=0.5, refractory=2.0) == 3


def test_false_alarms_merges_edges_inside_refractory():
    # Edge at 4 is 3 s after the alarm at 1 -> merged; edge at 8 is 7 s after -> counted.
    assert false_alarms(FA_TIMES, FA_PROBS, tau=0.5, refractory=3.5) == 2


def test_false_alarms_sustained_danger_re_alarms_every_refractory_period():
    # Stuck above threshold for 10 s: a device keeps warning (0, 2, 4, 6, 8), it doesn't go quiet.
    # Counting only rising edges would let a model that is always "on" score 1 alarm per hour.
    times = [float(t) for t in range(10)]
    assert false_alarms(times, [0.9] * 10, tau=0.5, refractory=2.0) == 5


def test_false_alarms_counts_alarm_on_first_sample():
    assert false_alarms([0.0, 1.0], [0.9, 0.9], tau=0.5, refractory=2.0) == 1


def test_false_alarms_per_hour_scales_by_duration():
    assert false_alarms_per_hour([(FA_TIMES, FA_PROBS, 1800.0)], tau=0.5, refractory=2.0) == pytest.approx(6.0)


def test_false_alarms_per_hour_pools_videos_before_dividing():
    # 3 alarms in 1800 s + 0 alarms in 5400 s = 3 alarms / 2 h = 1.5 per hour.
    # Averaging per-video rates instead would give (6 + 0) / 2 = 3.0.
    quiet = ([0.0, 1.0], [0.0, 0.0], 5400.0)
    assert false_alarms_per_hour([(FA_TIMES, FA_PROBS, 1800.0), quiet], tau=0.5, refractory=2.0) == pytest.approx(1.5)


def test_ece_known_miscalibration():
    # bin 0.2-0.3: conf .25, acc 0 -> gap .25 (1 of 4); bin 0.7-0.8: conf .75, acc 1 -> gap .25 (3 of 4)
    assert expected_calibration_error([0, 1, 1, 1], [0.25, 0.75, 0.75, 0.75], n_bins=10) == pytest.approx(0.25)


def test_ece_perfect_calibration_is_zero():
    assert expected_calibration_error([0, 1], [0.5, 0.5], n_bins=10) == pytest.approx(0.0)


def test_ece_puts_probability_one_in_last_bin():
    # Shared bin 0.9-1.0: acc .5, conf .975 -> .475. A separate overflow bin for p=1.0
    # would give |1-.95|*.5 + |0-1|*.5 = .525 instead.
    assert expected_calibration_error([1, 0], [0.95, 1.0], n_bins=10) == pytest.approx(0.475)


def test_group_bootstrap_resamples_whole_groups():
    # g1 = four 1s, g2 = one 0. Group resampling draws {g1,g1}, {g1,g2} or {g2,g2},
    # so the mean is 0 with prob 1/4 and 1 with prob 1/4: the 95% CI is [0, 1].
    # Row resampling would almost never produce a mean of 0 (needs all 5 draws = the single 0).
    y = np.array([1, 1, 1, 1, 0])
    groups = ["g1", "g1", "g1", "g1", "g2"]
    lo, hi = group_bootstrap_ci(lambda idx: float(y[idx].mean()), groups, n_boot=2000, seed=0)
    assert lo == 0.0
    assert hi == 1.0


def test_group_bootstrap_skips_nan_resamples():
    # A metric undefined on some resamples (e.g. AP with one class) must not poison the CI.
    y = np.array([1, 0])
    groups = ["a", "b"]

    def metric(idx):
        sample = y[idx]
        return float("nan") if sample.min() == sample.max() else 0.5

    lo, hi = group_bootstrap_ci(metric, groups, n_boot=500, seed=0)
    assert (lo, hi) == (0.5, 0.5)
