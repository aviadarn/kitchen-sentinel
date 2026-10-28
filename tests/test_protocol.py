import math

import pandas as pd
import pytest

from ks.eval.protocol import choose_tau, easy_set, hard_set, risk_curves


def preds():
    # Two positive videos (event 4.0 / 3.0, alert 3.0 / 2.0) and one negative, windows every 0.5 s.
    rows = []
    for t, y, s in [(2.0, 0, 0.1), (2.5, None, 0.2), (3.0, 1, 0.7), (3.5, 1, 0.8), (4.0, 1, 0.9), (4.5, None, 0.9)]:
        rows.append(dict(video_id="p1", group="gA", video_label=1, t_end=t, window_label=y, score=s,
                         events=[(3.0, 4.0)], duration=5.0))
    for t, y, s in [(2.0, 1, 0.6), (2.5, 1, 0.65), (3.0, 1, 0.5)]:
        rows.append(dict(video_id="p2", group="gB", video_label=1, t_end=t, window_label=y, score=s,
                         events=[(2.0, 3.0)], duration=3.0))
    for t, s in [(2.0, 0.3), (2.5, 0.95), (3.0, 0.2), (3.5, 0.1)]:
        rows.append(dict(video_id="n1", group="gN", video_label=0, t_end=t, window_label=0, score=s,
                         events=[], duration=3600.0))
    return pd.DataFrame(rows)


def test_hard_set_uses_only_labeled_windows_of_positive_videos():
    y, s, g = hard_set(preds())
    assert list(y) == [0, 1, 1, 1, 1, 1, 1]
    assert list(s) == [0.1, 0.7, 0.8, 0.9, 0.6, 0.65, 0.5]
    assert list(g) == ["gA"] * 4 + ["gB"] * 3


def test_easy_set_takes_window_ending_offset_before_event_against_negative_windows():
    y, s, g = easy_set(preds(), offset=1.0)
    # p1: event 4.0 -> window ending 3.0 (0.7); p2: event 3.0 -> window ending 2.0 (0.6); then all n1 windows.
    assert list(y) == [1, 1, 0, 0, 0, 0]
    assert list(s) == [0.7, 0.6, 0.3, 0.95, 0.2, 0.1]
    assert list(g) == ["gA", "gB", "gN", "gN", "gN", "gN"]


def test_easy_set_skips_videos_without_a_window_at_that_offset():
    # p2's event is at 3.0: 1.5 s before is 1.5 s, earlier than any window can end (2.0 s).
    y, _, _ = easy_set(preds(), offset=1.5)
    assert list(y) == [1, 0, 0, 0, 0]


def test_risk_curves_keep_all_windows_up_to_event_in_time_order():
    curves = risk_curves(preds(), positive=True)
    times, probs, event = curves[0]
    assert (times, probs, event) == ([2.0, 2.5, 3.0, 3.5, 4.0], [0.1, 0.2, 0.7, 0.8, 0.9], 4.0)


def test_choose_tau_picks_lowest_threshold_meeting_false_alarm_budget():
    # One hour of negative video with peaks at 0.95, 0.3, 0.2 as separate edges.
    neg = [([0.0, 1.0, 10.0, 11.0, 20.0, 21.0], [0.95, 0.0, 0.3, 0.0, 0.2, 0.0], 3600.0)]
    # Budget 1/hour: only the 0.95 peak may fire, so tau sits just above the 0.3 peak.
    assert 0.3 < choose_tau(neg, max_fa_per_hour=1.0, refractory=2.0) <= 0.95
    # Budget 2/hour admits the 0.3 peak too: tau sits just above 0.2.
    assert 0.2 < choose_tau(neg, max_fa_per_hour=2.0, refractory=2.0) <= 0.3
    # Budget 0: nothing may fire.
    assert choose_tau(neg, max_fa_per_hour=0.0, refractory=2.0) > 0.95


def test_choose_tau_infinite_budget_never_needed():
    assert choose_tau([([0.0], [0.0], 3600.0)], max_fa_per_hour=0.0, refractory=2.0) <= 1.0
    assert not math.isnan(choose_tau([([0.0], [0.0], 3600.0)], max_fa_per_hour=0.0, refractory=2.0))


def two_event_preds():
    # One clip, accidents at 4.0 and 8.0 (alerts 3.0 and 7.0), a window every second.
    rows = [dict(video_id="m", group="gM", video_label=1, t_end=float(t), window_label=None, score=t / 10,
                 events=[(3.0, 4.0), (7.0, 8.0)], duration=10.0) for t in range(2, 11)]
    return pd.DataFrame(rows)


def test_easy_set_has_one_positive_per_event():
    y, s, _ = easy_set(two_event_preds(), offset=1.0)
    assert list(y) == [1, 1]
    assert list(s) == [0.3, 0.7]  # windows ending 3.0 and 7.0


def test_risk_curves_split_a_multi_event_clip_at_each_accident():
    # The second accident's warning time is measured only on windows after the first accident.
    curves = risk_curves(two_event_preds(), positive=True)
    assert [(c[0][0], c[0][-1], c[2]) for c in curves] == [(2.0, 4.0, 4.0), (5.0, 8.0, 8.0)]
