import pytest

from ks.data.windows import Window, sample_windows, window_label


# A window is labeled by the time its last frame ends (t_end):
#   alert <= t_end <= event      -> 1 (danger is visible, accident not yet happened)
#   t_end <  alert - gap         -> 0 (same-video hard negative)
#   alert - gap <= t_end < alert -> None (ambiguous, dropped)
#   t_end >  event               -> None (post-accident chaos, dropped)


@pytest.mark.parametrize(
    "t_end, want",
    [
        (5.0, 0),     # well before alert
        (5.49, 0),    # just outside the gap
        (5.5, None),  # gap starts exactly at alert - gap
        (5.75, None),
        (6.0, 1),     # alert itself is positive
        (7.5, 1),     # event itself is positive
        (7.51, None), # after the event
    ],
)
def test_window_label_positive_video(t_end, want):
    assert window_label(t_end, events=[(6.0, 7.5)], gap=0.5) == want


def test_window_label_negative_video_is_always_zero():
    assert window_label(3.0, events=[]) == 0
    assert window_label(300.0, events=[]) == 0


def test_window_label_rejects_alert_after_event():
    with pytest.raises(ValueError):
        window_label(3.0, events=[(8.0, 7.0)])


def test_window_label_rejects_overlapping_events():
    with pytest.raises(ValueError):
        window_label(3.0, events=[(1.0, 5.0), (4.0, 7.0)])


def test_sample_windows_positive_video_keeps_only_labeled_windows():
    # 10 s at 8 fps = 80 frames; 16-frame windows (2 s) every 8 frames (1 s).
    # t_end = 2,3,...,10. 8, 9 and 10 are after the event and are dropped.
    got = sample_windows(duration=10.0, events=[(6.0, 7.5)], fps=8.0, num_frames=16, stride_frames=8, gap=0.5)
    assert [(w.t_end, w.label) for w in got] == [(2.0, 0), (3.0, 0), (4.0, 0), (5.0, 0), (6.0, 1), (7.0, 1)]


def test_sample_windows_reports_frame_span():
    got = sample_windows(duration=3.0, events=[], fps=8.0, num_frames=16, stride_frames=8)
    # 24 frames -> windows ending at frame 16 and 24.
    assert got == [Window(start_frame=0, end_frame=16, t_end=2.0, label=0), Window(start_frame=8, end_frame=24, t_end=3.0, label=0)]


def test_sample_windows_never_pads_a_short_video():
    assert sample_windows(duration=1.5, events=[], fps=8.0, num_frames=16, stride_frames=8) == []


def test_sample_windows_drops_partial_trailing_frames():
    # 4.5 s -> 36 frames; ends at 16, 24, 32 only (40 would need frames that don't exist).
    got = sample_windows(duration=4.5, events=[], fps=8.0, num_frames=16, stride_frames=8)
    assert [w.t_end for w in got] == [2.0, 3.0, 4.0]


def test_sample_windows_positive_needs_context_before_alert():
    # Alert at 1.0 s: every window that ends in [alert, event] would need padding
    # before t=0 except those ending at >= 2.0 s.
    got = sample_windows(duration=3.0, events=[(1.0, 2.5)], fps=8.0, num_frames=16, stride_frames=4)
    assert [(w.t_end, w.label) for w in got] == [(2.0, 1), (2.5, 1)]


def test_sample_windows_keep_unlabeled_returns_every_window_for_risk_curves():
    # Evaluation needs the full curve: the gap before the alert and post-event windows come back as None.
    got = sample_windows(duration=10.0, events=[(6.0, 7.5)], fps=8.0, num_frames=16, stride_frames=4, gap=0.5,
                         keep_unlabeled=True)
    assert [(w.t_end, w.label) for w in got][6:10] == [(5.0, 0), (5.5, None), (6.0, 1), (6.5, 1)]
    assert got[-1].t_end == 10.0 and got[-1].label is None


# Two accidents in one clip: alert 4.0 -> event 5.0, then alert 8.0 -> event 9.0.
TWO = [(4.0, 5.0), (8.0, 9.0)]


@pytest.mark.parametrize(
    "t_end, want",
    [
        (3.0, 0),      # before the first alert (minus gap): safe
        (3.75, None),  # gap before first alert
        (4.5, 1),      # first warning window
        (5.0, 1),
        (6.0, None),   # after the first accident: aftermath is not "safe cooking", dropped
        (7.6, None),   # gap before the second alert
        (8.0, 1),      # second warning window
        (9.0, 1),
        (9.5, None),   # after the last accident
    ],
)
def test_window_label_multi_event(t_end, want):
    assert window_label(t_end, events=TWO, gap=0.5) == want


def test_window_label_accepts_events_in_any_order():
    assert window_label(8.5, events=list(reversed(TWO)), gap=0.5) == 1
