"""Sliding windows over a video and their BADAS-style labels.

Videos are resampled to `fps` before windowing. A window is the `num_frames` frames that end
(exclusive) at `end_frame`, so it covers [t_end - num_frames / fps, t_end) with t_end = end_frame / fps.
The model sees the window and predicts whether an accident is coming, so a window is labeled by t_end.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    start_frame: int
    end_frame: int
    t_end: float
    label: int


def window_label(t_end, events, gap=0.5):
    """1 if danger is visible and an accident is about to happen, 0 if clearly safe, None to drop.

    `events` holds (alert, event) pairs, one per accident in the clip; a negative (normal-cooking)
    video has none. A window is positive inside any alert..event span. It is a negative only before
    the first alert (minus `gap`, where annotators disagree about the exact moment). Everything after
    the first accident is dropped unless it is inside a later warning span: aftermath footage is not
    safe cooking, and post-accident chaos would teach "chaos = danger".
    """
    if not events:
        return 0
    events = sorted(events)
    for (alert, event), nxt in zip(events, events[1:] + [None]):
        if alert > event:
            raise ValueError(f"alert {alert} is after event {event}")
        if nxt is not None and event > nxt[0]:
            raise ValueError(f"events overlap: {alert}-{event} and {nxt[0]}-{nxt[1]}")
    if any(alert <= t_end <= event for alert, event in events):
        return 1
    if t_end < events[0][0] - gap:
        return 0
    return None


def sample_windows(duration, events, fps=8.0, num_frames=16, stride_frames=8, gap=0.5, keep_unlabeled=False):
    """All labeled windows of a video (with keep_unlabeled, every window, label None where training
    would drop it). Windows that would need padding are never produced."""
    total_frames = math.floor(duration * fps + 1e-9)
    windows = []
    for end in range(num_frames, total_frames + 1, stride_frames):
        t_end = end / fps
        label = window_label(t_end, events, gap)
        if label is not None or keep_unlabeled:
            windows.append(Window(start_frame=end - num_frames, end_frame=end, t_end=t_end, label=label))
    return windows
