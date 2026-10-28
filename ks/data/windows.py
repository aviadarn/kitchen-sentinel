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


def window_label(t_end, alert, event, gap=0.5):
    """1 if danger is visible and the accident hasn't happened, 0 if clearly safe, None to drop.

    `alert`/`event` are None for a negative (normal-cooking) video. The `gap` before the alert is
    dropped because annotators disagree about the exact alert moment.
    """
    if alert is None and event is None:
        return 0
    if alert is None or event is None:
        raise ValueError("a positive video needs both alert and event times")
    if alert > event:
        raise ValueError(f"alert {alert} is after event {event}")
    if t_end > event:
        return None  # post-accident frames would teach "chaos = danger"
    if t_end >= alert:
        return 1
    if t_end < alert - gap:
        return 0
    return None


def sample_windows(duration, alert, event, fps=8.0, num_frames=16, stride_frames=8, gap=0.5):
    """All labeled windows of a video. Windows that would need padding are never produced."""
    total_frames = math.floor(duration * fps + 1e-9)
    windows = []
    for end in range(num_frames, total_frames + 1, stride_frames):
        t_end = end / fps
        label = window_label(t_end, alert, event, gap)
        if label is not None:
            windows.append(Window(start_frame=end - num_frames, end_frame=end, t_end=t_end, label=label))
    return windows
