import numpy as np

from ks.data.sources.youtube import duplicate_of, parse_scene_times, segments, within_duplicates


def test_scene_times_come_from_ffmpeg_showinfo_lines():
    log = """[Parsed_showinfo_1 @ 0x1] n:   0 pts:  51200 pts_time:4.16667 duration: 512 fmt:yuv420p
[Parsed_showinfo_1 @ 0x1] n:   1 pts: 123904 pts_time:10.0833 duration: 512
frame=  2 fps=0.0 q=-0.0 Lsize=N/A time=00:00:10.08"""
    assert parse_scene_times(log) == [4.16667, 10.0833]


def test_segments_span_the_gaps_between_cuts_and_skip_short_ones():
    # cuts at 4.0 and 5.5 and 20.0 in a 30 s video; the 1.5 s scene is too short to label
    assert segments([4.0, 5.5, 20.0], duration=30.0, min_len=3.0, pad=0.1) == [
        (0.0, 3.9), (5.6, 19.9), (20.1, 30.0)]


def test_segments_of_a_video_without_cuts_is_the_whole_video():
    assert segments([], duration=12.0) == [(0.0, 12.0)]


def unit(rows):
    rows = np.asarray(rows, dtype=float)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_duplicate_of_finds_new_clips_that_match_an_existing_one():
    ref = unit([[1, 0, 0], [0, 1, 0]])
    new = unit([[0.99, 0.05, 0], [0, 0, 1]])
    assert duplicate_of(new, ref, threshold=0.95).tolist() == [0, -1]


def test_within_duplicates_keeps_the_first_of_each_repeated_clip():
    # The same viral clip shows up in several compilations; label it once.
    emb = unit([[1, 0, 0], [0, 1, 0], [0.99, 0.02, 0], [0, 0.98, 0.1]])
    assert within_duplicates(emb, threshold=0.95).tolist() == [-1, -1, 0, 1]
