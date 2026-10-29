import json
import math

import numpy as np
import pandas as pd
import pytest

from ks.features import chunk_plan, extract, feature_index


def manifest():
    return pd.DataFrame([
        dict(video_id="p", source="oops", path="p.mp4", view="mixed", group="gP", fold=0, label=1,
             events=json.dumps([{"alert": 2.75, "event": 3.0, "hazard": "cut"}]), duration=3.5),
        dict(video_id="n", source="charades", path="n.mp4", view="third", group="gN", fold=1, label=0,
             events="[]", duration=5.0),
    ])


def test_positive_videos_get_every_frame_and_negatives_every_second():
    idx = feature_index(manifest(), pos_stride=1, neg_stride=8)
    p, n = idx[idx.video_id == "p"], idx[idx.video_id == "n"]
    # 3.5 s at 8 fps = 28 frames; 16-frame windows end at frames 16..28.
    assert list(p.end_frame) == list(range(16, 29))
    assert list(n.end_frame) == [16, 24, 32, 40]
    assert list(idx.index) == list(range(len(idx)))


def test_window_labels_follow_the_training_rule_and_keep_dropped_windows_as_nan():
    p = feature_index(manifest()).query("video_id == 'p'").set_index("end_frame")
    assert p.loc[16, "window_label"] == 0  # t_end 2.0 < alert 2.75 - 0.5 gap
    assert math.isnan(p.loc[19, "window_label"])  # 2.375: inside the gap
    assert p.loc[22, "window_label"] == 1  # 2.75 = alert
    assert p.loc[24, "window_label"] == 1  # 3.0 = event
    assert math.isnan(p.loc[25, "window_label"])  # after the accident
    n = feature_index(manifest()).query("video_id == 'n'")
    assert set(n.window_label) == {0}


def test_index_carries_what_evaluation_needs():
    idx = feature_index(manifest())
    row = idx.iloc[0]
    assert (row.video_label, row.fold, row.group, row.source, row.duration) == (1, 0, "gP", "oops", 3.5)
    assert row.t_end == row.end_frame / 8


def test_chunks_cover_each_video_once_without_crossing_videos():
    idx = feature_index(manifest())
    chunks = chunk_plan(idx, max_windows=5)
    assert [len(c[1]) for c in chunks] == [5, 5, 3, 4]
    assert sorted(np.concatenate([c[1] for c in chunks])) == list(range(len(idx)))
    assert {c[0] for c in chunks[:3]} == {"p.mp4"} and chunks[3][0] == "n.mp4"


def fake_decode(path, start, stop):
    """Frame i of every video is a 1x1 'image' whose pixel value is i, so outputs reveal the window."""
    return np.arange(start, stop, dtype=np.uint8).reshape(-1, 1, 1, 1)


def fake_encode(windows):
    w = np.asarray(windows)
    return {"first": w[:, 0, 0, 0, 0].astype(np.float32), "last": w[:, -1, 0, 0, 0].astype(np.float32)}


SPECS = {"first": ((), np.float32), "last": ((), np.float32)}


def test_extract_writes_each_window_to_its_row(tmp_path):
    idx = feature_index(manifest())
    extract(idx, fake_encode, SPECS, tmp_path, decode=fake_decode, max_windows=5)
    first, last = np.load(tmp_path / "first.npy"), np.load(tmp_path / "last.npy")
    assert list(first) == list(idx.end_frame - 16)
    assert list(last) == list(idx.end_frame - 1)
    assert np.load(tmp_path / "done.npy").all()


def test_extract_resumes_without_redoing_finished_chunks(tmp_path):
    idx = feature_index(manifest())
    extract(idx, fake_encode, SPECS, tmp_path, decode=fake_decode, max_windows=5)

    def boom(windows):
        raise AssertionError("re-encoded a finished chunk")

    extract(idx, boom, SPECS, tmp_path, decode=fake_decode, max_windows=5)


def test_extract_leaves_windows_past_the_real_end_of_a_short_video_undone(tmp_path):
    # The manifest's duration can overshoot the decodable frames by one or two.
    idx = feature_index(manifest())

    def short_decode(path, start, stop):
        return fake_decode(path, start, min(stop, 26) if path == "p.mp4" else stop)

    missing = extract(idx, fake_encode, SPECS, tmp_path, decode=short_decode, max_windows=5)
    done = np.load(tmp_path / "done.npy")
    assert sorted(idx.end_frame[~done]) == [27, 28]
    assert missing == 2


def test_extract_refuses_a_changed_index(tmp_path):
    extract(feature_index(manifest()), fake_encode, SPECS, tmp_path, decode=fake_decode)
    with pytest.raises(ValueError, match="index"):
        extract(feature_index(manifest(), neg_stride=4), fake_encode, SPECS, tmp_path, decode=fake_decode)


def test_decode_of_a_missing_file_yields_no_frames(tmp_path):
    # Lets extraction start while videos are still uploading: missing files stay undone for a later pass.
    from ks.features import decode_frames

    assert len(decode_frames(tmp_path / "not_yet.mp4", 0, 16)) == 0
