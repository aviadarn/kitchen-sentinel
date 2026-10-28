import pytest

from ks.label.server import latest_labels, parse_range, queue_order, recheck_sample, validate_label


@pytest.mark.parametrize(
    "header, want",
    [
        ("bytes=0-", (0, 999)),
        ("bytes=100-199", (100, 199)),
        ("bytes=-300", (700, 999)),        # suffix range: last 300 bytes
        ("bytes=900-5000", (900, 999)),    # end past EOF is clamped
        ("bytes=1000-", None),             # starts past EOF: unsatisfiable
        ("bytes=200-100", None),
        ("items=0-10", None),
        ("bytes=0-1,5-6", None),           # multi-range not supported -> caller sends the whole file
    ],
)
def test_parse_range(header, want):
    assert parse_range(header, size=1000) == want


def keep(**over):
    base = {"video_id": "v1", "decision": "keep", "hazard_type": "cut", "time_of_alert": 2.0, "time_of_event": 3.5,
            "duration": 8.0}
    base.update(over)
    return base


def test_valid_keep_label_passes():
    assert validate_label(keep()) == []


@pytest.mark.parametrize(
    "over",
    [
        {"hazard_type": None},
        {"hazard_type": "explosion"},
        {"time_of_alert": None},
        {"time_of_event": None},
        {"time_of_alert": 4.0},          # alert after event
        {"time_of_event": 9.0},          # event past the end of the clip
        {"decision": "maybe"},
    ],
)
def test_invalid_labels_are_rejected(over):
    assert len(validate_label(keep(**over))) == 1


def test_reject_and_benign_need_no_times():
    assert validate_label({"video_id": "v1", "decision": "reject", "duration": 8.0}) == []
    assert validate_label({"video_id": "v1", "decision": "benign", "duration": 8.0}) == []


def test_latest_label_per_clip_wins():
    rows = [
        {"video_id": "a", "decision": "keep", "pass": "main"},
        {"video_id": "b", "decision": "reject", "pass": "main"},
        {"video_id": "a", "decision": "reject", "pass": "main"},  # relabel
        {"video_id": "b", "decision": "keep", "pass": "recheck"},  # recheck rows never override the main label
    ]
    got = latest_labels(rows)
    assert {k: v["decision"] for k, v in got.items()} == {"a": "reject", "b": "reject"}


def test_queue_puts_unlabeled_first_then_title_matches_then_score():
    cands = [
        {"video_id": "low", "kitchen_score": "0.35", "title_match": "False"},
        {"video_id": "done", "kitchen_score": "0.99", "title_match": "False"},
        {"video_id": "titled", "kitchen_score": "0.10", "title_match": "True"},
        {"video_id": "high", "kitchen_score": "0.95", "title_match": "False"},
        {"video_id": "noscore", "kitchen_score": "", "title_match": "False"},
    ]
    labels = {"done": {"decision": "keep"}}
    assert [c["video_id"] for c in queue_order(cands, labels)] == ["titled", "high", "low", "noscore", "done"]


def test_recheck_sample_draws_from_kept_clips_only_and_is_stable():
    labels = {f"k{i}": {"decision": "keep"} for i in range(10)} | {f"r{i}": {"decision": "reject"} for i in range(10)}
    a = recheck_sample(labels, n=4, seed=0)
    assert len(a) == 4 and all(v.startswith("k") for v in a)
    assert a == recheck_sample(labels, n=4, seed=0)


def test_recheck_sample_caps_at_available():
    assert len(recheck_sample({"k": {"decision": "keep"}}, n=50, seed=0)) == 1
