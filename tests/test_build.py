from ks.data.build import build_manifest
from ks.data.manifest import validate

CANDIDATES = [
    {"video_id": "Comp A0", "group": "Comp A"},
    {"video_id": "Comp A1", "group": "Comp A"},
    {"video_id": "Comp B3", "group": "Comp B"},
    {"video_id": "Comp C9", "group": "Comp C"},
    {"video_id": "Comp D2", "group": "Comp D"},
]
LABELS = {
    "Comp A0": {"decision": "keep", "hazard_type": "cut", "time_of_alert": "1.5", "time_of_event": "3.0", "duration": "6.0"},
    "Comp A1": {"decision": "reject"},
    "Comp B3": {"decision": "benign"},
    "Comp C9": {"decision": "keep", "hazard_type": "fire", "time_of_alert": "2.0", "time_of_event": "4.5", "duration": "9.0"},
    "Comp D2": {"decision": "keep", "hazard_type": "burn_scald", "time_of_alert": "1.0", "time_of_event": "2.0",
                "duration": "5.0"},
}
NEGATIVES = [
    {"video_id": "epic:P01_102", "source": "epic", "path": "n/epic/P01_102.mp4", "view": "ego", "group": "epic:P01",
     "label": "0", "duration": "480.0"},
    {"video_id": "charades:AAA01", "source": "charades", "path": "n/ch/AAA01.mp4", "view": "third",
     "group": "charades:S1", "label": "0", "duration": "30.0"},
]


def build(**kw):
    args = dict(candidates=CANDIDATES, labels=LABELS, negatives=NEGATIVES, oops_dir="n/oops", exclude=set(), seed=0)
    args.update(kw)
    return build_manifest(**args)


def test_only_kept_clips_become_positives():
    df = build()
    pos = df[df.label == 1]
    assert sorted(pos.video_id) == ["Comp A0", "Comp C9", "Comp D2"]
    a0 = pos.set_index("video_id").loc["Comp A0"]
    assert (a0.time_of_alert, a0.time_of_event, a0.hazard_type, a0.duration) == (1.5, 3.0, "cut", 6.0)
    assert a0.path == "n/oops/Comp A0.mp4"
    assert a0.source == "oops" and a0.view == "mixed"


def test_negatives_pass_through():
    df = build()
    neg = df[df.label == 0].set_index("video_id")
    assert sorted(neg.index) == ["charades:AAA01", "epic:P01_102"]
    assert neg.loc["epic:P01_102", "duration"] == 480.0


def test_excluded_clips_are_dropped():
    # PaSBench-Video's Oops clips are our out-of-domain test set and must not be trained on.
    assert "Comp C9" not in set(build(exclude={"Comp C9"}).video_id)


def test_result_is_a_valid_leak_free_manifest():
    df = build()
    assert set(df.split) <= {"train", "val", "test"}
    assert validate(df) == []
