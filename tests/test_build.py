import json

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
    "Comp A0": {"decision": "keep", "duration": "6.0",
                "events": '[{"alert": 1.5, "event": 3.0, "hazard": "cut"}, {"alert": 4.0, "event": 5.5, "hazard": "fire"}]'},
    "Comp A1": {"decision": "reject", "events": "[]"},
    "Comp B3": {"decision": "benign", "events": "[]"},
    "Comp C9": {"decision": "keep", "duration": "9.0", "events": '[{"alert": 2.0, "event": 4.5, "hazard": "fire"}]'},
    "Comp D2": {"decision": "keep", "duration": "5.0", "events": '[{"alert": 1.0, "event": 2.0, "hazard": "burn_scald"}]'},
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
    assert json.loads(a0.events) == [{"alert": 1.5, "event": 3.0, "hazard": "cut"},
                                     {"alert": 4.0, "event": 5.5, "hazard": "fire"}]
    assert a0.duration == 6.0
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
    assert set(df.fold) <= set(range(5))
    assert validate(df) == []


def test_every_fold_gets_positives_and_negatives():
    # 10 one-clip compilations and 10 actors over 5 folds: each fold holds 2 of each, so every test
    # fold of the cross-validation has both classes.
    cands = [{"video_id": f"Comp {g}0", "group": f"Comp {g}"} for g in "ABCDEFGHIJ"]
    labels = {c["video_id"]: {"decision": "keep", "duration": "5.0",
                              "events": '[{"alert": 3.0, "event": 4.0, "hazard": "cut"}]'} for c in cands}
    negs = [{"video_id": f"charades:{i}", "source": "charades", "path": f"n/{i}.mp4", "view": "third",
             "group": f"charades:S{i}", "label": "0", "duration": "30.0"} for i in range(10)]
    df = build(candidates=cands, labels=labels, negatives=negs, folds=5)
    assert df.groupby(["fold", "label"]).size().to_dict() == {(f, y): 2 for f in range(5) for y in (0, 1)}


def test_unkept_oops_clips_contribute_their_footage_before_the_fail_as_negatives():
    # Same YouTube source as the positives, so "looks like a fail compilation" stops predicting danger.
    cands = CANDIDATES + [
        {"video_id": "Comp A7", "group": "Comp A", "oops_event": "6.0", "candidate": "True"},   # unlabeled
        {"video_id": "Comp E1", "group": "Comp E", "oops_event": "3.0", "candidate": "True"},   # too short
        {"video_id": "Comp F1", "group": "Comp F", "oops_event": "", "candidate": "True"},      # no marker
        {"video_id": "Comp G1", "group": "Comp G", "oops_event": "9.0", "candidate": "False"},  # not a kitchen
    ]
    cands[1] = {**cands[1], "oops_event": "5.0", "candidate": "True"}  # Comp A1, rejected
    cands[0] = {**cands[0], "oops_event": "3.0", "candidate": "True"}  # Comp A0, kept: stays a positive
    df = build(candidates=cands).set_index("video_id")
    prefail = df[(df.source == "oops") & (df.label == 0)]
    assert sorted(prefail.index) == ["Comp A1", "Comp A7"]
    assert prefail.loc["Comp A7", "duration"] == 4.5  # marker 6.0 - 1.5 s margin
    assert prefail.loc["Comp A1", "events"] == "[]" and df.loc["Comp A0", "label"] == 1
    assert df.loc["Comp A7", "fold"] == df.loc["Comp A0", "fold"]  # one compilation, one fold
    assert validate(df.reset_index()) == []


def test_adding_youtube_negatives_leaves_the_positives_folds_unchanged():
    # Folds balance positives first, so the pilot's folds survive and runs stay comparable.
    cands = [{"video_id": f"Comp {g}{i}", "group": f"Comp {g}", "oops_event": "8.0", "candidate": "True"}
             for g in "ABCDEFGHIJ" for i in range(1 + (g in "AB") * 3)]
    labels = {f"Comp {g}0": {"decision": "keep", "duration": "5.0",
                             "events": '[{"alert": 3.0, "event": 4.0, "hazard": "cut"}]'} for g in "ABCDEFGHIJ"}
    with_neg = build(candidates=cands, labels=labels).set_index("video_id")
    without = build(candidates=[{k: v for k, v in c.items() if k != "candidate"} for c in cands],
                    labels=labels).set_index("video_id")
    pos = without[without.label == 1].index
    assert (with_neg.loc[pos, "fold"] == without.loc[pos, "fold"]).all()
    assert with_neg[with_neg.label == 1].groupby("fold").size().tolist() == [2] * 5


def test_prefail_from_every_oops_clip_with_its_own_margin_and_folder():
    # The full mirror has ~2,000 clips with 3 s of calm footage before their failure, mostly not kitchens.
    cands = CANDIDATES + [
        {"video_id": "Comp G1", "group": "Comp G", "oops_event": "9.0", "candidate": "False"},  # not a kitchen
        {"video_id": "Comp H1", "group": "Comp H", "oops_event": "4.5", "candidate": "False"},  # 1.5 s left: too short
    ]
    df = build(candidates=cands, prefail="all", prefail_margin=3.0, prefail_dir="n/prefail").set_index("video_id")
    g = df.loc["Comp G1"]
    assert (g.source, g.label, g.duration, g.path) == ("oops", 0, 6.0, "n/prefail/Comp G1.mp4")
    assert "Comp H1" not in df.index
    assert df.loc["Comp A0", "label"] == 1  # kept clips stay positives, never prefail
    assert validate(df.reset_index()) == []


def test_kitchen_only_prefail_stays_the_default():
    cands = CANDIDATES + [{"video_id": "Comp G1", "group": "Comp G", "oops_event": "9.0", "candidate": "False"}]
    assert "Comp G1" not in set(build(candidates=cands).video_id)
