import json

import pytest

from ks.data.sources.oops import event_time, load_index


@pytest.mark.parametrize(
    "t, want",
    [
        ([2.0, 5.0, 9.0], 5.0),
        ([-1.0, 3.6, 3.8], 3.7),        # 2 of 3 found it: median of the valid two
        ([-1.0, -1.0, 4.0], None),      # only 1 of 3 found a failure: not a majority
        ([-1.0, -1.0, -1.0], None),
        ([3.15], 3.15),                 # held-out set: single annotator
        ([-1.0], None),
    ],
)
def test_event_time_is_majority_median(t, want):
    got = event_time(t)
    assert got == pytest.approx(want) if want is not None else got is None


def write_annotations(d):
    (d / "train.txt").write_text("Comp A (May 2017)0\nComp A (May 2017)1\nCooking Fails _ FailArmy3\n")
    (d / "val.txt").write_text("Comp B (June 2017)7\n")
    (d / "transition_times.json").write_text(json.dumps({
        "Comp A (May 2017)0": {"t": [-1.0, 3.6, 3.8], "len": 10.0, "rel_t": [], "stdev": 2.7, "rel_stdev": 0.2,
                               "n_notfound": 1},
        "Comp B (June 2017)7": {"t": [-1.0, -1.0, -1.0], "len": 5.0, "rel_t": [], "stdev": 0.0, "rel_stdev": 0.0,
                                "n_notfound": 3},
        "Cooking Fails _ FailArmy3": {"t": [2.0, -1.0, 3.0], "len": 8.0, "rel_t": [], "stdev": 0.5, "rel_stdev": 0.1,
                                      "n_notfound": 1},
    }))
    # The held-out file is a 4th annotator on clips that are already in transition_times.json.
    (d / "heldout_transition_times.json").write_text(json.dumps({
        "Cooking Fails _ FailArmy3": {"t": [2.5], "stdev": 0, "rel_stdev": 0, "n_notfound": 0},
    }))


def test_load_index_joins_lists_with_both_label_files(tmp_path):
    write_annotations(tmp_path)
    rows = {r["video_id"]: r for r in load_index(tmp_path)}
    assert set(rows) == {"Comp A (May 2017)0", "Comp A (May 2017)1", "Cooking Fails _ FailArmy3", "Comp B (June 2017)7"}

    a0 = rows["Comp A (May 2017)0"]
    assert a0["oops_split"] == "train"
    assert a0["group"] == "Comp A (May 2017)"
    assert a0["oops_event"] == pytest.approx(3.7)
    assert a0["n_annotators"] == 3
    assert a0["event_spread"] == pytest.approx(0.2)  # max - min of the valid times

    # 3 annotators [2.0, -1, 3.0] + held-out [2.5] = 4 votes, 3 valid -> median 2.5.
    held = rows["Cooking Fails _ FailArmy3"]
    assert (held["oops_event"], held["n_annotators"]) == (2.5, 4)


def test_load_index_keeps_unlabeled_and_no_failure_videos_without_event(tmp_path):
    # Unlabeled kitchen clips are still useful: the labeler sets the event time by hand.
    write_annotations(tmp_path)
    rows = {r["video_id"]: r for r in load_index(tmp_path)}
    assert rows["Comp A (May 2017)1"]["oops_event"] is None
    assert rows["Comp A (May 2017)1"]["n_annotators"] == 0
    assert rows["Comp B (June 2017)7"]["oops_event"] is None
    assert rows["Comp B (June 2017)7"]["n_annotators"] == 3
