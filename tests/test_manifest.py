import pandas as pd

from ks.data.manifest import validate


def row(**over):
    base = dict(
        video_id="v1", source="oops", path="data/v1.mp4", view="third", group="comp A", split="train",
        label=1, time_of_event=7.5, time_of_alert=6.0, hazard_type="burn_scald", duration=10.0,
    )
    base.update(over)
    return base


def negative(**over):
    base = dict(video_id="n1", source="epic", group="P01", label=0, time_of_event=None, time_of_alert=None,
                hazard_type=None)
    base.update(over)
    return row(**base)


def test_valid_manifest_has_no_errors():
    assert validate(pd.DataFrame([row(), negative()])) == []


def test_alert_after_event_is_an_error():
    errs = validate(pd.DataFrame([row(time_of_alert=8.0)]))
    assert len(errs) == 1 and "v1" in errs[0]


def test_event_beyond_duration_is_an_error():
    errs = validate(pd.DataFrame([row(time_of_event=12.0)]))
    assert len(errs) == 1 and "v1" in errs[0]


def test_positive_without_times_is_an_error():
    assert len(validate(pd.DataFrame([row(time_of_alert=None)]))) == 1


def test_positive_without_hazard_type_is_an_error():
    assert len(validate(pd.DataFrame([row(hazard_type=None)]))) == 1


def test_unknown_hazard_type_is_an_error():
    assert len(validate(pd.DataFrame([row(hazard_type="explosion")]))) == 1


def test_negative_with_times_is_an_error():
    assert len(validate(pd.DataFrame([negative(time_of_event=3.0, time_of_alert=2.0)]))) == 1


def test_group_in_two_splits_is_a_leak():
    df = pd.DataFrame([row(), row(video_id="v2", split="test")])
    errs = validate(df)
    assert len(errs) == 1 and "comp A" in errs[0]


def test_duplicate_video_id_is_an_error():
    assert len(validate(pd.DataFrame([row(), row()]))) == 1


def test_missing_column_is_reported_without_crashing():
    errs = validate(pd.DataFrame([row()]).drop(columns=["split"]))
    assert errs == ["missing column: split"]
