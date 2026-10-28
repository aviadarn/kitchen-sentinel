import json

import pandas as pd

from ks.data.manifest import validate


def ev(alert=6.0, event=7.5, hazard="burn_scald"):
    return {"alert": alert, "event": event, "hazard": hazard}


def row(events=None, **over):
    base = dict(video_id="v1", source="oops", path="data/v1.mp4", view="mixed", group="comp A", split="train",
                label=1, events=json.dumps([ev()] if events is None else events), duration=10.0)
    base.update(over)
    return base


def negative(**over):
    base = dict(video_id="n1", source="epic", group="P01", label=0, events=[])
    base.update(over)
    return row(**base)


def test_valid_manifest_has_no_errors():
    assert validate(pd.DataFrame([row(), negative()])) == []


def test_two_events_are_valid():
    assert validate(pd.DataFrame([row([ev(1.0, 2.0, "cut"), ev(5.0, 7.0, "fire")])])) == []


def test_alert_after_event_is_an_error():
    errs = validate(pd.DataFrame([row([ev(alert=8.0)])]))
    assert len(errs) == 1 and "v1" in errs[0]


def test_event_beyond_duration_is_an_error():
    errs = validate(pd.DataFrame([row([ev(event=12.0)])]))
    assert len(errs) == 1 and "v1" in errs[0]


def test_overlapping_events_are_an_error():
    assert len(validate(pd.DataFrame([row([ev(1.0, 5.0), ev(4.0, 7.0)])]))) == 1


def test_positive_without_events_is_an_error():
    assert len(validate(pd.DataFrame([row([])]))) == 1


def test_event_without_hazard_type_is_an_error():
    assert len(validate(pd.DataFrame([row([ev(hazard=None)])]))) == 1


def test_unknown_hazard_type_is_an_error():
    assert len(validate(pd.DataFrame([row([ev(hazard="explosion")])]))) == 1


def test_negative_with_events_is_an_error():
    assert len(validate(pd.DataFrame([negative(events=[ev(2.0, 3.0)])]))) == 1


def test_group_in_two_splits_is_a_leak():
    df = pd.DataFrame([row(), row(video_id="v2", split="test")])
    errs = validate(df)
    assert len(errs) == 1 and "comp A" in errs[0]


def test_duplicate_video_id_is_an_error():
    assert len(validate(pd.DataFrame([row(), row()]))) == 1


def test_missing_column_is_reported_without_crashing():
    errs = validate(pd.DataFrame([row()]).drop(columns=["split"]))
    assert errs == ["missing column: split"]
