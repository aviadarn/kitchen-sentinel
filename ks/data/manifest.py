"""The unified video manifest (one row per video, all sources) and its validation.

A positive video carries `events`: a JSON list of {"alert", "event", "hazard"}, one per accident in the
clip (some fails contain several). A negative video has "[]".
"""

import json

import pandas as pd

COLUMNS = ["video_id", "source", "path", "view", "group", "fold", "label", "events", "duration"]
HAZARD_TYPES = {"cut", "burn_scald", "fire", "falling_breaking", "slip_fall", "appliance", "impact", "other"}


def parse_events(events):
    """JSON string or list -> list of event dicts."""
    return json.loads(events) if isinstance(events, str) else list(events or [])


def check_events(events, duration):
    """Problems with a kept clip's events (empty list = fine). Shared by the labeler and the manifest."""
    if not events:
        return ["mark at least one event (A, E, 1-8)"]
    errors = []
    for i, e in enumerate(events, 1):
        alert, event = e.get("alert"), e.get("event")
        if e.get("hazard") not in HAZARD_TYPES:
            errors.append(f"event {i}: pick a hazard type (1-8)")
        if alert is None or event is None:
            errors.append(f"event {i}: mark both alert (A) and event (E)")
        elif not 0 <= alert < event <= duration:
            errors.append(f"event {i}: need 0 <= alert ({alert:.2f}) < event ({event:.2f}) <= clip length "
                          f"({duration:.2f})")
    if errors:
        return errors
    ordered = sorted(events, key=lambda e: e["alert"])
    for a, b in zip(ordered, ordered[1:]):
        if a["event"] > b["alert"]:
            errors.append(f"events overlap: one warning starts ({b['alert']:.2f}) before the previous accident "
                          f"({a['event']:.2f})")
    return errors


def validate(df):
    """Return a list of human-readable problems; empty means the manifest is usable."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        return [f"missing column: {c}" for c in missing]

    errors = [f"duplicate video_id: {v}" for v in df.loc[df.video_id.duplicated(), "video_id"].unique()]

    for r in df.itertuples(index=False):
        events = parse_events(r.events)
        if r.label == 1:
            errors += [f"{r.video_id}: {e}" for e in check_events(events, r.duration)]
        elif r.label == 0:
            if events:
                errors.append(f"{r.video_id}: negative must not have events")
        else:
            errors.append(f"{r.video_id}: label must be 0 or 1, got {r.label!r}")

    folds_per_group = df.groupby("group")["fold"].nunique()
    for g in folds_per_group[folds_per_group > 1].index:
        errors.append(f"group {g!r} appears in several folds (leak)")
    return errors
