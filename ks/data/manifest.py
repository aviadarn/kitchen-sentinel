"""The unified video manifest (one row per video, all sources) and its validation."""

import pandas as pd

COLUMNS = [
    "video_id", "source", "path", "view", "group", "split", "label",
    "time_of_event", "time_of_alert", "hazard_type", "duration",
]
HAZARD_TYPES = {"cut", "burn_scald", "fire", "falling_breaking", "slip_fall", "appliance", "impact", "other"}


def validate(df):
    """Return a list of human-readable problems; empty means the manifest is usable."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        return [f"missing column: {c}" for c in missing]

    errors = [f"duplicate video_id: {v}" for v in df.loc[df.video_id.duplicated(), "video_id"].unique()]

    for r in df.itertuples(index=False):
        has_alert, has_event = not pd.isna(r.time_of_alert), not pd.isna(r.time_of_event)
        if r.label == 1:
            if not (has_alert and has_event):
                errors.append(f"{r.video_id}: positive needs time_of_alert and time_of_event")
            elif not 0 <= r.time_of_alert < r.time_of_event <= r.duration:
                errors.append(f"{r.video_id}: need 0 <= alert ({r.time_of_alert}) < event ({r.time_of_event}) "
                              f"<= duration ({r.duration})")
            if r.hazard_type not in HAZARD_TYPES:
                errors.append(f"{r.video_id}: hazard_type {r.hazard_type!r} not in {sorted(HAZARD_TYPES)}")
        elif r.label == 0:
            if has_alert or has_event:
                errors.append(f"{r.video_id}: negative must not have alert/event times")
        else:
            errors.append(f"{r.video_id}: label must be 0 or 1, got {r.label!r}")

    splits_per_group = df.groupby("group")["split"].nunique()
    for g in splits_per_group[splits_per_group > 1].index:
        errors.append(f"group {g!r} appears in several splits (leak)")
    return errors
