"""Charades (third-person home video by crowd actors, non-commercial research): kitchen-scene
negatives. Closest in look to Oops! handheld footage among the normal-activity sources."""


def kitchen_rows(rows):
    """Rows of Charades_v1_{train,test}.csv -> kitchen-scene videos, grouped by actor."""
    return [
        {"video_id": f"charades:{r['id']}", "charades_id": r["id"], "group": f"charades:{r['subject']}",
         "duration": float(r["length"])}
        for r in rows
        if r["scene"] == "Kitchen"
    ]
