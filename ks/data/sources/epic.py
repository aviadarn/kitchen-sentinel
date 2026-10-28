"""EPIC-KITCHENS-100 (egocentric cooking, CC BY-NC 4.0): normal-cooking negatives.

Only the EPIC-100 extension videos (3-digit ids such as P01_102) are used: they share one URL
layout, and 21 participants have them. Videos are 1080p at 50-60 fps (5-6 GB each), so they are
transcoded straight from the URL instead of being stored.
"""

BASE_URL = "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m"


def video_url(video_id):
    participant = video_id.split("_")[0]
    return f"{BASE_URL}/{participant}/videos/{video_id}.MP4"


def select_videos(info, per_participant=1, target_seconds=480):
    """Rows of EPIC_100_video_info.csv -> the `per_participant` videos nearest `target_seconds` each."""
    by_participant = {}
    for r in info:
        participant, suffix = r["video_id"].split("_")
        if len(suffix) == 3:
            by_participant.setdefault(participant, []).append(r)

    picked = []
    for participant in sorted(by_participant):
        nearest = sorted(by_participant[participant],
                         key=lambda r: (abs(float(r["duration"]) - target_seconds), r["video_id"]))
        for r in sorted(nearest[:per_participant], key=lambda r: r["video_id"]):
            picked.append({"video_id": r["video_id"], "group": f"epic:{participant}",
                           "duration": float(r["duration"]), "url": video_url(r["video_id"])})
    return picked
