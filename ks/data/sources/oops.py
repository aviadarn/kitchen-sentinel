"""Oops! dataset (Epstein et al., CVPR 2020): index of clips and their failure-moment labels.

The annotation archive holds train.txt / val.txt (every clip, 29,940 in all; 10,961 labeled) plus
two label files: transition_times.json (3 annotators per clip) and heldout_transition_times.json (a
4th annotator for 4,791 of those clips). Each annotator's `t` is the moment the action turns into a
failure, or -1 if they found none.
"""

import json
import statistics
from pathlib import Path

from ks.data.splits import oops_group


def event_time(t):
    """Median of the valid annotator times, or None unless a majority of annotators found a failure."""
    valid = [x for x in t if x >= 0]
    if len(valid) * 2 <= len(t):
        return None
    return statistics.median(valid)


def load_index(ann_dir):
    """One row per clip listed in train.txt / val.txt, with its Oops failure time when there is one."""
    ann_dir = Path(ann_dir)
    votes = {}  # the held-out file adds a 4th annotator to clips already in transition_times.json
    for name in ("transition_times.json", "heldout_transition_times.json"):
        for video_id, entry in json.loads((ann_dir / name).read_text()).items():
            votes.setdefault(video_id, []).extend(entry["t"])

    rows = []
    for oops_split in ("train", "val"):
        for video_id in (ann_dir / f"{oops_split}.txt").read_text().splitlines():
            if not video_id:
                continue
            t = votes.get(video_id, [])
            valid = [x for x in t if x >= 0]
            rows.append({
                "video_id": video_id,
                "group": oops_group(video_id),
                "oops_split": oops_split,
                "oops_event": event_time(t) if t else None,
                "event_spread": max(valid) - min(valid) if valid else None,
                "n_annotators": len(t),
            })
    return rows
