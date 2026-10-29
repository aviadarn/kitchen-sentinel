"""Phase 2 build: labels + candidates + negatives -> one validated manifest.csv, group-split into
cross-validation folds.

    uv run python -m ks.data.build --candidates data/candidates.csv --labels data/labels.csv \
        --negatives data/norm/epic/negatives.csv data/norm/charades/negatives.csv \
        --oops-dir data/norm/oops --pasbench-meta data/pasbench_metadata.csv --out data/manifest.csv
"""

import argparse
import csv
import json
from pathlib import Path, PurePosixPath

import pandas as pd

from ks.data.manifest import COLUMNS, parse_events, validate
from ks.data.splits import assign_splits

FOLDS = 5
PREFAIL_MARGIN = 1.5  # s before the Oops failure marker; 95% of labeled warnings are shorter
MIN_PREFAIL = 2.0  # one full window


def build_manifest(candidates, labels, negatives, oops_dir, exclude, seed, folds=FOLDS):
    """`labels` maps video_id -> latest main-pass label row (see ks.label.server.latest_labels)."""
    rows = []
    for c in candidates:
        label = labels.get(c["video_id"])
        if c["video_id"] in exclude:
            continue
        if label is None or label["decision"] != "keep":
            # A kitchen clip we didn't keep is still normal activity until its failure starts: negatives in
            # the positives' own YouTube style, so "looks like a fail compilation" stops predicting danger.
            marker = float(c.get("oops_event") or "nan")
            if str(c.get("candidate")) == "True" and marker - PREFAIL_MARGIN >= MIN_PREFAIL:
                rows.append({"video_id": c["video_id"], "source": "oops", "path": f"{oops_dir}/{c['video_id']}.mp4",
                             "view": "mixed", "group": c["group"], "label": 0, "events": "[]",
                             "duration": marker - PREFAIL_MARGIN})
            continue
        rows.append({
            "video_id": c["video_id"], "source": "oops", "path": f"{oops_dir}/{c['video_id']}.mp4", "view": "mixed",
            "group": c["group"], "label": 1, "events": json.dumps(parse_events(label["events"])),
            "duration": float(label["duration"]),
        })
    for n in negatives:
        rows.append({**{k: n[k] for k in ("video_id", "source", "path", "view", "group")}, "label": 0,
                     "events": "[]", "duration": float(n["duration"])})

    df = pd.DataFrame(rows)
    # Too few positives for a fixed test split, so every clip is tested once by k-fold cross-validation.
    # Fold each source separately so every fold gets its share of positives, ego and third-person
    # negatives; groups (compilation / participant / actor) never straddle folds.
    # Groups holding positives are placed by their positive count first, so adding negatives to them
    # (YouTube footage before the failure) leaves every positive in the same fold.
    fold_of, fractions = {}, dict.fromkeys(range(folds), 1 / folds)
    for _, part in df.groupby("source"):
        positives = part[part.label == 1].groupby("group").size().to_dict()
        fold_of.update(assign_splits(positives, fractions, seed) if positives else {})
        rest = part[~part.group.isin(positives)].groupby("group").size().to_dict()
        fold_of.update(assign_splits(rest, fractions, seed) if rest else {})
    df["fold"] = df.group.map(fold_of)
    return df[COLUMNS]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    from ks.label.server import latest_labels

    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--negatives", nargs="+", required=True)
    ap.add_argument("--oops-dir", required=True)
    ap.add_argument("--pasbench-meta", help="PaSBench-Video test/metadata.csv; its Oops clips are held out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    exclude = set()
    if a.pasbench_meta:
        with open(a.pasbench_meta, encoding="utf-8-sig") as f:
            exclude = {PurePosixPath(r["file_name"]).stem for r in csv.DictReader(f) if r["dataset"] == "oops"}
    df = build_manifest(read_csv(a.candidates), latest_labels(read_csv(a.labels)),
                        [n for p in a.negatives for n in read_csv(p)], a.oops_dir, exclude, a.seed)
    errors = validate(df)
    if errors:
        raise SystemExit("manifest invalid:\n" + "\n".join(errors[:20]))
    df.to_csv(a.out, index=False)
    Path(a.out).with_suffix(".summary.txt").write_text(
        pd.crosstab([df.source, df.label], df.fold).to_string() + "\n\n"
        + hazard_counts(df).to_string() + "\n")
    print(pd.crosstab([df.source, df.label], df.fold))
    print(hazard_counts(df))


def hazard_counts(df):
    """Events per hazard type, plus how many clips hold more than one event."""
    events = [parse_events(e) for e in df[df.label == 1].events]
    counts = pd.Series([e["hazard"] for evs in events for e in evs]).value_counts()
    counts["(clips with 2+ events)"] = sum(len(evs) > 1 for evs in events)
    return counts


if __name__ == "__main__":
    main()
