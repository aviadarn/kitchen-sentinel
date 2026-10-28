"""Phase 2 build: labels + candidates + negatives -> one validated, group-split manifest.csv.

    uv run python -m ks.data.build --candidates data/candidates.csv --labels data/labels.csv \
        --negatives data/norm/epic/negatives.csv data/norm/charades/negatives.csv \
        --oops-dir data/norm/oops --pasbench-meta data/pasbench_metadata.csv --out data/manifest.csv
"""

import argparse
import csv
from pathlib import Path, PurePosixPath

import pandas as pd

from ks.data.manifest import COLUMNS, validate
from ks.data.splits import assign_splits

FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def build_manifest(candidates, labels, negatives, oops_dir, exclude, seed):
    """`labels` maps video_id -> latest main-pass label row (see ks.label.server.latest_labels)."""
    rows = []
    for c in candidates:
        label = labels.get(c["video_id"])
        if label is None or label["decision"] != "keep" or c["video_id"] in exclude:
            continue
        rows.append({
            "video_id": c["video_id"], "source": "oops", "path": f"{oops_dir}/{c['video_id']}.mp4", "view": "mixed",
            "group": c["group"], "label": 1, "time_of_event": float(label["time_of_event"]),
            "time_of_alert": float(label["time_of_alert"]), "hazard_type": label["hazard_type"],
            "duration": float(label["duration"]),
        })
    for n in negatives:
        rows.append({**{k: n[k] for k in ("video_id", "source", "path", "view", "group")}, "label": 0,
                     "time_of_event": None, "time_of_alert": None, "hazard_type": None,
                     "duration": float(n["duration"])})

    df = pd.DataFrame(rows)
    # Split each source separately so every split gets its share of positives, ego and third-person
    # negatives; groups (compilation / participant / actor) never straddle splits.
    split_of = {}
    for _, part in df.groupby("source"):
        split_of.update(assign_splits(part.groupby("group").size().to_dict(), FRACTIONS, seed))
    df["split"] = df.group.map(split_of)
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
        pd.crosstab([df.source, df.label], df.split).to_string() + "\n\n"
        + df[df.label == 1].hazard_type.value_counts().to_string() + "\n")
    print(pd.crosstab([df.source, df.label], df.split))
    print(df[df.label == 1].hazard_type.value_counts())


if __name__ == "__main__":
    main()
