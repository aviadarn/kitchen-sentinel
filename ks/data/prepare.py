"""Phase 1 data jobs, run on the GPU box. Each writes normalized clips plus a CSV of manifest rows.

    python -m ks.data.prepare oops-annotations --out data/oops
    python -m ks.data.prepare oops-videos      --out data/oops            # ~48 GB streamed + extracted
    python -m ks.data.prepare oops-normalize   --candidates data/candidates.csv --out data/norm/oops
    python -m ks.data.prepare oops-prefail     --manifest data/manifest.csv --candidates data/candidates.csv \
        --out data/norm/oops_prefail
    python -m ks.data.prepare charades         --out data/norm/charades --work data/tmp
    python -m ks.data.prepare epic             --out data/norm/epic
"""

import argparse
import csv
import os
import subprocess
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ks.data.normalize import normalize_video
from ks.data.sources.charades import kitchen_rows
from ks.data.sources.epic import select_videos

OOPS_URL = "https://huggingface.co/datasets/pengxiang/oops/resolve/main/video_and_anns.tar.gz"
CHARADES_ANN_URL = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip"
CHARADES_480_URL = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"
EPIC_INFO_URL = ("https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/"
                 "EPIC_100_video_info.csv")
NEGATIVE_FIELDS = ["video_id", "source", "path", "view", "group", "label", "duration"]


def sh(cmd):
    subprocess.run(["bash", "-o", "pipefail", "-c", cmd], check=True)


def oops_annotations(out):
    # The outer tar's first member is the 1.1 MB annotations archive; 3 MB covers it.
    out.mkdir(parents=True, exist_ok=True)
    sh(f"curl -sL -r 0-3000000 '{OOPS_URL}' | tar -xzOf - oops_dataset/annotations.tar.gz | tar -xzf - -C '{out}'"
       " || true")  # tar complains about the truncated outer archive after extracting what we need
    if not (out / "annotations" / "transition_times.json").exists():
        raise RuntimeError("Oops annotations not extracted")


def oops_videos(out):
    # Outer tar -> inner video.tar.gz -> clips, streamed so the 48 GB archive never sits on disk twice.
    out.mkdir(parents=True, exist_ok=True)
    sh(f"curl -sL '{OOPS_URL}' | tar -xzOf - oops_dataset/video.tar.gz | tar -xzf - -C '{out}'")


def normalize_all(jobs, workers):
    """jobs: (src, dst) pairs. Returns the dsts that failed, so one bad clip doesn't stop the run."""
    def run(job):
        src, dst = job
        if dst.exists():
            return None
        # Write to a temp name and rename on success: a killed run must not leave a half-written
        # file that the next run would skip as already done.
        part = dst.with_name(dst.stem + ".part.mp4")
        try:
            normalize_video(src, part)
            part.rename(dst)
            return None
        except subprocess.CalledProcessError:
            part.unlink(missing_ok=True)
            return dst

    with ThreadPoolExecutor(max_workers=workers) as pool:
        failed = [d for d in pool.map(run, jobs) if d is not None]
    print(f"normalized {len(jobs) - len(failed)}/{len(jobs)}; failed {len(failed)}")
    return failed


def write_rows(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def oops_normalize(candidates, out, workers):
    out.mkdir(parents=True, exist_ok=True)
    with open(candidates) as f:
        rows = [r for r in csv.DictReader(f) if r["candidate"] == "True"]
    normalize_all([(Path(r["path"]), out / f"{r['video_id']}.mp4") for r in rows], workers)


def prefail_jobs(manifest_rows, candidate_rows, out):
    """(raw mirror clip, normalized destination) for the manifest's prefail negatives that live in `out`."""
    raw = {r["video_id"]: Path(r["path"]) for r in candidate_rows}
    out = Path(out)
    return [(raw[r["video_id"]], out / f"{r['video_id']}.mp4") for r in manifest_rows
            if r["source"] == "oops" and str(r["label"]) == "0" and Path(r["path"]).parent == out]


def oops_prefail(manifest, candidates, out, workers):
    out.mkdir(parents=True, exist_ok=True)
    with open(manifest) as f, open(candidates) as g:
        jobs = prefail_jobs(list(csv.DictReader(f)), list(csv.DictReader(g)), out)
    failed = normalize_all(jobs, workers)
    if failed:
        print("failed:", *failed, sep="\n  ")


def charades(out, work, workers):
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    ann_zip = work / "Charades.zip"
    urllib.request.urlretrieve(CHARADES_ANN_URL, ann_zip)
    with zipfile.ZipFile(ann_zip) as z:
        rows = []
        for split in ("train", "test"):
            with z.open(f"Charades/Charades_v1_{split}.csv") as f:
                rows += kitchen_rows(csv.DictReader(line.decode() for line in f))

    video_zip = work / "Charades_v1_480.zip"
    if not video_zip.exists():
        # 16 GB from S3: a reset connection once cost a whole run, so resume (-C -) and retry
        sh(f"curl -sL --retry 10 --retry-all-errors -C - -o '{video_zip}.part' '{CHARADES_480_URL}' "
           f"&& mv '{video_zip}.part' '{video_zip}'")
    raw = work / "charades_raw"
    with zipfile.ZipFile(video_zip) as z:
        for r in rows:
            z.extract(f"Charades_v1_480/{r['charades_id']}.mp4", raw)
    failed = set(normalize_all(
        [(raw / "Charades_v1_480" / f"{r['charades_id']}.mp4", out / f"{r['video_id']}.mp4") for r in rows],
        workers))
    kept = [{**r, "source": "charades", "path": str(out / f"{r['video_id']}.mp4"), "view": "third", "label": 0}
            for r in rows if out / f"{r['video_id']}.mp4" not in failed]
    write_rows(out / "negatives.csv", kept, NEGATIVE_FIELDS)


def epic(out, workers, per_participant):
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(EPIC_INFO_URL) as f:
        info = list(csv.DictReader(line.decode() for line in f))
    rows = select_videos(info, per_participant=per_participant)
    # ffmpeg reads the 5-6 GB originals straight from the URL; only the 8 fps 256 px output is kept.
    failed = set(normalize_all([(r["url"], out / f"{r['video_id']}.mp4") for r in rows], workers))
    kept = [{**r, "video_id": f"epic:{r['video_id']}", "source": "epic", "path": str(out / f"{r['video_id']}.mp4"),
             "view": "ego", "label": 0} for r in rows if out / f"{r['video_id']}.mp4" not in failed]
    write_rows(out / "negatives.csv", kept, NEGATIVE_FIELDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["oops-annotations", "oops-videos", "oops-normalize", "oops-prefail", "charades",
                                    "epic"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--work", type=Path, default=Path("data/tmp"))
    ap.add_argument("--candidates", type=Path)
    ap.add_argument("--manifest", type=Path, help="oops-prefail: normalize this manifest's prefail clips")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 4))
    ap.add_argument("--per-participant", type=int, default=1)
    a = ap.parse_args()
    if a.job == "oops-annotations":
        oops_annotations(a.out)
    elif a.job == "oops-videos":
        oops_videos(a.out)
    elif a.job == "oops-normalize":
        oops_normalize(a.candidates, a.out, a.workers)
    elif a.job == "oops-prefail":
        oops_prefail(a.manifest, a.candidates, a.out, a.workers)
    elif a.job == "charades":
        charades(a.out, a.work, a.workers)
    else:
        epic(a.out, a.workers, a.per_participant)


if __name__ == "__main__":
    main()
