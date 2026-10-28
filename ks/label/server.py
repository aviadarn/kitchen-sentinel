"""Local labeling tool: one candidate clip at a time in the browser, keyboard-driven.

    uv run python -m ks.label.server --candidates data/candidates.csv --videos data/label_clips \
        --labels data/labels.csv [--recheck 50]

Labels are appended to labels.csv and never edited in place; the latest main-pass row per clip wins.
A kept clip carries one or more events (some fails contain several accidents), stored as a JSON list
of {"alert", "event", "hazard"} in the `events` column.
--recheck re-serves a fixed random sample of kept clips with the earlier answers hidden, to measure
how consistently one labeler places the alert time (BADAS used 10 raters; we have one).
"""

import argparse
import csv
import json
import random
import re
import shutil
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from ks.data.manifest import check_events

LABEL_FIELDS = ["video_id", "decision", "events", "duration", "pass", "labeler", "labeled_at"]
DECISIONS = {"keep", "reject", "benign"}
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")


def parse_range(header, size):
    """(start, end) inclusive for a single-range `Range` header, or None to send the whole file."""
    m = RANGE_RE.match(header.strip())
    if not m or (not m.group(1) and not m.group(2)):
        return None
    if not m.group(1):
        return max(0, size - int(m.group(2))), size - 1
    start = int(m.group(1))
    end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
    if start >= size or start > end:
        return None
    return start, end


def validate_label(label):
    if label.get("decision") not in DECISIONS:
        return [f"decision must be one of {sorted(DECISIONS)}"]
    if label["decision"] != "keep":
        return []
    return check_events(label.get("events") or [], label["duration"])


def migrate_labels_csv(path):
    """Rewrite a pre-multi-event labels.csv (hazard_type / time_of_alert / time_of_event columns) to the
    `events` format, keeping the original as labels.csv.bak. No-op on an already migrated file."""
    path = Path(path)
    with path.open() as f:
        reader = csv.DictReader(f)
        if "events" in reader.fieldnames:
            return
        old = list(reader)
    shutil.copy(path, path.with_name(path.name + ".bak"))
    rows = []
    for r in old:
        events = []
        if r["decision"] == "keep" and r["time_of_alert"] and r["time_of_event"]:
            events = [{"alert": float(r["time_of_alert"]), "event": float(r["time_of_event"]), "hazard": r["hazard_type"]}]
        rows.append({**{k: r[k] for k in LABEL_FIELDS if k in r}, "events": json.dumps(events)})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def latest_labels(rows):
    latest = {}
    for r in rows:
        if r.get("pass", "main") == "main":
            latest[r["video_id"]] = r
    return latest


def queue_order(candidates, labels):
    """Unlabeled first; within that, title matches, then highest kitchen score."""
    def key(c):
        score = float(c["kitchen_score"]) if c["kitchen_score"] else -1.0
        return (c["video_id"] in labels, c["title_match"] != "True", -score)
    return sorted(candidates, key=key)


def recheck_sample(labels, n, seed):
    kept = sorted(v for v, r in labels.items() if r["decision"] == "keep")
    return random.Random(seed).sample(kept, min(n, len(kept)))


class LabelStore:
    def __init__(self, candidates_csv, videos_dir, labels_csv, labeler, recheck):
        with open(candidates_csv) as f:
            self.candidates = [r for r in csv.DictReader(f) if r.get("candidate", "True") == "True"]
        self.videos = Path(videos_dir)
        self.labels_csv = Path(labels_csv)
        self.labeler = labeler
        self.recheck = recheck
        if self.labels_csv.exists():
            migrate_labels_csv(self.labels_csv)
        else:
            with self.labels_csv.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=LABEL_FIELDS).writeheader()

    def rows(self):
        with self.labels_csv.open() as f:
            return list(csv.DictReader(f))

    def queue(self):
        rows = self.rows()
        labels = latest_labels(rows)
        if self.recheck:
            sample = recheck_sample(labels, self.recheck, seed=0)
            done = {r["video_id"] for r in rows if r["pass"] == "recheck"}
            by_id = {c["video_id"]: c for c in self.candidates}
            clips = [{**by_id[v], "label": None, "done": v in done} for v in sample]
            return {"mode": "recheck", "clips": clips, "done": len(done & set(sample)), "total": len(sample)}
        clips = [{**c, "label": labels.get(c["video_id"]), "done": c["video_id"] in labels}
                 for c in queue_order(self.candidates, labels)]
        return {"mode": "main", "clips": clips, "done": len(labels), "total": len(self.candidates)}

    def save(self, label):
        errors = validate_label(label)
        if errors:
            return errors
        row = {k: label.get(k) for k in LABEL_FIELDS}
        events = sorted(label.get("events") or [], key=lambda e: e["alert"]) if label["decision"] == "keep" else []
        row["events"] = json.dumps([{"alert": e["alert"], "event": e["event"], "hazard": e["hazard"]} for e in events])
        row.update({"pass": "recheck" if self.recheck else "main", "labeler": self.labeler,
                    "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        with self.labels_csv.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=LABEL_FIELDS).writerow(row)
        return []

    def video_path(self, video_id):
        known = {c["video_id"] for c in self.candidates}
        return self.videos / f"{video_id}.mp4" if video_id in known else None  # no arbitrary file reads


def make_handler(store):
    page = (Path(__file__).parent / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_body(self, status, body, ctype):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                return self.send_body(200, page, "text/html; charset=utf-8")
            if self.path == "/api/queue":
                return self.send_body(200, json.dumps(store.queue()).encode(), "application/json")
            if self.path.startswith("/video/"):
                return self.send_video(unquote(self.path[len("/video/"):]))
            self.send_body(404, b"not found", "text/plain")

        def send_video(self, video_id):
            path = store.video_path(video_id)
            if path is None or not path.exists():
                return self.send_body(404, b"no such clip", "text/plain")
            size = path.stat().st_size
            rng = parse_range(self.headers.get("Range", ""), size)
            start, end = rng if rng else (0, size - 1)
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                self.wfile.write(f.read(end - start + 1))

        def do_POST(self):
            if self.path != "/api/label":
                return self.send_body(404, b"not found", "text/plain")
            label = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            errors = store.save(label)
            self.send_body(200 if not errors else 400, json.dumps({"errors": errors}).encode(), "application/json")

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--videos", required=True, help="folder of <video_id>.mp4 originals")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--labeler", default="aviad")
    ap.add_argument("--recheck", type=int, default=0, help="blind re-label N random kept clips")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    store = LabelStore(a.candidates, a.videos, a.labels, a.labeler, a.recheck)
    print(f"labeling {len(store.candidates)} candidates -> http://localhost:{a.port}")
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(store)).serve_forever()


if __name__ == "__main__":
    main()
