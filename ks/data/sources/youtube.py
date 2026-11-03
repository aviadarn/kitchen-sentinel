"""Kitchen accidents from recent YouTube fail compilations and single caught-on-camera videos.

Downloaded with yt-dlp into data/youtube/raw (for private research; never published), cut at scene
changes, de-duplicated against the Oops clips and against each other (one viral clip turns up in many
compilations), and ranked by how dangerous SigLIP thinks they look, for labeling in ks.label.server.

    uv run --extra model python -m ks.data.sources.youtube --raw data/youtube/raw \
        --videos data/youtube/videos.csv --oops-clips data/label_clips --out data/youtube
"""

import argparse
import csv
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SCENE_RE = re.compile(r"pts_time:([0-9.]+)")

# Each caption is scored on its own (SigLIP sigmoid), so a clip's danger score is its best match.
HAZARD_PROMPTS = [
    "flames rising from a pan on a stove", "a grease fire in a kitchen", "an oven on fire", "smoke pouring from a pot",
    "boiling water spilling from a pot", "a pot falling off a stove", "a glass falling and shattering on the floor",
    "a plate breaking on a kitchen floor", "something falling off a kitchen counter", "a person slipping on a wet floor",
    "a person falling down in a kitchen", "a knife slipping while cutting", "food exploding in a microwave",
    "a blender lid flying off", "hot oil splashing from a pan",
]
FRACTIONS = (0.3, 0.6, 0.85)  # fails tend to happen late in a clip
DUPLICATE = 0.93  # cosine between mean SigLIP embeddings; re-encoded copies of one clip score above this


def parse_scene_times(log):
    """Cut times from ffmpeg's select=gt(scene,...),showinfo output."""
    return [float(t) for t in SCENE_RE.findall(log)]


def scene_cuts(path, threshold=0.3):
    run = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-vf", f"select='gt(scene,{threshold})',showinfo",
                          "-an", "-f", "null", "-"], capture_output=True, text=True)
    return parse_scene_times(run.stderr)


def segments(cuts, duration, min_len=3.0, pad=0.1):
    """(start, end) of each scene at least `min_len` long, `pad` seconds clear of the cuts."""
    bounds = [0.0, *sorted(cuts), duration]
    out = []
    for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
        start = a + (pad if i > 0 else 0.0)
        end = b - (pad if i < len(bounds) - 2 else 0.0)
        if end - start >= min_len:
            out.append((round(start, 3), round(end, 3)))
    return out


def duplicate_of(emb, ref, threshold=DUPLICATE):
    """Per row of `emb` (unit vectors), the index of its best match in `ref` if that clears `threshold`, else -1."""
    if len(ref) == 0:
        return np.full(len(emb), -1)
    sims = emb @ ref.T
    best = sims.argmax(axis=1)
    return np.where(sims[np.arange(len(emb)), best] >= threshold, best, -1)


def within_duplicates(emb, threshold=DUPLICATE):
    """Per row, the earlier kept row it repeats (-1 if it is the first of its kind)."""
    out = np.full(len(emb), -1)
    kept = []
    for i in range(len(emb)):
        if kept:
            sims = emb[kept] @ emb[i]
            j = int(sims.argmax())
            if sims[j] >= threshold:
                out[i] = kept[j]
                continue
        kept.append(i)
    return out


def cut(src, start, end, dst):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start}", "-to", f"{end}", "-i", str(src), "-an",
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(dst)], check=True)


class Scorer:
    """SigLIP: kitchen and danger scores (best caption match) and a unit embedding per clip."""

    def __init__(self, model_name="google/siglip-so400m-patch14-384", device=None):
        import torch
        from transformers import AutoModel, AutoProcessor

        from ks.data.filter_kitchen import KITCHEN_PROMPTS

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.prompts = {"kitchen": KITCHEN_PROMPTS, "hazard": HAZARD_PROMPTS}
        with torch.no_grad():
            self.text = {}
            for name, prompts in self.prompts.items():
                t = self.processor(text=prompts, padding="max_length", return_tensors="pt").to(self.device)
                feats = self.model.get_text_features(**t)
                feats = getattr(feats, "pooler_output", feats)
                self.text[name] = feats / feats.norm(dim=-1, keepdim=True)

    def __call__(self, images):
        """images: frames of one clip -> (kitchen score, danger score, unit embedding)."""
        torch = self.torch
        with torch.no_grad():
            pix = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
            img = self.model.get_image_features(pixel_values=pix)
            img = getattr(img, "pooler_output", img)
            img = img / img.norm(dim=-1, keepdim=True)
            scale, bias = self.model.logit_scale.exp(), self.model.logit_bias
            scores = {name: torch.sigmoid(img @ t.T * scale + bias).max().item() for name, t in self.text.items()}
            mean = img.mean(0)
        return scores["kitchen"], scores["hazard"], (mean / mean.norm()).cpu().numpy()


def main():
    import av

    from ks.data.filter_kitchen import frames_at

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="yt-dlp downloads named <youtube id>.mp4")
    ap.add_argument("--videos", required=True, help="videos.csv: youtube_id, kind, title, ...")
    ap.add_argument("--oops-clips", required=True, help="Oops clips already labeled, to skip repeats of them")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="google/siglip-so400m-patch14-384",
                    help="SigLIP checkpoint; siglip-base-patch16-224 ranks well enough on a laptop")
    a = ap.parse_args()

    out = Path(a.out)
    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    with open(a.videos) as f:
        videos = {r["youtube_id"]: r for r in csv.DictReader(f)}

    jobs, rows = [], []
    for src in sorted(Path(a.raw).glob("*.mp4")):
        meta = videos.get(src.stem)
        if meta is None:
            continue
        with av.open(str(src)) as c:
            duration = c.duration / av.time_base
        for n, (start, end) in enumerate(segments(scene_cuts(src), duration)):
            clip_id = f"yt_{src.stem}_{n:03d}"
            rows.append(dict(video_id=clip_id, youtube_id=src.stem, kind=meta["kind"], source_title=meta["title"],
                             start=start, end=end))
            if not (clips_dir / f"{clip_id}.mp4").exists():
                jobs.append((src, start, end, clips_dir / f"{clip_id}.mp4"))
    print(f"{len(rows)} scenes from {len({r['youtube_id'] for r in rows})} videos; cutting {len(jobs)}", flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(lambda j: cut(*j), jobs))

    scorer = Scorer(a.model)
    print(f"scoring on {scorer.device}", flush=True)
    emb = []
    for i, r in enumerate(rows):
        kitchen, hazard, e = scorer(frames_at(clips_dir / f"{r['video_id']}.mp4", FRACTIONS))
        r.update(kitchen_raw=round(kitchen, 4), hazard_score=round(hazard, 4))
        emb.append(e)
        if i % 200 == 199:
            print(f"  scored {i + 1}/{len(rows)}", flush=True)
    emb = np.stack(emb)

    oops, ref = [], []
    for p in sorted(Path(a.oops_clips).glob("*.mp4")):
        try:
            ref.append(scorer(frames_at(p, FRACTIONS))[2])
            oops.append(p)
        except Exception as e:  # one unreadable original (it also failed to normalize) must not stop the run
            print(f"  skipped {p.name}: {type(e).__name__}", flush=True)
    ref = np.stack(ref) if ref else np.zeros((0, emb.shape[1]))
    dup_oops, dup_self = duplicate_of(emb, ref), within_duplicates(emb)
    np.save(out / "embeddings.npy", emb)

    for r, d_o, d_s in zip(rows, dup_oops, dup_self):
        r.update(group=f"yt:{r['youtube_id']}", dup_oops=oops[d_o].stem if d_o >= 0 else "",
                 dup_of=rows[d_s]["video_id"] if d_s >= 0 else "",
                 # The labeling queue sorts by title_match, then kitchen_score: single caught-on-camera
                 # incidents first, then compilation scenes by how dangerous they look.
                 title_match=str(r["kind"] == "single"), kitchen_score=r["hazard_score"], oops_event="",
                 candidate=str(d_o < 0 and d_s < 0))
    fields = ["video_id", "youtube_id", "kind", "group", "start", "end", "kitchen_score", "hazard_score", "kitchen_raw",
              "title_match", "candidate", "dup_oops", "dup_of", "oops_event", "source_title"]
    with open(out / "candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    kept = sum(r["candidate"] == "True" for r in rows)
    print(f"{len(rows)} scenes: {sum(d >= 0 for d in dup_oops)} repeat an Oops clip, "
          f"{sum(d >= 0 for d in dup_self)} repeat another scene; {kept} to label", flush=True)


if __name__ == "__main__":
    main()
