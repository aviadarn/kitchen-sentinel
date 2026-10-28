"""Find kitchen clips among the ~30k Oops! fails: SigLIP zero-shot scene scores plus title keywords.

Tuned for recall. Every candidate is reviewed by hand in the labeling tool, so a false positive
costs a keypress while a false negative is a lost training example.

    uv run python -m ks.data.filter_kitchen --videos <dir> --ann <annotations dir> --out candidates.csv
"""

import argparse
import csv
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av

TITLE_RE = re.compile(r"kitchen|cook|chef|food|\bbak(e|er|ers|ing)\b|grill|bbq|barbecue|knife|knives", re.I)

POSITIVE_PROMPTS = [
    "a photo of a kitchen", "a person cooking at a stove", "food being prepared on a kitchen counter",
    "a person cutting food with a knife", "a frying pan on a stove", "a restaurant kitchen",
    "a person cooking on a barbecue grill", "a person taking something out of an oven",
]
NEGATIVE_PROMPTS = [
    "a photo of a backyard", "a street", "a skate park", "a swimming pool", "a gym", "a living room",
    "a bedroom", "a beach", "a snowy mountain", "a grass field", "a bathroom", "a garage", "a store",
    "a playground", "a lake", "a stage", "a car interior", "an office",
]
FRACTIONS = (0.25, 0.5, 0.75)


def title_match(title):
    return TITLE_RE.search(title) is not None


def frames_at(path, fractions):
    """RGB PIL frames at the given fractions of the clip's duration."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None:
            duration = float(stream.duration * stream.time_base)
        else:
            duration = container.duration / av.time_base
        frames = []
        for fraction in fractions:
            target = fraction * duration
            container.seek(int(target / stream.time_base), stream=stream)  # lands on a keyframe at or before
            picked = None
            for frame in container.decode(stream):
                picked = frame
                if frame.time is not None and frame.time >= target - 1e-3:
                    break
            frames.append(picked.to_image())
        return frames


class KitchenScorer:
    """P(kitchen scene) per image: softmax over kitchen and non-kitchen prompts, kitchen mass summed."""

    def __init__(self, model_name="google/siglip-so400m-patch14-384", device=None):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.prompts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS

    def __call__(self, images):
        inputs = self.processor(text=self.prompts, images=images, padding="max_length", return_tensors="pt")
        with self.torch.no_grad():
            logits = self.model(**inputs.to(self.device)).logits_per_image
        probs = logits.softmax(dim=-1)
        return probs[:, : len(POSITIVE_PROMPTS)].sum(dim=-1).cpu().tolist()


def _safe_frames(path):
    try:
        return str(path), frames_at(path, FRACTIONS)
    except Exception as e:  # a corrupt clip must not stop a 30k-clip run
        return str(path), e


def main():
    from ks.data.sources.oops import load_index

    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="directory searched recursively for *.mp4")
    ap.add_argument("--ann", required=True, help="Oops annotations directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=64, help="clips per scoring batch")
    # Containers often report the host's core count; pass the real share explicitly.
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    index = {r["video_id"]: r for r in load_index(args.ann)}
    out = Path(args.out)
    done = set()
    if out.exists():
        with out.open() as f:
            done = {r["video_id"] for r in csv.DictReader(f)}
    paths = sorted(p for p in Path(args.videos).rglob("*.mp4") if p.stem not in done)
    print(f"{len(index)} indexed clips; {len(done)} already scored; {len(paths)} to score")

    fields = ["video_id", "path", "group", "oops_split", "oops_event", "event_spread", "n_annotators",
              "kitchen_score", "title_match", "candidate", "error"]
    scorer = KitchenScorer()
    # spawn, not fork: the parent has CUDA initialised, and decode workers only need PyAV.
    spawn = multiprocessing.get_context("spawn")
    with out.open("a", newline="") as f, ProcessPoolExecutor(args.workers, mp_context=spawn) as pool:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not done:
            writer.writeheader()
        for start in range(0, len(paths), args.batch):
            batch = list(pool.map(_safe_frames, paths[start:start + args.batch]))
            ok = [(p, frames) for p, frames in batch if not isinstance(frames, Exception)]
            scores = scorer([img for _, frames in ok for img in frames]) if ok else []
            per_clip = {p: max(scores[i * len(FRACTIONS):(i + 1) * len(FRACTIONS)]) for i, (p, _) in enumerate(ok)}
            for p, frames in batch:
                stem = Path(p).stem
                meta = index.get(stem, {"group": "", "oops_split": "", "oops_event": None, "event_spread": None,
                                        "n_annotators": 0})
                score = per_clip.get(p)
                titled = title_match(meta["group"] or stem)
                writer.writerow({
                    "video_id": stem, "path": p, **{k: meta[k] for k in fields[2:7]},
                    "kitchen_score": "" if score is None else round(score, 4), "title_match": titled,
                    "candidate": titled or (score is not None and score >= args.threshold),
                    "error": "" if score is not None else repr(frames)[:200],
                })
            f.flush()
            print(f"{min(start + args.batch, len(paths))}/{len(paths)}", flush=True)


if __name__ == "__main__":
    main()
