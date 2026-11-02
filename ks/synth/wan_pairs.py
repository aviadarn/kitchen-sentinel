"""Matched synthetic pairs: from one real normal-cooking frame, Wan 2.2 continues the scene twice,
once into an accident and once with nothing happening.

Every positive clip is a YouTube fail compilation, so "looks like a fail video" predicts danger
(reports/v2_*/audit.md). Starting both continuations from a Charades frame gives accidents the
negatives' look, a controlled lead-in, and a twin that differs only in the danger. Generated
clips are for training only; evaluation stays on real footage.

    uv run --extra model --extra gen python -m ks.synth.wan_pairs --jobs configs/synth_trial.csv \
        --videos data/charades_480 --out data/synth/trial
"""

import argparse
import csv
import math
import time
from pathlib import Path

import pandas as pd

from ks.data.manifest import HAZARD_TYPES

MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
AREA = 1280 * 704  # the model's 720p training area; the aspect follows the source frame
STYLE = "Realistic handheld home video in a real kitchen, natural indoor lighting, ordinary people. "
NEGATIVE = ("cartoon, animation, illustration, painting, CGI, text, subtitles, watermark, static frame, frozen, "
            "slow motion, blurry, low quality, JPEG artifacts, deformed hands, extra fingers, extra limbs, "
            "distorted face, morphing objects, objects appearing from nowhere")


def fit_size(width, height, area=AREA, multiple=32):
    """(w, h) with the source's aspect, about `area` pixels, both multiples of `multiple`."""
    aspect = width / height
    h = math.sqrt(area / aspect)
    w = aspect * h
    return int(round(w / multiple)) * multiple, int(round(h / multiple)) * multiple


def load_jobs(path):
    """One row per pair: pair, video_id, t (seconds into the source), hazard, accident, safe."""
    jobs = pd.read_csv(path)
    missing = {"pair", "video_id", "t", "hazard", "accident", "safe"} - set(jobs.columns)
    if missing:
        raise ValueError(f"jobs file lacks columns {sorted(missing)}")
    bad = sorted(set(jobs.hazard) - HAZARD_TYPES)
    if bad:
        raise ValueError(f"unknown hazard types {bad}")
    if jobs.pair.duplicated().any():
        raise ValueError("pair ids must be unique")
    return jobs


def source_frame(path, t):
    """The first decoded frame at or after `t` seconds, as a PIL image."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(t / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.time is not None and frame.time >= t:
                return frame.to_image()
    raise ValueError(f"{path} has no frame at {t:.2f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--videos", required=True, help="directory of source videos named <id>.mp4 (id without prefix)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--frames", type=int, default=121, help="121 frames = 5 s at 24 fps")
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N pairs (timing runs)")
    a = ap.parse_args()

    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from diffusers.utils import export_to_video

    jobs = load_jobs(a.jobs)
    if a.limit:
        jobs = jobs.head(a.limit)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKLWan.from_pretrained(a.model, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(a.model, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
    pipe.vae.enable_tiling()  # the full-frame decode is what runs out of memory on a 24 GB card
    print(f"expand_timesteps={pipe.config.expand_timesteps}", flush=True)

    log = out / "timings.csv"
    new_log = not log.exists()
    with open(log, "a", newline="") as f:
        writer = csv.writer(f)
        if new_log:
            writer.writerow(["pair", "kind", "width", "height", "frames", "steps", "seconds"])
        for n, job in enumerate(jobs.itertuples(index=False)):
            image = source_frame(Path(a.videos) / f"{job.video_id.split(':')[-1]}.mp4", job.t)
            width, height = fit_size(*image.size)
            image = image.resize((width, height))
            image.save(out / f"{job.pair}_source.jpg", quality=92)
            for kind in ("safe", "accident"):
                path = out / f"{job.pair}_{kind}.mp4"
                if path.exists():
                    continue
                t0 = time.time()
                frames = pipe(image=image, prompt=STYLE + getattr(job, kind), negative_prompt=NEGATIVE,
                              height=height, width=width, num_frames=a.frames, guidance_scale=a.guidance,
                              num_inference_steps=a.steps,
                              # Same seed for both twins: they share the start until the prompts diverge.
                              generator=torch.Generator("cuda").manual_seed(n)).frames[0]
                export_to_video(frames, str(path), fps=24)
                seconds = time.time() - t0
                writer.writerow([job.pair, kind, width, height, a.frames, a.steps, f"{seconds:.0f}"])
                f.flush()
                print(f"{job.pair} {kind}: {seconds:.0f} s", flush=True)


if __name__ == "__main__":
    main()
