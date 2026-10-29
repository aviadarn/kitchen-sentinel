"""Phase 3 feature cache: decode every window once and store what the probe and baselines need.

Per window: pooled frozen V-JEPA2 tokens (the probe's input), BADAS-Open's zero-shot P(danger), and a
SigLIP embedding of the window's last frame (the single-frame baseline). Positive clips are windowed
at every frame (risk curves, warning time); negative videos every second (false alarms per hour).

    uv run python -m ks.features --manifest data/manifest.csv --out features --workers 8
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap

from ks.data.manifest import parse_events
from ks.data.windows import sample_windows

FPS = 8.0
NUM_FRAMES = 16


def feature_index(manifest, pos_stride=1, neg_stride=8, fps=FPS, num_frames=NUM_FRAMES):
    """One row per window, in manifest order; the row number is the window's row in every feature file."""
    rows = []
    for v in manifest.itertuples(index=False):
        events = [(e["alert"], e["event"]) for e in parse_events(v.events)]
        windows = sample_windows(v.duration, events, fps=fps, num_frames=num_frames,
                                 stride_frames=pos_stride if v.label == 1 else neg_stride, keep_unlabeled=True)
        for w in windows:
            rows.append(dict(video_id=v.video_id, source=v.source, group=v.group, fold=v.fold, video_label=v.label,
                             path=v.path, start_frame=w.start_frame, end_frame=w.end_frame, t_end=w.t_end,
                             window_label=np.nan if w.label is None else float(w.label), events=v.events,
                             duration=v.duration))
    return pd.DataFrame(rows)


def chunk_plan(index, max_windows=64):
    """(path, rows, end_frames) work items of at most `max_windows` consecutive windows of one video."""
    chunks = []
    for path, part in index.groupby("path", sort=False):
        rows, ends = part.index.to_numpy(copy=True), part.end_frame.to_numpy(copy=True)
        for i in range(0, len(rows), max_windows):
            chunks.append((path, rows[i:i + max_windows], ends[i:i + max_windows]))
    return chunks


def decode_frames(path, start, stop):
    """RGB frames [start, stop) of a normalized video as uint8 (F, H, W, 3); fewer if the video ends,
    none if the file isn't there (yet: extraction can run while videos upload, then resume)."""
    import av

    if not Path(path).exists():
        return np.zeros((0, 1, 1, 3), np.uint8)
    frames = []
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i >= stop:
                break
            if i >= start:
                frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames) if frames else np.zeros((0, 1, 1, 3), np.uint8)


class _Chunks(torch.utils.data.Dataset):
    def __init__(self, chunks, decode, num_frames):
        self.chunks, self.decode, self.num_frames = chunks, decode, num_frames

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        path, rows, ends = self.chunks[i]
        start = int(ends.min()) - self.num_frames
        return rows, ends - start, self.decode(path, start, int(ends.max()))


def _same_windows(a, b):
    return len(a) == len(b) and (a.video_id.astype(str).to_numpy() == b.video_id.astype(str).to_numpy()).all() \
        and (a.end_frame.to_numpy() == b.end_frame.to_numpy()).all()


def extract(index, encode, specs, out_dir, decode=decode_frames, max_windows=64, workers=0, device="cpu",
            num_frames=NUM_FRAMES, log=print):
    """Fill `<out_dir>/<name>.npy` (one row per index row) with `encode(windows)[name]`; resumable.

    `encode` takes uint8 windows (n, num_frames, H, W, 3) and returns {name: array (n, *shape)};
    `specs` maps name -> (shape, dtype). Returns how many windows remain undone (videos that decode to
    fewer frames than their manifest duration implies).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "index.csv").exists():
        if not _same_windows(pd.read_csv(out / "index.csv"), index):
            raise ValueError(f"{out}/index.csv differs from this feature index; use a new --out")
    else:
        index.to_csv(out / "index.csv", index=False)
    n = len(index)
    arrays = {name: open_memmap(out / f"{name}.npy", mode="r+" if (out / f"{name}.npy").exists() else "w+",
                                dtype=dtype, shape=(n, *shape)) for name, (shape, dtype) in specs.items()}
    done = np.load(out / "done.npy") if (out / "done.npy").exists() else np.zeros(n, dtype=bool)

    todo = [c for c in chunk_plan(index, max_windows) if not done[c[1]].all()]
    loader = torch.utils.data.DataLoader(_Chunks(todo, decode, num_frames), batch_size=None, num_workers=workers)
    t0, finished = time.time(), 0
    for i, (rows, ends, frames) in enumerate(loader):
        rows, ends = np.asarray(rows), np.asarray(ends)
        complete = ends <= len(frames)
        rows, ends = rows[complete], ends[complete]
        if len(rows):
            frames = torch.as_tensor(frames).to(device)
            offsets = torch.as_tensor(ends[:, None] - num_frames + np.arange(num_frames), device=frames.device)
            for name, values in encode(frames[offsets]).items():
                arrays[name][rows] = np.asarray(values)
            done[rows] = True
            finished += len(rows)
        if i % 20 == 19 or i == len(todo) - 1:
            for a in arrays.values():
                a.flush()
            np.save(out / "done.npy", done)
            rate = finished / (time.time() - t0)
            log(f"{done.sum()}/{n} windows, {rate:.1f}/s, eta {(n - done.sum()) / max(rate, 1e-9) / 60:.0f} min")
    for a in arrays.values():
        a.flush()
    np.save(out / "done.npy", done)
    return int((~done).sum())


class Encoders:
    """Frozen V-JEPA2 tokens, BADAS-Open zero-shot score and SigLIP last-frame embedding in one call."""

    def __init__(self, names, device, batch=16, spatial_pool=2, siglip=None):
        from transformers import AutoModel, AutoProcessor, AutoVideoProcessor, VJEPA2Model

        from ks.models.vjepa_anticipator import BACKBONE, Anticipator, load_badas

        self.names, self.device, self.batch = names, device, batch
        self.specs = {}
        self.video_processor = AutoVideoProcessor.from_pretrained(BACKBONE)
        if "vjepa" in names:
            self.vjepa = Anticipator(VJEPA2Model.from_pretrained(BACKBONE), spatial_pool=spatial_pool).to(device).eval()
            grid = NUM_FRAMES // self.vjepa.backbone.config.tubelet_size * (self.vjepa.hw[0] // spatial_pool) ** 2
            self.specs["vjepa"] = ((grid, self.vjepa.backbone.config.hidden_size), np.float16)
        if "badas" in names:
            self.badas = load_badas(VJEPA2Model.from_pretrained(BACKBONE)).to(device).eval()
            self.specs["badas"] = ((), np.float32)
        if "siglip" in names:
            self.siglip = AutoModel.from_pretrained(siglip or SIGLIP).to(device).eval()
            self.siglip_processor = AutoProcessor.from_pretrained(siglip or SIGLIP)
            self.specs["siglip"] = ((self.siglip.config.vision_config.hidden_size,), np.float16)

    @torch.no_grad()
    def __call__(self, windows):
        out = {name: [] for name in self.specs}
        for i in range(0, len(windows), self.batch):
            clips = windows[i:i + self.batch].permute(0, 1, 4, 2, 3)  # (n, T, C, H, W)
            with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
                if "vjepa" in self.specs or "badas" in self.specs:
                    pixels = self.video_processor(videos=clips, return_tensors="pt", device=self.device)
                    pixels = pixels["pixel_values_videos"]
                if "vjepa" in self.specs:
                    out["vjepa"].append(self.vjepa.encode(pixels).half().cpu())
                if "badas" in self.specs:
                    logits = self.badas(pixels).float()
                    out["badas"].append(torch.softmax(logits / BADAS_TEMPERATURE, dim=1)[:, 1].cpu())
                if "siglip" in self.specs:
                    images = self.siglip_processor(images=clips[:, -1], return_tensors="pt", device=self.device)
                    emb = self.siglip.get_image_features(pixel_values=images["pixel_values"]).pooler_output
                    out["siglip"].append(emb.half().cpu())
        return {name: torch.cat(parts).numpy() for name, parts in out.items()}


SIGLIP = "google/siglip-so400m-patch14-384"
BADAS_TEMPERATURE = 2.0  # BADAS-Open's inference divides logits by 2 before the softmax, as in training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=["vjepa", "badas", "siglip"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-windows", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--siglip", default=SIGLIP)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = feature_index(pd.read_csv(a.manifest))
    print(f"{len(index)} windows from {index.video_id.nunique()} videos on {device}", flush=True)
    enc = Encoders(a.models, device, batch=a.batch, siglip=a.siglip)
    missing = extract(index, enc, enc.specs, a.out, max_windows=a.max_windows, workers=a.workers, device=device,
                      log=lambda s: print(s, flush=True))
    print(f"done; {missing} windows could not be decoded", flush=True)


if __name__ == "__main__":
    main()
