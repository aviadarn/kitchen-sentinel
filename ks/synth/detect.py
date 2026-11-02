"""Can our features tell the synthetic pairs from real footage, and the accident twin from the safe one?

Windows (2 s, 1 s apart) from each generated clip and from the real Charades segment it starts from
go through the frozen encoders; a logistic regression is scored with each pair held out in turn.
High AUC on generated-vs-real means a model trained on the pairs could learn "generated look"
instead of danger; accident-vs-safe shows whether the generated danger is visible to the features.

    uv run --extra model python -m ks.synth.detect --jobs configs/synth_trial.csv \
        --synth data/synth/trial_norm --videos data/charades_480 --out data/synth/trial/detect.json
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ks.features import FPS, NUM_FRAMES, decode_frames


def held_out_auc(x, y, groups, c=0.1):
    """AUC of out-of-group logistic-regression scores, each group held out once."""
    scores = np.empty(len(y))
    for g in np.unique(groups):
        test = groups == g
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
        model.fit(x[~test], y[~test])
        scores[test] = model.predict_proba(x[test])[:, 1]
    return float(roc_auc_score(y, scores))


def windows(path):
    """uint8 (n, 16, H, W, 3): every 2 s window of a normalized clip, one per second."""
    frames = decode_frames(path, 0, 10_000)
    ends = range(NUM_FRAMES, len(frames) + 1, int(FPS))
    return np.stack([frames[e - NUM_FRAMES:e] for e in ends])


def main():
    import torch

    from ks.data.normalize import normalize_video
    from ks.features import Encoders
    from ks.synth.wan_pairs import load_jobs

    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--synth", required=True, help="normalized generated clips <pair>_<kind>.mp4")
    ap.add_argument("--videos", required=True, help="source videos <id>.mp4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    enc = Encoders(["vjepa", "siglip"], "cuda")
    feats, kind, group = [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        for g, job in enumerate(load_jobs(a.jobs).itertuples(index=False)):
            real = Path(tmp) / f"{job.pair}_real.mp4"
            clip = Path(tmp) / f"{job.pair}_cut.mp4"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(job.t), "-t", "5",
                            "-i", str(Path(a.videos) / f"{job.video_id.split(':')[-1]}.mp4"), str(clip)], check=True)
            normalize_video(clip, real)
            for name, path in (("real", real), ("safe", Path(a.synth) / f"{job.pair}_safe.mp4"),
                               ("accident", Path(a.synth) / f"{job.pair}_accident.mp4")):
                w = windows(path)
                out = enc(torch.from_numpy(w).to("cuda"))
                feats.append(np.concatenate([out["vjepa"].astype(np.float32).mean(1),
                                             out["siglip"].astype(np.float32)], axis=1))
                kind += [name] * len(w)
                group += [g] * len(w)
    x, kind, group = np.concatenate(feats), np.array(kind), np.array(group)
    synth = kind != "real"
    result = {}
    for name, cols in (("vjepa", slice(0, 1024)), ("siglip", slice(1024, None))):
        result[f"{name}_generated_vs_real_auc"] = held_out_auc(x[:, cols], synth.astype(int), group)
        result[f"{name}_accident_vs_safe_auc"] = held_out_auc(x[synth][:, cols], (kind[synth] == "accident").astype(int),
                                                              group[synth])
    result["windows"] = {k: int((kind == k).sum()) for k in ("real", "safe", "accident")}
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
