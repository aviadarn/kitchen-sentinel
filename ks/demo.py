"""Example videos: a clip as the model sees it, with its P(danger) drawn in as the clip plays.

Examples are drawn at random (seeded) from each outcome, so they show typical behaviour rather than
the best cases: accidents caught on time, alarms before any danger was visible, missed accidents,
and false alarms on normal cooking and on YouTube footage where nothing goes wrong. Scores are
out-of-fold: each clip is scored by models that never trained on its fold.

    uv run python -m ks.demo --features features_v2 --run reports/v2_with_youtube --out demos
"""

import argparse
import html
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ks.eval.cv import _attach, accident_outcomes, fold_taus, youtube_normal
from ks.features import FPS, NUM_FRAMES, decode_frames

KINDS = {
    "on_time": "Caught on time",
    "early": "Alarm before any danger was visible",
    "miss": "Missed",
    "false_alarm_cooking": "False alarm: normal cooking",
    "false_alarm_youtube": "False alarm: YouTube footage, nothing goes wrong",
}


def pick_examples(preds, taus, per_kind=2, seed=0, pad=6.0):
    """Up to `per_kind` random clips per outcome, each clip at most once. Fail clips are shown whole;
    false alarms as the `pad` seconds around the alarm (and its 2 s window), within the scored footage."""
    rng = np.random.default_rng(seed)
    videos = preds.groupby("video_id").first()
    picked, rows = set(), []

    def take(kind, candidates, make):
        candidates = [c for c in dict.fromkeys(candidates) if c[0] not in picked]
        for i in rng.permutation(len(candidates))[:per_kind]:
            vid, extra = candidates[i]
            picked.add(vid)
            v = videos.loc[vid]
            hazards = sorted({e.get("hazard") for e in json.loads(v.events)} - {None})
            title = " · ".join([*hazards, vid])
            rows.append(dict(kind=kind, video_id=vid, path=v.path, fold=v.fold, title=title, **make(v, extra)))

    outcomes = accident_outcomes(preds, taus)
    for kind in ("on_time", "early", "miss"):
        take(kind, [(vid, None) for vid in outcomes[outcomes.outcome == kind].video_id.sort_values()],
             lambda v, _: dict(start=0.0, stop=float(v.duration), alarm=np.nan))

    fired = preds[(preds.video_label == 0) & (preds.score >= preds.fold.map(taus))].sort_values(["video_id", "t_end"])
    window = NUM_FRAMES / FPS
    for kind, mask in (("false_alarm_cooking", ~youtube_normal(fired)), ("false_alarm_youtube", youtube_normal(fired))):
        first = fired[mask].groupby("video_id").t_end.first()
        take(kind, list(first.items()),
             lambda v, t: dict(start=max(0.0, t - window - pad), stop=min(float(v.duration), t + pad), alarm=t))
    return pd.DataFrame(rows)


P_FLOOR = 1e-3
INK = {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "series": "#2a78d6", "critical": "#d03b3b"}


def render(example, curve, tau, accidents, out_path, subtitle=""):
    """Write an mp4 at the clip's own 8 fps: the frames on top, P(danger) so far below, with the alarm
    threshold, the labeled warning spans (alert to accident) and an ALARM badge while P >= tau."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first, last = int(round(example.start * FPS)), int(round(example.stop * FPS))
    frames = decode_frames(example.path, first, last)
    curve = curve.sort_values("t_end")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.edgecolor": INK["axis"], "axes.linewidth": 0.8,
                         "xtick.color": INK["muted"], "ytick.color": INK["muted"]})
    fig = plt.figure(figsize=(6.4, 6.0), dpi=100, facecolor=INK["surface"])
    fig.text(0.04, 0.965, KINDS.get(example.kind, example.kind), fontsize=11, weight="bold", color=INK["primary"])
    fig.text(0.04, 0.935, example.title[-80:], fontsize=7.5, color=INK["secondary"])
    ax_img = fig.add_axes((0.04, 0.33, 0.92, 0.585))
    ax_img.set_axis_off()
    image = ax_img.imshow(frames[0], interpolation="bilinear")
    badge = ax_img.text(0.03, 0.95, "⚠ ALARM", transform=ax_img.transAxes, va="top", fontsize=12, weight="bold",
                        color="white", family="DejaVu Sans",
                        bbox=dict(boxstyle="round,pad=0.35", fc=INK["critical"], ec="none"))
    ax = fig.add_axes((0.11, 0.08, 0.85, 0.2), facecolor=INK["surface"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color=INK["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(example.start, example.stop)
    # Logit axis: fold thresholds range from ~0.01 to ~0.99, and both ends must stay readable.
    ax.set_yscale("logit")
    ax.set_ylim(P_FLOOR, 1 - P_FLOOR)
    ax.set_yticks([0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999], ["0.001", "0.01", "0.1", "0.5", "0.9", "0.99", "0.999"])
    ax.minorticks_off()
    ax.set_ylabel("P(danger)", color=INK["secondary"])
    for alert, event in accidents:
        if example.start <= event <= example.stop:
            ax.axvspan(alert, event, color=INK["muted"], alpha=0.18, linewidth=0)
            ax.axvline(event, color=INK["primary"], linewidth=1.0)
            ax.text(event, 1 - P_FLOOR, " accident", fontsize=6.5, va="top", ha="left", color=INK["primary"])
    ax.axhline(tau, color=INK["secondary"], linewidth=0.8)
    ax.text(example.start, tau, " alarm threshold", fontsize=6.5, va="bottom", ha="left", color=INK["secondary"])
    line, = ax.plot([], [], color=INK["series"], linewidth=2, solid_capstyle="round", solid_joinstyle="round")
    cursor = ax.axvline(example.start, color=INK["grid"], linewidth=1.6)
    fig.text(0.04, 0.012, subtitle, fontsize=6.5, color=INK["muted"])

    w, h = fig.canvas.get_width_height()
    ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                               "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                               "-crf", "23", str(out_path)], stdin=subprocess.PIPE)
    for i, frame in enumerate(frames):
        now = (first + i + 1) / FPS  # end of this frame: the newest window ending by now has been scored
        past = curve[curve.t_end <= now + 1e-9]
        image.set_data(frame)
        line.set_data(past.t_end, past.score.clip(P_FLOOR, 1 - P_FLOOR))
        cursor.set_xdata([now, now])
        badge.set_visible(bool(len(past)) and past.score.iloc[-1] >= tau)
        fig.canvas.draw()
        ffmpeg.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait():
        raise RuntimeError(f"ffmpeg failed writing {out_path}")
    plt.close(fig)
    return Path(out_path)


def _gallery(examples):
    cards = []
    for kind, label in KINDS.items():
        part = examples[examples.kind == kind]
        if part.empty:
            continue
        cards.append(f"<h2>{html.escape(label)}</h2><div class=row>")
        for r in part.itertuples():
            cards.append(f"<figure><video src='{html.escape(r.file)}' controls loop muted playsinline></video>"
                         f"<figcaption>{html.escape(r.title)}</figcaption></figure>")
        cards.append("</div>")
    return ("<!doctype html><meta charset=utf-8><title>kitchen-sentinel examples</title>"
            "<style>body{font-family:system-ui,sans-serif;background:#fcfcfb;color:#0b0b0b;margin:24px;max-width:1360px}"
            ".row{display:flex;flex-wrap:wrap;gap:16px}figure{margin:0;width:420px}video{width:100%}"
            "figcaption{font-size:12px;color:#52514e}h2{font-size:16px;margin-top:28px}</style>"
            "<h1>kitchen-sentinel: example predictions</h1><p>Random examples of each outcome from the V-JEPA2 probe "
            "(seed 0), scored out-of-fold. Grey band: labeled warning (alert to accident); vertical line: the "
            "accident.</p>" + "".join(cards))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--run", required=True, help="cross-validation run directory holding scores.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="probe")
    ap.add_argument("--per-kind", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    index = pd.read_csv(Path(a.features) / "index.csv")
    index = index[np.load(Path(a.features) / "done.npy")]
    scores = pd.read_csv(Path(a.run) / "scores.csv")
    scores = scores[(scores.method == a.method) & (scores.seed == 0)]
    taus = fold_taus(index, scores)
    preds = _attach(index, scores[scores.role == "test"])
    examples = pick_examples(preds, taus, per_kind=a.per_kind, seed=a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for n, ex in enumerate(examples.itertuples(index=False)):
        ex = pd.Series(ex._asdict())
        v = preds[preds.video_id == ex.video_id]
        accidents = [(e["alert"], e["event"]) for e in json.loads(v.events.iloc[0])]
        name = f"{n:02d}_{ex.kind}.mp4"
        render(ex, v[["t_end", "score"]], taus[ex.fold], accidents, out / name,
               subtitle=f"Model's view (8 fps, squeezed to 256 px). {a.method} seed 0, out-of-fold; "
                        f"threshold {taus[ex.fold]:.3f} = at most 1 false alarm/h on cooking.")
        files.append(name)
        print(f"{name}: {ex.video_id} [{ex.start:.1f}-{ex.stop:.1f} s]", flush=True)
    examples["file"] = files
    examples.to_csv(out / "examples.csv", index=False)
    (out / "index.html").write_text(_gallery(examples))


if __name__ == "__main__":
    main()
