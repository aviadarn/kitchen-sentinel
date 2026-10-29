"""Shortcut audit and error analysis of a cross-validated run, from its saved scores (no GPU).

Positives all come from YouTube fail compilations and negatives from Charades / EPIC, so a model can
score well by recognising the *source*. Its pre-alert windows (same fail clip, before any danger is
visible) give it away: a model that anticipates scores them like normal cooking; one that reads the
source scores them like the warning windows.

    uv run python -m ks.eval.audit --features features --run reports/pilot
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ks.eval.cv import _attach, accident_outcomes, fold_taus
from ks.eval.protocol import accident_curves

KINDS = ["pre-alert", "gap", "warning", "after", "charades", "epic"]


def window_kind(preds):
    """warning / pre-alert / gap (just before the alert) / after (an accident) for fail-clip windows;
    the source name for normal-cooking windows."""
    def kind(r):
        if r.video_label == 0:
            return r.source
        if r.window_label == 1:
            return "warning"
        if r.window_label == 0:
            return "pre-alert"
        return "gap" if r.t_end < _first_alert(r.events) else "after"

    return [kind(r) for r in preds.itertuples(index=False)]


def _first_alert(events):
    events = json.loads(events) if isinstance(events, str) else events
    return min(e["alert"] for e in events)


def fire_rates(preds, taus):
    """Share of each window kind at or above its fold's alarm threshold."""
    fired = preds.score.to_numpy() >= preds.fold.map(taus).to_numpy()
    return pd.Series(fired).groupby(pd.Series(window_kind(preds))).mean().to_dict()


def pre_alert_auc(preds):
    """AUC of pre-alert windows vs normal-cooking windows: 0.5 if the model can't tell a fail clip
    from cooking until danger is visible, 1.0 if it scores the source."""
    p = preds[((preds.video_label == 1) & (preds.window_label == 0)) | (preds.video_label == 0)]
    return float(roc_auc_score(p.video_label, p.score))


def audit(index, scores):
    """Per method (mean over seeds): pre-alert AUC, fire rates by window kind, outcomes by hazard."""
    shortcut, hazards = [], []
    for (method, seed), s in scores.groupby(["method", "seed"]):
        taus = fold_taus(index, s)
        preds = _attach(index, s[s.role == "test"])
        shortcut.append(dict(method=method, seed=seed, pre_alert_auc=pre_alert_auc(preds),
                             **{f"fires@{k}": v for k, v in fire_rates(preds, taus).items()}))
        hazards.append(accident_outcomes(preds, taus).assign(method=method, seed=seed))
    shortcut = pd.DataFrame(shortcut).drop(columns="seed").groupby("method").mean()
    hazards = pd.concat(hazards)
    groups = hazards.groupby(["method", "hazard"])
    per_seed_n = (groups.size() / groups.seed.nunique()).rename("n")
    by_hazard = groups.outcome.value_counts(normalize=True).unstack(fill_value=0).join(per_seed_n)
    return shortcut, by_hazard


INK = {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}
SERIES = {"probe": ("#2a78d6", "probe (seed 0)"), "siglip_lr": ("#eb6834", "SigLIP last frame")}


def plot_curves(index, scores, path, n=20, seed=0):
    """Small multiples: P(danger) over time for `n` random fail clips, probe vs single-frame SigLIP,
    each with its own fold threshold; the labeled warning span shaded."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {m: scores[(scores.method == m) & (scores.seed == 0)] for m in SERIES}
    taus = {m: fold_taus(index, s) for m, s in runs.items()}
    preds = {m: _attach(index, s[s.role == "test"]) for m, s in runs.items()}
    outcomes = {m: accident_outcomes(preds[m], taus[m]) for m in SERIES}
    seen = sorted(set(outcomes["probe"].video_id))
    clips = sorted(np.random.default_rng(seed).choice(seen, size=min(n, len(seen)), replace=False))

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 7, "axes.edgecolor": INK["axis"], "axes.linewidth": 0.8,
                         "xtick.color": INK["muted"], "ytick.color": INK["muted"]})
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(len(clips) / cols)), cols, figsize=(11, 2.4 * np.ceil(len(clips) / cols)),
                             sharey=True, facecolor=INK["surface"])
    for ax, clip in zip(axes.flat, clips):
        ax.set_facecolor(INK["surface"])
        ax.grid(axis="y", color=INK["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        notes = []
        for method, (color, _) in SERIES.items():
            v = preds[method][preds[method].video_id == clip].sort_values("t_end")
            ax.plot(v.t_end, v.score, color=color, linewidth=1.4, solid_capstyle="round")
            ax.axhline(taus[method][v.fold.iloc[0]], color=color, linewidth=0.7, alpha=0.55)
            o = outcomes[method][outcomes[method].video_id == clip].outcome.tolist()
            notes.append(f"{'probe' if method == 'probe' else 'SigLIP'}: {', '.join(x.replace('_', ' ') for x in o)}")
        for c in accident_curves(preds["probe"][preds["probe"].video_id == clip]):
            ax.axvspan(c["alert"], c["event"], color=INK["muted"], alpha=0.18, linewidth=0)
            ax.axvline(c["event"], color=INK["secondary"], linewidth=0.8)
        hazard = outcomes["probe"][outcomes["probe"].video_id == clip].hazard.iloc[0]
        name = clip if len(clip) <= 34 else "…" + clip[-33:]
        ax.set_title(f"{name}  ·  {hazard}", fontsize=6.5, color=INK["primary"], loc="left", pad=13)
        ax.text(0, 1.03, "   ".join(notes), transform=ax.transAxes, fontsize=6, va="bottom", color=INK["secondary"])
        ax.set_ylim(0, 1.02)
    for ax in axes.flat[len(clips):]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], color=c, linewidth=1.4, label=label) for c, label in SERIES.values()]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=INK["muted"], alpha=0.18, label="labeled warning (alert to event)"))
    fig.legend(handles=handles, loc="upper left", ncol=3, frameon=False, fontsize=8, labelcolor=INK["primary"],
               bbox_to_anchor=(0.01, 1.0))
    fig.text(0.01, 0.955, "P(danger) per 2 s window vs window end time (s); thin horizontal lines = each model's "
             "alarm threshold for that fold (<= 1 false alarm/h)", fontsize=7, color=INK["secondary"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor=INK["surface"])
    plt.close(fig)
    return clips


def _markdown(shortcut, by_hazard):
    cols = [c for c in (f"fires@{k}" for k in KINDS) if c in shortcut]
    lines = ["## Shortcut audit", "",
             "pre-alert AUC: pre-alert windows of fail clips vs normal-cooking windows (0.5 = can't tell them apart "
             "before danger is visible). Fires: share of windows at or above the fold's alarm threshold.", "",
             "| method | pre-alert AUC | " + " | ".join(c.removeprefix("fires@") for c in cols) + " |",
             "|---" * (2 + len(cols)) + "|"]
    for method, r in shortcut.iterrows():
        lines.append(f"| {method} | {r.pre_alert_auc:.3f} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    lines += ["", "## Outcome by hazard (share of accidents; probe averaged over seeds)", "",
              "| method | hazard | n | on time | early | miss |", "|---|---|---|---|---|---|"]
    for (method, hazard), r in by_hazard.iterrows():
        if method in ("probe", "siglip_lr"):
            lines.append(f"| {method} | {hazard} | {r.n:.0f} | {r.get('on_time', 0):.2f} | {r.get('early', 0):.2f} | "
                         f"{r.get('miss', 0):.2f} |")
    lines += ["", "![risk curves](curves.png)", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--run", required=True, help="directory holding scores.csv; audit.md and curves.png go there")
    a = ap.parse_args()
    index = pd.read_csv(Path(a.features) / "index.csv")
    index = index[np.load(Path(a.features) / "done.npy")]
    scores = pd.read_csv(Path(a.run) / "scores.csv")
    shortcut, by_hazard = audit(index, scores)
    clips = plot_curves(index, scores, Path(a.run) / "curves.png")
    (Path(a.run) / "audit.md").write_text(_markdown(shortcut, by_hazard))
    print(_markdown(shortcut, by_hazard))
    print("plotted:", *clips, sep="\n  ")


if __name__ == "__main__":
    main()
