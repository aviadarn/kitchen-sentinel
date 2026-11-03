"""Site-card figure: false alarms per hour on normal cooking vs calm YouTube footage, same threshold.

    uv run python scripts/card_figure.py reports/false_alarms.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "reports"
INK = {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}
SERIES = [("fa_per_h", "normal cooking (Charades, EPIC-KITCHENS)", "#2a78d6"),
          ("fa_per_h_youtube", "YouTube fail clips, before anything goes wrong", "#eb6834")]


def main(out):
    without = json.loads((REPORTS / "v3_without_youtube" / "metrics.json").read_text())
    with_yt = json.loads((REPORTS / "v3_with_youtube" / "metrics.json").read_text())
    rows = [("single frame (SigLIP)", without["siglip_lr"]),
            ("video model (V-JEPA2 probe)", without["probe"]),
            ("video model, also trained on\n3 h of calm YouTube footage", with_yt["probe"])]

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 9, "axes.edgecolor": INK["axis"], "xtick.color": INK["muted"],
                         "ytick.color": INK["secondary"]})
    fig, ax = plt.subplots(figsize=(8.5, 3.4), dpi=200, facecolor=INK["surface"])
    ax.set_facecolor(INK["surface"])
    height, gap = 0.3, 0.02
    ax_pad = max(m[k] for _, m in rows for k, _, _ in SERIES) * 0.012
    for i, (_, metrics) in enumerate(rows):
        for j, (key, _, color) in enumerate(SERIES):
            y = i + (j - 0.5) * (height + gap)
            value = metrics[key]
            ax.barh(y, value, height=height, color=color, linewidth=0)
            ax.text(value + ax_pad, y, f"{value:.1f}", va="center", fontsize=8.5, color=INK["primary"])
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(m[k] for _, m in rows for k, _, _ in SERIES) * 1.1)
    ax.set_xlabel("false alarms per hour, at the threshold set for at most 1 per hour on normal cooking",
                  color=INK["secondary"])
    ax.grid(axis="x", color=INK["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=label) for _, label, c in SERIES]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK["primary"],
               bbox_to_anchor=(0.5, 0.0))
    fig.text(0.01, 0.95, "5-fold cross-validation, out-of-fold scores; the same threshold applies to both kinds "
             "of footage", fontsize=8.5, color=INK["secondary"])
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    fig.savefig(out, facecolor=INK["surface"])
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else REPORTS / "false_alarms.png")
