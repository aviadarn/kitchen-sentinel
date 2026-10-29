"""Site-card loop: one out-of-fold example in the 16:9 layout, as a web-ready mp4.

    uv run python scripts/card_loop.py reports/card_loop.mp4
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ks.demo import pick_examples, render
from ks.eval.cv import _attach, fold_taus

ROOT = Path(__file__).resolve().parents[1]
CLIP = "Fails of the Week - Go ahead and send it! (May 2017)9"  # a pan catching fire, caught on time


def main(out):
    index = pd.read_csv(ROOT / "features_v2" / "index.csv")
    index = index[np.load(ROOT / "features_v2" / "done.npy")]
    scores = pd.read_csv(ROOT / "reports" / "v2_with_youtube" / "scores.csv")
    scores = scores[(scores.method == "probe") & (scores.seed == 0)]
    taus = fold_taus(index, scores)
    preds = _attach(index, scores[scores.role == "test"])
    examples = pick_examples(preds, taus, per_kind=100, seed=0)
    ex = examples[examples.video_id == CLIP].iloc[0]
    v = preds[preds.video_id == CLIP]
    accidents = [(e["alert"], e["event"]) for e in json.loads(v.events.iloc[0])]
    render(ex, v[["t_end", "score"]], taus[ex.fold], accidents, out, layout="wide", hold=1.5,
           subtitle="V-JEPA2 probe, scored by a model that never trained on this clip. Model's view: 8 fps, 256 px.")
    print(out, ex.kind)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "reports" / "card_loop.mp4")
