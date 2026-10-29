import json

import numpy as np
import pandas as pd
import pytest

from ks.eval.cv import report, run_cv, summarize, train_rows
from ks.features import feature_index


def synthetic(n_pos=10, n_neg=5, folds=5):
    """n_pos 6 s clips (alert 3.5, event 5.0) and n_neg 60 s negative videos, spread over the folds."""
    rows = [dict(video_id=f"p{i}", source="oops", path=f"p{i}.mp4", view="mixed", group=f"gp{i}", fold=i % folds,
                 label=1, events=json.dumps([{"alert": 3.5, "event": 5.0, "hazard": "cut"}]), duration=6.0)
            for i in range(n_pos)]
    rows += [dict(video_id=f"n{i}", source="charades" if i % 2 else "epic", path=f"n{i}.mp4", view="third",
                  group=f"gn{i}", fold=i % folds, label=0, events="[]", duration=60.0) for i in range(n_neg)]
    index = feature_index(pd.DataFrame(rows))
    rng = np.random.default_rng(0)
    signal = (index.window_label == 1).to_numpy()
    tokens = rng.normal(0, 0.5, (len(index), 4, 8)).astype(np.float16)
    tokens[signal, :, 0] += 1.5
    feats = {"vjepa": tokens, "vjepa_mean": tokens.mean(1), "siglip": rng.normal(0, 1, (len(index), 6)),
             "badas": rng.uniform(0, 1, len(index)).astype(np.float32)}
    return index, feats


def test_train_rows_are_labeled_train_fold_windows_capped_per_negative_stratum():
    index, _ = synthetic()
    index["role"] = index.fold.map({0: "test", 1: "val", 2: "train", 3: "train", 4: "train"})
    rows = train_rows(index, max_per_stratum=20, rng=np.random.default_rng(0))
    picked = index.loc[rows]
    assert set(picked.role) == {"train"}
    assert picked.window_label.notna().all()
    assert (picked.window_label == 1).sum() == ((index.role == "train") & (index.window_label == 1)).sum()
    assert picked[picked.video_label == 0].groupby("source").size().max() <= 20


def perfect_preds():
    index, _ = synthetic()
    preds = index.copy()
    preds["score"] = np.where(preds.window_label == 1, 0.9, 0.1)
    return preds


def test_summarize_perfect_model():
    m = summarize(perfect_preds(), taus=dict.fromkeys(range(5), 0.5))
    assert m["hard_ap"] == pytest.approx(1.0)
    assert m["recall"] == 1.0 and m["fa_per_h"] == 0.0
    assert m["mtta"] == pytest.approx(1.5)  # first window at/after alert 3.5, event 5.0
    assert m["easy_ap@0.5"] == pytest.approx(1.0)


def test_summarize_uses_each_folds_own_threshold():
    # tau 0.95 on fold 0 misses its 2 clips; others catch theirs.
    taus = dict.fromkeys(range(5), 0.5) | {0: 0.95}
    assert summarize(perfect_preds(), taus)["recall"] == pytest.approx(8 / 10)


def test_run_cv_scores_every_window_once_as_test_and_report_has_every_method(tmp_path):
    index, feats = synthetic()
    scores = run_cv(index, feats, folds=5, seeds=[0, 1], max_per_stratum=200, device="cpu",
                    probe_kw=dict(steps=100, batch=32, lr=1e-3, eval_every=20, hidden_size=8, num_heads=2,
                                  head_hidden=16),
                    lr_grid=(0.1,))
    test = scores[scores.role == "test"]
    for (method, seed), part in test.groupby(["method", "seed"]):
        assert sorted(part.row) == list(index.index), (method, seed)
    assert set(scores.method) == {"prior", "badas", "siglip_lr", "vjepa_lr", "probe"}
    results = report(scores, index, tmp_path)
    assert set(results) == {"prior", "badas", "siglip_lr", "vjepa_lr", "probe"}
    assert results["probe"]["hard_ap"] > 0.9
    assert (tmp_path / "results.md").exists() and (tmp_path / "metrics.json").exists()
