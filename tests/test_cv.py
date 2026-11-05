import json

import numpy as np
import pandas as pd
import pytest

from ks.eval.cv import report, run_cv, summarize, train_rows, zero_shot
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


def test_load_features_caches_mean_tokens(tmp_path):
    from ks.eval.cv import load_features

    rng = np.random.default_rng(0)
    tokens = rng.normal(0, 1, (5000, 4, 8)).astype(np.float16)
    np.save(tmp_path / "vjepa.npy", tokens)
    np.save(tmp_path / "badas.npy", np.zeros(5000, np.float32))
    np.save(tmp_path / "siglip.npy", np.zeros((5000, 3), np.float16))
    feats = load_features(tmp_path, device="cpu")
    np.testing.assert_allclose(feats["vjepa_mean"], tokens.astype(np.float32).mean(1), rtol=1e-3, atol=1e-3)
    assert (tmp_path / "vjepa_mean.npy").exists()


def test_summarize_separates_alarms_before_the_alert_from_on_time_ones():
    # Firing from the very first window (2.0 s) is 1.5 s before alert - 0.5 s: an early fire, not anticipation.
    preds = perfect_preds()
    preds.loc[preds.video_label == 1, "score"] = 0.9
    m = summarize(preds, taus=dict.fromkeys(range(5), 0.5))
    assert m["recall"] == 1.0
    assert m["early"] == 1.0 and m["on_time_recall"] == 0.0
    assert summarize(perfect_preds(), taus=dict.fromkeys(range(5), 0.5))["on_time_recall"] == 1.0


def test_accident_outcomes_list_each_accident_with_its_hazard_and_first_alarm():
    from ks.eval.cv import accident_outcomes

    preds = perfect_preds()
    preds.loc[preds.video_id == "p0", "score"] = 0.9  # fires from its first window: early
    preds.loc[preds.video_id == "p1", "score"] = 0.1  # never fires: miss
    out = accident_outcomes(preds, dict.fromkeys(range(5), 0.5)).set_index("video_id")
    assert len(out) == 10 and set(out.hazard) == {"cut"}
    assert (out.loc["p0", "outcome"], out.loc["p0", "first_alarm"]) == ("early", 2.0)
    assert out.loc["p1", "outcome"] == "miss" and np.isnan(out.loc["p1", "first_alarm"])
    assert (out.loc["p2", "outcome"], out.loc["p2", "first_alarm"]) == ("on_time", 3.5)


def with_youtube_negatives():
    """synthetic() plus two same-source negative clips (source oops, label 0) that a source-reading
    model scores high."""
    index, feats = synthetic()
    extra = pd.DataFrame([dict(video_id=f"y{i}", source="oops", path=f"y{i}.mp4", view="mixed", group=f"gy{i}",
                               fold=i, label=0, events="[]", duration=10.0) for i in range(2)])
    yt = feature_index(extra)
    return pd.concat([index, yt], ignore_index=True)


def test_false_alarms_are_reported_separately_for_youtube_normal_footage():
    index = with_youtube_negatives()
    preds = index.copy()
    preds["score"] = np.where(preds.window_label == 1, 0.9, 0.1)
    preds.loc[preds.video_id.str.startswith("y"), "score"] = 0.9  # fires on every same-source window
    m = summarize(preds, taus=dict.fromkeys(range(5), 0.5))
    assert m["fa_per_h"] == 0.0  # normal cooking: quiet
    assert m["fa_per_h_youtube"] == pytest.approx(2 * 5 / (20 / 3600))  # 9 windows, 1 s apart, 2 s refractory


def test_train_rows_can_leave_a_negative_stratum_out():
    index = with_youtube_negatives()
    index["role"] = "train"
    rng = np.random.default_rng(0)
    assert "y0" in set(index.loc[train_rows(index, 100, rng)].video_id)
    assert "y0" not in set(index.loc[train_rows(index, 100, rng, exclude=("oops",))].video_id)


def test_zero_shot_scores_match_run_cv_for_a_model_used_as_is(tmp_path):
    # A zero-shot model's per-window scores are added to an earlier run without retraining anything,
    # with the same val/test rows per round that run_cv gives BADAS-Open
    index, feats = synthetic()
    scores = run_cv(index, feats, folds=5, seeds=[0], max_per_stratum=200, device="cpu",
                    probe_kw=dict(steps=20, batch=32, eval_every=10, hidden_size=8, num_heads=2, head_hidden=16),
                    lr_grid=(0.1,))
    badas = scores[scores.method == "badas"].reset_index(drop=True)
    again = zero_shot(index, feats["badas"], "badas", folds=5)
    pd.testing.assert_frame_equal(again, badas, check_dtype=False)


def test_zero_shot_skips_windows_the_index_leaves_out():
    # ks.eval.cv drops undone windows; their rows must not be scored
    index, feats = synthetic()
    kept = index.iloc[::2]
    s = zero_shot(kept, feats["badas"], "cosmos", folds=5)
    assert set(s.row) == set(kept.index)
    assert np.allclose(s.score, feats["badas"][s.row])
