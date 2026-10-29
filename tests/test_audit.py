import json

import numpy as np
import pandas as pd
import pytest

from ks.eval.audit import fire_rates, pre_alert_auc, window_kind


def preds():
    rows = []
    # One fail clip (alert 3.0, event 4.0) and one normal-cooking video, a window every 0.5 s.
    for t, y in [(2.0, 0.0), (2.5, np.nan), (3.0, 1.0), (3.5, 1.0), (4.0, 1.0), (4.5, np.nan)]:
        rows.append(dict(video_id="p", source="oops", video_label=1, fold=0, t_end=t, window_label=y, duration=5.0,
                         events=json.dumps([{"alert": 3.0, "event": 4.0, "hazard": "cut"}])))
    for t in (2.0, 3.0, 4.0, 5.0):
        rows.append(dict(video_id="n", source="charades", video_label=0, fold=0, t_end=t, window_label=0.0,
                         duration=6.0, events="[]"))
    return pd.DataFrame(rows)


def test_window_kind_names_where_each_window_sits():
    assert list(window_kind(preds())) == ["pre-alert", "gap", "warning", "warning", "warning", "after",
                                          "charades", "charades", "charades", "charades"]


def test_fire_rates_are_the_share_of_each_kind_at_or_above_its_folds_tau():
    p = preds().assign(score=[0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.7])
    rates = fire_rates(p, {0: 0.75})
    assert rates["pre-alert"] == 1.0 and rates["warning"] == 1.0
    assert rates["charades"] == pytest.approx(0.0)


def test_pre_alert_auc_measures_scoring_the_source_before_any_danger_is_visible():
    # Pre-alert window above every normal-cooking window: the model tells fail videos apart by look alone.
    p = preds().assign(score=[0.8, 0.0, 0.9, 0.9, 0.9, 0.0, 0.1, 0.2, 0.3, 0.4])
    assert pre_alert_auc(p) == 1.0
    p = preds().assign(score=[0.25, 0.0, 0.9, 0.9, 0.9, 0.0, 0.1, 0.2, 0.3, 0.4])
    assert pre_alert_auc(p) == 0.5


def test_audit_counts_each_accident_once_per_method_however_many_seeds():
    from ks.eval.audit import audit
    from tests.test_cv import synthetic

    index, _ = synthetic()
    perfect = np.where(index.window_label == 1, 0.9, 0.1)
    parts = []
    for method, seeds in (("probe", (0, 1)), ("siglip_lr", (0,))):
        for seed in seeds:
            for k in range(5):
                for role, fold in (("test", k), ("val", (k + 1) % 5)):
                    rows = index.index[index.fold == fold]
                    parts.append(pd.DataFrame(dict(row=rows, method=method, seed=seed, fold=k, role=role,
                                                   score=perfect[rows])))
    shortcut, by_hazard = audit(index, pd.concat(parts, ignore_index=True))
    assert by_hazard.loc[("probe", "cut"), "n"] == 10 and by_hazard.loc[("probe", "cut"), "on_time"] == 1.0
    assert shortcut.loc["probe", "pre_alert_auc"] == 0.5
