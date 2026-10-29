import numpy as np
import pytest
import torch

from ks.train import balance_weights, fold_roles, negative_stratum, sample_batch, score, train_probe


def test_each_fold_is_tested_once_and_validates_the_next():
    roles = [fold_roles(5, k) for k in range(5)]
    assert roles[0] == {0: "test", 1: "val", 2: "train", 3: "train", 4: "train"}
    assert roles[4] == {4: "test", 0: "val", 1: "train", 2: "train", 3: "train"}


def test_negative_stratum_separates_same_clip_negatives_from_each_negative_source():
    assert negative_stratum("oops", 1) == "oops-prealert"
    assert negative_stratum("charades", 0) == "charades"
    assert negative_stratum("epic", 0) == "epic"


def test_weights_balance_classes_and_split_negatives_equally_across_strata():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    strata = np.array(["", "", "oops-prealert", "charades", "charades", "charades", "epic", "epic"])
    w = balance_weights(y, strata)
    assert w[y == 1].sum() == pytest.approx(0.5)
    for s in ("oops-prealert", "charades", "epic"):
        assert w[strata == s].sum() == pytest.approx(0.5 / 3)
    assert w.sum() == pytest.approx(1.0)


def test_sampled_batches_follow_the_weights():
    y = np.array([1] * 10 + [0] * 90)
    strata = np.array([""] * 10 + ["charades"] * 90)
    rng = np.random.default_rng(0)
    picks = np.concatenate([sample_batch(balance_weights(y, strata), 64, rng) for _ in range(200)])
    assert y[picks].mean() == pytest.approx(0.5, abs=0.02)


def separable(n, rng, dim=8, tokens=4):
    """Tokens whose first channel is +1 for positives and -1 for negatives, plus noise."""
    y = rng.integers(0, 2, n)
    x = rng.normal(0, 0.5, (n, tokens, dim)).astype(np.float16)
    x[:, :, 0] += np.where(y == 1, 1.0, -1.0)[:, None]
    return x, y


def test_probe_learns_a_separable_problem_and_scores_new_windows():
    rng = np.random.default_rng(0)
    x, y = separable(400, rng)
    xv, yv = separable(100, rng)
    model, history = train_probe(torch.as_tensor(x), y, np.where(y == 1, "", "charades"), torch.as_tensor(xv), yv,
                                 seed=0, steps=150, batch=32, eval_every=50, hidden_size=8, num_heads=2,
                                 head_hidden=16, device="cpu")
    p = score(model, xv, batch=64, device="cpu")
    assert ((p > 0.5) == yv).mean() > 0.95
    assert len(history) >= 1 and all(0 <= ap <= 1 for _, ap in history)


def test_probe_training_is_reproducible_per_seed():
    rng = np.random.default_rng(1)
    x, y = separable(200, rng)
    xv, yv = separable(50, rng)
    args = (torch.as_tensor(x), y, np.where(y == 1, "", "charades"), torch.as_tensor(xv), yv)
    kw = dict(steps=20, batch=16, eval_every=10, hidden_size=8, num_heads=2, head_hidden=16, device="cpu")
    a = score(train_probe(*args, seed=3, **kw)[0], xv, device="cpu")
    b = score(train_probe(*args, seed=3, **kw)[0], xv, device="cpu")
    np.testing.assert_array_equal(a, b)
