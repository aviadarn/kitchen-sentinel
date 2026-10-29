"""Frozen probe: BADAS-Open's head trained on cached V-JEPA2 tokens (Phase 3).

Batches are balanced 1:1, with the negative half split equally between the clip's own pre-alert
windows (same source and style as the positives) and each normal-cooking source, so the probe can't
win by telling YouTube from Charades. Early stopping picks the step with the best hard AP on the
validation fold.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from ks.models.vjepa_anticipator import ProbeHead

TEMPERATURE = 2.0  # BADAS-Open trains and predicts on logits / 2


def fold_roles(folds, test_fold):
    """Fold -> train / val / test for one cross-validation round; the next fold validates."""
    val = (test_fold + 1) % folds
    return {f: "test" if f == test_fold else "val" if f == val else "train" for f in range(folds)}


def negative_stratum(source, video_label):
    return "oops-prealert" if video_label == 1 else source


def balance_weights(y, strata):
    """Per-row sampling weights summing to 1: half on positives, the other half split equally between
    the negative strata, uniform within each."""
    y, strata = np.asarray(y), np.asarray(strata)
    w = np.zeros(len(y))
    w[y == 1] = 0.5 / (y == 1).sum()
    groups = np.unique(strata[y == 0])
    for s in groups:
        rows = (y == 0) & (strata == s)
        w[rows] = 0.5 / len(groups) / rows.sum()
    return w


def sample_batch(weights, batch, rng):
    return rng.choice(len(weights), size=batch, p=weights)


def _autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")


@torch.no_grad()
def score(model, x, batch=256, device="cuda"):
    """P(danger) for every row of `x` (tokens as a torch tensor, numpy array or memmap)."""
    was_training = model.training
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        xb = x[i:i + batch]
        xb = (xb if isinstance(xb, torch.Tensor) else torch.from_numpy(np.asarray(xb))).to(device).float()
        with _autocast(device):
            logits = model(xb).float()
        out.append(torch.softmax(logits / TEMPERATURE, dim=1)[:, 1].cpu())
    model.train(was_training)
    return torch.cat(out).numpy()


def train_probe(x_train, y_train, strata, x_val, y_val, seed, steps=2000, batch=64, lr=1e-4, weight_decay=1e-2,
                eval_every=100, patience=5, hidden_size=1024, num_heads=8, head_hidden=768, device="cuda"):
    """Train a ProbeHead; returns (model at its best validation hard AP, [(step, val AP), ...])."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ProbeHead(hidden_size, num_heads, head_hidden).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    x_train = x_train.to(device)
    y_t = torch.as_tensor(np.array(y_train), dtype=torch.long, device=device)
    weights = balance_weights(y_train, strata)
    history, best, best_ap, stale = [], None, -1.0, 0
    for step in range(1, steps + 1):
        idx = torch.as_tensor(sample_batch(weights, batch, rng), device=device)
        with _autocast(device):
            loss = F.cross_entropy(model(x_train[idx].float()).float() / TEMPERATURE, y_t[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # as BADAS-Open
        opt.step()
        if step % eval_every == 0 or step == steps:
            ap = float(average_precision_score(y_val, score(model, x_val, device=device)))
            history.append((step, ap))
            if ap > best_ap:
                best_ap, best, stale = ap, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
                if stale >= patience:
                    break
    model.load_state_dict(best)
    return model.eval(), history
