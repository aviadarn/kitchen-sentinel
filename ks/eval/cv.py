"""Phase 3 pilot: k-fold cross-validation of the frozen probe and its baselines on cached features.

Round k tests fold k, validates on fold k+1 (early stopping, logistic-regression C, the alarm
threshold tau) and trains on the rest. Every window is scored once as test; metrics pool those
out-of-fold scores, and each fold's tau applies to its own test videos.

    uv run python -m ks.eval.cv --features features --out reports/pilot
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ks.eval.metrics import expected_calibration_error, false_alarms, group_bootstrap_ci, time_to_accident
from ks.eval.protocol import accident_curves, choose_tau, easy_set, hard_set, risk_curves
from ks.train import balance_weights, fold_roles, negative_stratum, score, train_probe

EASY_OFFSETS = (0.25, 0.5, 1.0)  # labeled warnings are short (median 0.43 s), so BADAS's 0.5/1.0/1.5 shift down
MAX_FA_PER_HOUR = 1.0
EARLY_GAP = 0.5  # same margin as the training labels' pre-alert gap
METHODS = ["prior", "badas", "siglip_lr", "vjepa_lr", "probe"]


def train_rows(index, max_per_stratum, rng):
    """Labeled windows of the train folds: every positive and pre-alert window, and at most
    `max_per_stratum` windows from each normal-cooking source (they are many and redundant)."""
    rows = []
    train = index[(index.role == "train") & index.window_label.notna()]
    strata = [negative_stratum(s, v) for s, v in zip(train.source, train.video_label)]
    for stratum, part in train.groupby(train.window_label.astype(int).astype(str) + "/" + pd.Series(strata, train.index)):
        r = part.index.to_numpy()
        if stratum.startswith("0/") and stratum != "0/oops-prealert" and len(r) > max_per_stratum:
            r = rng.choice(r, max_per_stratum, replace=False)
        rows.append(r)
    return np.sort(np.concatenate(rows))


def _lr(x_tr, y_tr, w_tr, x_val, y_val, grid):
    best = None
    for c in grid:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
        model.fit(x_tr, y_tr, logisticregression__sample_weight=w_tr * len(w_tr))
        ap = average_precision_score(y_val, model.predict_proba(x_val)[:, 1])
        if best is None or ap > best[0]:
            best = (ap, model)
    return best[1]


def run_cv(index, feats, folds=5, seeds=(0, 1, 2), max_per_stratum=3000, device="cuda", probe_kw=None,
           lr_grid=(0.001, 0.01, 0.1, 1.0), log=print):
    """Scores of every method on the val and test windows of every round: columns row, method, seed,
    fold (the round's test fold), role, score."""
    index = index.copy()
    index["stratum"] = [negative_stratum(s, v) for s, v in zip(index.source, index.video_label)]
    out = []
    for k in range(folds):
        index["role"] = index.fold.map(fold_roles(folds, k))
        tr = train_rows(index, max_per_stratum, np.random.default_rng(k))
        y_tr = index.window_label[tr].astype(int).to_numpy()
        w_tr = balance_weights(y_tr, index.stratum[tr].to_numpy())
        ev = index.index[index.role.isin(["val", "test"])].to_numpy()
        val = index.index[(index.role == "val") & (index.video_label == 1) & index.window_label.notna()].to_numpy()
        y_val = index.window_label[val].astype(int).to_numpy()
        log(f"round {k}: {len(tr)} train windows ({y_tr.sum()} positive), {len(val)} val hard-set windows")

        def add(method, seed, s):
            out.append(pd.DataFrame(dict(row=ev, method=method, seed=seed, fold=k, role=index.role[ev].to_numpy(),
                                         score=s)))

        add("prior", 0, np.full(len(ev), 0.5))
        add("badas", 0, np.asarray(feats["badas"][ev], dtype=float))
        for name, key in (("siglip_lr", "siglip"), ("vjepa_lr", "vjepa_mean")):
            x = feats[key]
            model = _lr(np.asarray(x[tr], dtype=np.float32), y_tr, w_tr, np.asarray(x[val], dtype=np.float32), y_val,
                        lr_grid)
            add(name, 0, model.predict_proba(np.asarray(x[ev], dtype=np.float32))[:, 1])

        x_tr = torch.from_numpy(np.asarray(feats["vjepa"][tr])).to(device)
        x_val = torch.from_numpy(np.asarray(feats["vjepa"][val])).to(device)
        models = []
        for seed in seeds:
            model, history = train_probe(x_tr, y_tr, index.stratum[tr].to_numpy(), x_val, y_val, seed=seed,
                                         device=device, **(probe_kw or {}))
            log(f"  probe seed {seed}: best val AP {max(ap for _, ap in history):.3f} at step "
                f"{max(history, key=lambda h: h[1])[0]}")
            models.append(model)
        del x_tr, x_val
        chunks = [[] for _ in seeds]
        for i in range(0, len(ev), 1024):
            xb = torch.from_numpy(np.asarray(feats["vjepa"][ev[i:i + 1024]])).to(device)
            for j, model in enumerate(models):
                chunks[j].append(score(model, xb, device=device))
        for seed, parts in zip(seeds, chunks):
            add("probe", seed, np.concatenate(parts))
    return pd.concat(out, ignore_index=True)


def _attach(index, scores):
    return index.loc[scores.row].assign(score=scores.score.to_numpy(), fold=scores.fold.to_numpy())


def summarize(preds, taus):
    """Metrics of one method's out-of-fold test predictions; `taus` maps fold -> its alarm threshold."""
    y, s, _ = hard_set(preds)
    m = {"hard_ap": float(average_precision_score(y, s)), "hard_auc": float(roc_auc_score(y, s)),
         "ece": expected_calibration_error(y, s)}
    for offset in EASY_OFFSETS:
        ye, se, _ = easy_set(preds, offset)
        m[f"easy_ap@{offset}"] = float(average_precision_score(ye, se))
    ttas, events, early, alarms, hours = [], 0, 0, 0, 0.0
    for fold, part in preds.groupby("fold"):
        tau = taus[fold]
        for times, probs, alert, event in accident_curves(part):
            if not times:
                continue  # accident before the first full window: nothing can see it coming
            events += 1
            tta = time_to_accident(times, probs, tau, event)
            if tta is not None:
                ttas.append(tta)
                # An alarm before the danger is visible says "this is a fail video", not "it's coming".
                early += event - tta < alert - EARLY_GAP
        for times, probs, duration in risk_curves(part, positive=False):
            alarms += false_alarms(times, probs, tau)
            hours += duration / 3600
    m.update(recall=len(ttas) / events, on_time_recall=(len(ttas) - early) / events, early=early / events,
             mtta=float(np.mean(ttas)) if ttas else float("nan"), fa_per_h=alarms / hours, events=events)
    return m


def report(scores, index, out_dir, n_boot=1000):
    """Pool each method's test scores (probe: mean of per-seed metrics), write results.md + metrics.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results, hard = {}, {}
    for method in [m for m in METHODS if m in set(scores.method)]:
        per_seed, hard_rows = [], []
        for seed, s in scores[scores.method == method].groupby("seed"):
            taus = {k: choose_tau(risk_curves(_attach(index, v), positive=False), MAX_FA_PER_HOUR)
                    for k, v in s[s.role == "val"].groupby("fold")}
            preds = _attach(index, s[s.role == "test"])
            per_seed.append(summarize(preds, taus) | {"taus": taus})
            hard_rows.append(hard_set(preds))
        results[method] = {k: float(np.mean([r[k] for r in per_seed])) for k in per_seed[0] if k != "taus"}
        results[method]["seeds"] = len(per_seed)
        results[method]["taus"] = {str(k): v for k, v in per_seed[0]["taus"].items()}
        hard[method] = hard_rows
        y, _, g = hard_rows[0]
        results[method]["hard_ap_ci"] = group_bootstrap_ci(
            lambda i: float(np.mean([average_precision_score(y[i], s[i]) if 0 < y[i].sum() < len(i) else np.nan
                                     for _, s, _ in hard_rows])), g, n_boot=n_boot)
    if "probe" in hard and "siglip_lr" in hard:
        y, s_sig, g = hard["siglip_lr"][0]

        def delta(i):
            if not 0 < y[i].sum() < len(i):
                return np.nan
            probe = np.mean([average_precision_score(y[i], s[i]) for _, s, _ in hard["probe"]])
            return probe - average_precision_score(y[i], s_sig[i])

        results["probe"]["delta_vs_siglip"] = float(delta(np.arange(len(y))))
        results["probe"]["delta_vs_siglip_ci"] = group_bootstrap_ci(delta, g, n_boot=n_boot)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    (out / "results.md").write_text(_markdown(results, index))
    return results


def _markdown(results, index):
    pos = index[index.video_label == 1]
    neg_hours = index[index.video_label == 0].groupby("video_id").duration.first().sum() / 3600
    lines = [
        f"{pos.video_id.nunique()} positive clips, {results[next(iter(results))]['events']:.0f} accidents with a "
        f"visible lead-up; {neg_hours:.1f} h of normal cooking. Out-of-fold over {index.fold.nunique()} folds; "
        f"tau per fold for <= {MAX_FA_PER_HOUR:g} false alarm/h on its validation fold. Of the accidents, "
        f"'on time' = first alarm no earlier than {EARLY_GAP:g} s before the labeled alert; 'early' = before that "
        f"(the clip looks like a fail video, but no danger is visible yet).", "",
        "| method | hard AP [95% CI] | hard AUC | " + " | ".join(f"easy AP @{o}s" for o in EASY_OFFSETS)
        + " | recall @tau | on time | early | mTTA (s) | FA/h @tau | ECE |",
        "|---" * (10 + len(EASY_OFFSETS)) + "|",
    ]
    for method, r in results.items():
        lo, hi = r["hard_ap_ci"]
        lines.append(f"| {method}{' (%d seeds)' % r['seeds'] if r['seeds'] > 1 else ''} | {r['hard_ap']:.3f} "
                     f"[{lo:.3f}, {hi:.3f}] | {r['hard_auc']:.3f} | "
                     + " | ".join(f"{r[f'easy_ap@{o}']:.3f}" for o in EASY_OFFSETS)
                     + f" | {r['recall']:.2f} | {r['on_time_recall']:.2f} | {r['early']:.2f} | {r['mtta']:.2f} | "
                       f"{r['fa_per_h']:.2f} | {r['ece']:.3f} |")
    if "delta_vs_siglip" in results.get("probe", {}):
        lo, hi = results["probe"]["delta_vs_siglip_ci"]
        lines += ["", f"Gate (probe beats single-frame SigLIP on hard AP): delta = "
                      f"{results['probe']['delta_vs_siglip']:+.3f} [{lo:+.3f}, {hi:+.3f}]"]
    return "\n".join(lines) + "\n"


def load_features(features_dir, device="cuda"):
    d = Path(features_dir)
    feats = {name: np.load(d / f"{name}.npy", mmap_mode="r") for name in ("vjepa", "badas", "siglip")}
    mean_path = d / "vjepa_mean.npy"
    if not mean_path.exists():
        # On the GPU: numpy's single-threaded float16 conversion takes ~40 min over the 59 GB of tokens.
        v = feats["vjepa"]
        mean = np.empty((len(v), v.shape[2]), dtype=np.float32)
        for i in range(0, len(v), 2048):
            mean[i:i + 2048] = torch.from_numpy(np.asarray(v[i:i + 2048])).to(device).float().mean(1).cpu().numpy()
        np.save(mean_path, mean)
    feats["vjepa_mean"] = np.load(mean_path)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=2000)
    a = ap.parse_args()

    index = pd.read_csv(Path(a.features) / "index.csv")
    done = np.load(Path(a.features) / "done.npy")
    index = index[done]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scores = run_cv(index, load_features(a.features, device), folds=a.folds, seeds=a.seeds, device=device,
                    probe_kw=dict(steps=a.steps), log=lambda s: print(s, flush=True))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    scores.to_csv(Path(a.out) / "scores.csv", index=False)
    report(scores, index, a.out)
    print((Path(a.out) / "results.md").read_text())


if __name__ == "__main__":
    main()
