"""Train and evaluate shortcut probe models.

Loads pre-extracted per-tnp feature npz files for train / val / test and
fits three models:

  1. LogisticRegression (linear ceiling)
  2. Small MLP (PyTorch, 128 -> 64 -> 1)
  3. Best per-feature univariate logistic regression (which single
     scalar goes farthest by itself?)

Reports Tnp AUROC + AUPRC + per-violation-profile AUROC on both val and
test. Also prints the top-15 features by |logistic weight| after
standardization.

Interpretation:
  - If any model approaches V1's ~1.0 AUPRC using these simple features,
    the synthetic task is trivially shortcut-solvable and V1's fancy
    architecture is overengineered for THIS dataset (positive result of
    the interpretability probe).
  - If even the best simple model plateaus below V1, we have evidence
    that V1's structure/alignment fusion is doing genuine work.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training.metrics import _auprc, _auroc, stratified_auroc


def _load(path):
    d = np.load(path, allow_pickle=True)
    return {
        "X": d["X"].astype(np.float32),
        "y": d["y"].astype(bool),
        "tnp_ids": list(d["tnp_ids"]),
        "groups": list(d["groups"]),
        "feature_names": list(d["feature_names"]),
    }


EASY_PROFILES = ("level1_marginal_matched",)


def _tnp_metrics(scores, labels, groups=None):
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    out = {
        "auroc": _auroc(scores, labels),
        "auprc": _auprc(scores, labels),
        "n_pos": int(labels.sum()),
        "n_neg": int((~labels).sum()),
    }
    if groups is not None:
        out.update(stratified_auroc(scores, labels, groups))
        # Also report "hard-only" metrics that EXCLUDE the easy warm-up
        # negative profile(s). Keeps all positives + only hard-negatives.
        g_arr = np.asarray(groups)
        hard_mask = labels | np.array([g not in EASY_PROFILES for g in g_arr])
        if hard_mask.any() and (~labels[hard_mask]).any() and labels[hard_mask].any():
            out["auroc_hard_only"] = _auroc(scores[hard_mask], labels[hard_mask])
            out["auprc_hard_only"] = _auprc(scores[hard_mask], labels[hard_mask])
            out["n_hard_neg"] = int((~labels[hard_mask]).sum())
        else:
            out["auroc_hard_only"] = float("nan")
            out["auprc_hard_only"] = float("nan")
            out["n_hard_neg"] = 0
    return out


def _fmt(m):
    strat = " ".join(
        f"{k[6:-1][:10]}={m[k]:.4f}"
        for k in sorted(m) if k.startswith("auroc[")
    )
    hard = ""
    if "auroc_hard_only" in m:
        hard = (f"  |  HARD-ONLY (excl level1): AUROC={m['auroc_hard_only']:.4f} "
                f"AUPRC={m['auprc_hard_only']:.4f} (n_hard_neg={m['n_hard_neg']})")
    return f"AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  {strat}{hard}"


def train_logistic(X_tr, y_tr, X_val, X_te, C=1.0):
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_tr)
    Xs_val = scaler.transform(X_val)
    Xs_te = scaler.transform(X_te)
    clf = LogisticRegression(max_iter=5000, C=C, class_weight=None)
    clf.fit(Xs_tr, y_tr)
    return {
        "train_scores": clf.predict_proba(Xs_tr)[:, 1],
        "val_scores": clf.predict_proba(Xs_val)[:, 1],
        "test_scores": clf.predict_proba(Xs_te)[:, 1],
        "weights": clf.coef_[0],
        "scaler": scaler,
    }


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64)):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(0.1)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_tr, y_tr, X_val, y_val, X_te, epochs=100, batch_size=64, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(X_tr)
    Xs_tr = scaler.transform(X_tr).astype(np.float32)
    Xs_val = scaler.transform(X_val).astype(np.float32)
    Xs_te = scaler.transform(X_te).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(Xs_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    x_tr_t = torch.tensor(Xs_tr).to(device)
    y_tr_t = torch.tensor(y_tr.astype(np.float32)).to(device)
    x_val_t = torch.tensor(Xs_val).to(device)
    x_te_t = torch.tensor(Xs_te).to(device)

    best_val = -1.0
    best_state = None
    for ep in range(epochs):
        model.train()
        # simple mini-batch loop
        idx = torch.randperm(len(x_tr_t), device=device)
        for i in range(0, len(idx), batch_size):
            b = idx[i:i + batch_size]
            logit = model(x_tr_t[b])
            loss = nn.functional.binary_cross_entropy_with_logits(logit, y_tr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_score = torch.sigmoid(model(x_val_t)).cpu().numpy()
        val_auprc = _auprc(val_score, y_val)
        if val_auprc > best_val:
            best_val = val_auprc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return {
            "val_scores": torch.sigmoid(model(x_val_t)).cpu().numpy(),
            "test_scores": torch.sigmoid(model(x_te_t)).cpu().numpy(),
            "train_scores": torch.sigmoid(model(x_tr_t)).cpu().numpy(),
        }


def univariate_probes(X_tr, y_tr, X_val, y_val, X_te, y_te, feature_names):
    """Fit a single-variable logistic regression per feature, report test AUROC."""
    results = []
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_tr)
    Xs_te = scaler.transform(X_te)
    for i, name in enumerate(feature_names):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xs_tr[:, i:i + 1], y_tr)
        s = clf.predict_proba(Xs_te[:, i:i + 1])[:, 1]
        results.append((name, _auroc(s, y_te), _auprc(s, y_te)))
    results.sort(key=lambda r: -r[1])
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--val", required=True, type=Path)
    p.add_argument("--test", required=True, type=Path)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    tr = _load(args.train); va = _load(args.val); te = _load(args.test)
    print(f"[data] train {tr['X'].shape}  val {va['X'].shape}  test {te['X'].shape}")
    print(f"       train pos={int(tr['y'].sum())}  val pos={int(va['y'].sum())}  test pos={int(te['y'].sum())}")
    feature_names = tr["feature_names"]
    assert feature_names == va["feature_names"] == te["feature_names"]

    print()
    print("=== 1) Logistic regression (linear ceiling) ===")
    t0 = time.time()
    log_out = train_logistic(tr["X"], tr["y"], va["X"], te["X"])
    print(f"    ({time.time()-t0:.1f}s)")
    m_val = _tnp_metrics(log_out["val_scores"], va["y"], va["groups"])
    m_te = _tnp_metrics(log_out["test_scores"], te["y"], te["groups"])
    print(f"  val:  {_fmt(m_val)}")
    print(f"  test: {_fmt(m_te)}")
    print()
    top_w = sorted(
        zip(feature_names, log_out["weights"]),
        key=lambda p: -abs(p[1]),
    )[:15]
    print("  Top 15 features by |weight| (standardized):")
    for name, w in top_w:
        print(f"    {name:<45s}  {w:+.3f}")

    print()
    print("=== 2) Small MLP (128 -> 64 -> 1) ===")
    t0 = time.time()
    mlp_out = train_mlp(tr["X"], tr["y"], va["X"], va["y"], te["X"], epochs=200)
    print(f"    ({time.time()-t0:.1f}s)")
    m_val = _tnp_metrics(mlp_out["val_scores"], va["y"], va["groups"])
    m_te = _tnp_metrics(mlp_out["test_scores"], te["y"], te["groups"])
    print(f"  val:  {_fmt(m_val)}")
    print(f"  test: {_fmt(m_te)}")

    print()
    print("=== 3) Univariate probes: single-feature logistic on test ===")
    uni = univariate_probes(
        tr["X"], tr["y"], va["X"], va["y"], te["X"], te["y"], feature_names
    )
    print("  Top 15 single features by test AUROC:")
    for name, auroc, auprc in uni[:15]:
        print(f"    {name:<45s}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({
                "logistic_val": m_val,
                "logistic_test": _tnp_metrics(log_out["test_scores"], te["y"], te["groups"]),
                "mlp_val": _tnp_metrics(mlp_out["val_scores"], va["y"], va["groups"]),
                "mlp_test": _tnp_metrics(mlp_out["test_scores"], te["y"], te["groups"]),
                "univariate_top": [(n, a, ap) for (n, a, ap) in uni[:20]],
                "logistic_weights": {n: float(w) for n, w in zip(feature_names, log_out["weights"])},
            }, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
