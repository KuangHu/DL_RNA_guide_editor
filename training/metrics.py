"""Evaluation metrics for V1.

All functions accept numpy arrays (or torch tensors that get .cpu().numpy()'d).
Kept dependency-free (no sklearn) so the training loop can compute them
without extra installs.

Metrics:
  tnp_metrics(scores, labels)                 - AUROC + AUPRC
  candidate_recall(scores, targets, ks)       - Recall@k on true candidate slot
  nc_selection_accuracy(nc_attn, true_slot)   - argmax of NC attention == true
  stratified_auroc(scores, labels, groups)    - per-group AUROC
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney-U based AUROC. Handles ties by averaging ranks.
    Returns NaN if only one class is present.
    """
    scores = scores.astype(np.float64)
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    # Rank with tie-averaging.
    sorted_scores = scores[order]
    i = 0
    N = len(scores)
    while i < N:
        j = i
        while j + 1 < N and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    sum_pos_ranks = ranks[labels].sum()
    u = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (area under precision-recall curve).
    Uses the standard sklearn-style AP definition: sum over each recall step of
    P(k) * dR(k), which is equivalent to sum_i (R_i - R_{i-1}) * P_i.
    Returns NaN if no positives.
    """
    scores = scores.astype(np.float64)
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    denom = (tp + fp).clip(min=1)
    precision = tp / denom
    recall = tp / n_pos
    # AP = sum over samples where label=1 of precision at that rank / n_pos.
    return float((precision * y).sum() / n_pos)


def tnp_metrics(scores, labels) -> dict:
    """Return {'auroc': ..., 'auprc': ..., 'n_pos': int, 'n_neg': int}."""
    s = _to_numpy(scores).reshape(-1)
    y = _to_numpy(labels).astype(bool).reshape(-1)
    return {
        "auroc": _auroc(s, y),
        "auprc": _auprc(s, y),
        "n_pos": int(y.sum()),
        "n_neg": int((~y).sum()),
    }


def candidate_recall(
    cand_scores,             # (N_pos, K) — per-candidate scores at the ACTIVE NC slot,
                              #   already gathered by the caller
    true_slot: Sequence[int],  # (N_pos,)  int; -1 to exclude
    ks: Sequence[int] = (1, 5, 10),
) -> dict:
    """Fraction of positive sites whose true candidate ranks in the top-k
    by score. Sites with true_slot == -1 are dropped (aux label unknown)."""
    s = _to_numpy(cand_scores)
    t = np.asarray(true_slot, dtype=np.int64)
    keep = t >= 0
    if not keep.any():
        return {f"recall@{k}": float("nan") for k in ks} | {"n": 0}
    s = s[keep]
    t = t[keep]
    ranks = np.argsort(-s, axis=-1, kind="stable")  # (N, K), best first
    # rank position of each true slot
    rank_of_true = np.empty(len(t), dtype=np.int64)
    for i in range(len(t)):
        pos = np.where(ranks[i] == t[i])[0]
        rank_of_true[i] = int(pos[0]) if pos.size else -1
    out = {}
    for k in ks:
        out[f"recall@{k}"] = float((rank_of_true < k).mean())
    out["n"] = int(keep.sum())
    return out


def nc_selection_accuracy(
    nc_attn,                # (N_pos, N_nc)
    true_active_nc,         # (N_pos,) int; -1 to exclude
) -> dict:
    """Fraction of positive sites whose argmax NC attention == true active
    NC slot. Excludes sites with true_active_nc == -1."""
    a = _to_numpy(nc_attn)
    t = np.asarray(true_active_nc, dtype=np.int64)
    keep = t >= 0
    if not keep.any():
        return {"nc_top1": float("nan"), "n": 0}
    a = a[keep]
    t = t[keep]
    pred = np.argmax(a, axis=-1)
    return {"nc_top1": float((pred == t).mean()), "n": int(keep.sum())}


def stratified_auroc(
    scores, labels, groups,
    positive_group_name: str = "positive",
) -> dict:
    """Compute AUROC per group. Each group's AUROC is computed by pairing
    positives against negatives OF THAT GROUP ONLY.

    scores: (N,)  logits or probabilities
    labels: (N,)  bool
    groups: (N,)  str — for positives use `positive_group_name`; for
                        negatives use their violation_profile
    """
    s = _to_numpy(scores).reshape(-1)
    y = _to_numpy(labels).astype(bool).reshape(-1)
    g = np.asarray(groups)

    out: dict[str, float] = {}
    unique_groups = sorted(set(g[~y].tolist()))
    for grp in unique_groups:
        mask = y | (g == grp)  # keep all positives + negatives of this group
        out[f"auroc[{grp}]"] = _auroc(s[mask], y[mask])
    return out
