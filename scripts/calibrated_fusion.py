"""48C1h-B: post-hoc calibrated soft-min / min fusion of pair + geom aux heads.

Loads a two-branch multi-branch checkpoint (48C1g/h-A style), runs val inference,
splits val by parent_tnp_id into (calib, report) halves, fits Platt (affine)
calibration on the calib half using property-specific labels, then evaluates
three fusion rules on the report half:

  1. RAW_AND:      logit = log σ(s_p) + log σ(s_g)          (48C1f original)
  2. HARD_MIN:     p_final = min(σ(z'_p), σ(z'_g))           after calibration
  3. SOFT_MIN:     z_final = -logsumexp([-z'_p, -z'_g] / τ)  (τ = 1.0)

For each rule, reports:
  - Per-profile AUROC (POS vs each profile)
  - Paired Δ statistics (median, MAD, P(Δ>0)) matched by parent tnp
  - Overall AUROC / AUPRC

Success criteria (per user 2026-08-28):
  - shuffle ≥ 0.92, length ≥ 0.92
  - position 0.535 → 0.65-0.72
  - No learned fusion parameters
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

sys.path.insert(0, "/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer")

from model.v1 import V1Config, V1Model
from preprocess.site import StructureCache
from preprocess.tnp_dataset import (
    TnpGroupedDataset, collate_tnp_batch, make_torch_tnp_dataset,
)
from torch.utils.data import DataLoader


PROFILE_Y_PAIR = {
    "positive":                 1,
    "paired_shuffle_v42":       0,
    "wrong_length_v42":         0,
    "wrong_orientation_v42":    1,
    "wrong_position_v42":       1,
    "wrong_structure_role_v42": 1,
}
PROFILE_Y_GEOM = {
    "positive":                 1,
    "paired_shuffle_v42":       1,
    "wrong_length_v42":         1,
    "wrong_orientation_v42":    0,
    "wrong_position_v42":       0,
    "wrong_structure_role_v42": 1,
}
NEG_SUFFIXES = {
    "paired_shuffle_v42":       "__neg_paired_shuffle_v42",
    "wrong_length_v42":         "__neg_wrong_length_v42",
    "wrong_orientation_v42":    "__neg_wrong_orientation_v42",
    "wrong_position_v42":       "__neg_wrong_position_v42",
    "wrong_structure_role_v42": "__neg_wrong_structure_role_v42",
}


def _parent_tnp(tnp_id: str) -> str:
    for suf in NEG_SUFFIXES.values():
        if tnp_id.endswith(suf):
            return tnp_id[: -len(suf)]
    return tnp_id  # already a positive tnp id


def _profile_of(tnp_id: str) -> str:
    for prof, suf in NEG_SUFFIXES.items():
        if tnp_id.endswith(suf):
            return prof
    return "positive"


def _auroc(scores, labels):
    from sklearn.metrics import roc_auc_score
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


def _auprc(scores, labels):
    from sklearn.metrics import average_precision_score
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return average_precision_score(labels, scores)


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit y = σ(a·x + b) via logistic regression. Returns (a, b)."""
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(labels)) < 2:
        return 1.0, 0.0
    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
    clf.fit(scores.reshape(-1, 1), labels)
    a = float(clf.coef_.flatten()[0])
    b = float(clf.intercept_.flatten()[0])
    return a, b


def paired_delta_stats(scores_by_tnp, profile_of_tnp):
    """For each profile, compute paired Δ statistics matched by parent tnp."""
    parents = defaultdict(dict)
    for tnp, s in scores_by_tnp.items():
        p = profile_of_tnp[tnp]
        parents[_parent_tnp(tnp)][p] = s
    reports = {}
    for prof in NEG_SUFFIXES:
        deltas = []
        for _, mp in parents.items():
            if "positive" in mp and prof in mp:
                deltas.append(mp["positive"] - mp[prof])
        if not deltas:
            continue
        a = np.asarray(deltas, dtype=np.float32)
        reports[prof] = {
            "n": int(len(a)),
            "median": float(np.median(a)),
            "MAD": float(np.median(np.abs(a - np.median(a)))),
            "std": float(a.std()),
            "p_gt_0": float((a > 0).mean()),
            "q10": float(np.quantile(a, 0.10)),
            "q90": float(np.quantile(a, 0.90)),
        }
    return reports


def per_profile_auroc(scores_by_tnp, profile_of_tnp):
    """POS vs each profile AUROC on the given score dict."""
    from sklearn.metrics import roc_auc_score
    pos_scores = np.asarray([s for t, s in scores_by_tnp.items()
                              if profile_of_tnp[t] == "positive"])
    out = {}
    for prof in NEG_SUFFIXES:
        neg_scores = np.asarray([s for t, s in scores_by_tnp.items()
                                   if profile_of_tnp[t] == prof])
        if len(neg_scores) == 0:
            continue
        combined = np.concatenate([pos_scores, neg_scores])
        labels = np.concatenate([np.ones(len(pos_scores), int),
                                   np.zeros(len(neg_scores), int)])
        out[prof] = {
            "auroc": float(_auroc(combined, labels)),
            "auprc": float(_auprc(combined, labels)),
            "n_pos": int(len(pos_scores)),
            "n_neg": int(len(neg_scores)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-frac", type=float, default=0.5,
                    help="Fraction of parent-tnps to use for calibration; rest is report.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tnp-batch", type=int, default=8)
    ap.add_argument("--sites", type=int, default=50)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--soft-min-tau", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- load ckpt ----------
    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    use_additive = "alpha_pair" in state or "alpha_geom" in state
    normalize_aux = any(k.startswith("bn_pair_aux.") or k.startswith("bn_geom_aux.")
                          for k in state.keys())
    v1_cfg = V1Config(
        use_multi_branch=True,
        use_explicit_geom_stats=True,
        use_additive_fusion=use_additive,
        normalize_aux_logits=normalize_aux,
    )
    model = V1Model(v1_cfg).to(device)
    r = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(r.missing_keys)}  unexpected={len(r.unexpected_keys)}", flush=True)
    if r.unexpected_keys:
        print(f"  unexpected: {r.unexpected_keys[:5]}", flush=True)
    model.eval()

    # ---------- inference ----------
    cache = StructureCache(args.cache)
    ds = TnpGroupedDataset(args.jsonl, cache, site_subsample_size=args.sites)
    print(f"[data] tnps={len(ds)}", flush=True)
    dl = DataLoader(
        make_torch_tnp_dataset(ds),
        batch_size=args.tnp_batch, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda items: collate_tnp_batch(items, to_torch=True),
        pin_memory=(device.type == "cuda"),
    )
    all_tnp_ids: list[str] = []
    all_s_pair: list[float] = []
    all_s_geom: list[float] = []
    n_batches = 0
    with torch.no_grad():
        for batch in dl:
            batch_gpu = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                          for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                out = model(
                    batch_gpu["candidate_patches"],
                    batch_gpu["candidate_features"],
                    batch_gpu["candidate_mask"],
                    batch_gpu["nc_region_mask"],
                )
            all_tnp_ids.extend(list(batch["tnp_id"]))
            all_s_pair.extend(out["s_pair_aux"].detach().float().cpu().numpy().tolist())
            all_s_geom.extend(out["s_geom_aux"].detach().float().cpu().numpy().tolist())
            n_batches += 1
            if n_batches % 50 == 0:
                print(f"  [batch {n_batches}] done", flush=True)

    tnp_ids = np.asarray(all_tnp_ids)
    s_pair = np.asarray(all_s_pair, dtype=np.float32)
    s_geom = np.asarray(all_s_geom, dtype=np.float32)
    profile_of_tnp = {t: _profile_of(t) for t in tnp_ids}

    # ---------- split calib/report by parent_tnp ----------
    rng = np.random.default_rng(args.seed)
    parents = sorted({_parent_tnp(t) for t in tnp_ids})
    rng.shuffle(parents)
    n_cal = int(round(len(parents) * args.calib_frac))
    calib_parents = set(parents[:n_cal])
    report_parents = set(parents[n_cal:])
    print(f"[split] parents: calib={len(calib_parents)}  report={len(report_parents)}", flush=True)

    def _mask_for(parents_set):
        return np.asarray([_parent_tnp(t) in parents_set for t in tnp_ids])

    m_cal = _mask_for(calib_parents)
    m_rep = _mask_for(report_parents)

    # ---------- fit Platt calibration ----------
    y_pair_cal = np.asarray([PROFILE_Y_PAIR[profile_of_tnp[t]] for t in tnp_ids[m_cal]], dtype=np.int32)
    y_geom_cal = np.asarray([PROFILE_Y_GEOM[profile_of_tnp[t]] for t in tnp_ids[m_cal]], dtype=np.int32)
    a_p, b_p = fit_platt(s_pair[m_cal], y_pair_cal)
    a_g, b_g = fit_platt(s_geom[m_cal], y_geom_cal)
    print(f"[calib] pair: a={a_p:+.3f}  b={b_p:+.3f}", flush=True)
    print(f"[calib] geom: a={a_g:+.3f}  b={b_g:+.3f}", flush=True)

    # ---------- compute fusion rules on REPORT half ----------
    rep_tnps = tnp_ids[m_rep]
    rep_s_pair = s_pair[m_rep]
    rep_s_geom = s_geom[m_rep]

    z_pair = a_p * rep_s_pair + b_p
    z_geom = a_g * rep_s_geom + b_g

    # 1) raw AND (48C1f baseline): log σ(s_p) + log σ(s_g)
    def _logsig(x):
        # log(sigmoid(x)) = -softplus(-x); numerically stable
        return -np.logaddexp(0.0, -x)
    raw_and = _logsig(rep_s_pair) + _logsig(rep_s_geom)

    # 2) calibrated hard-min in probability space
    p_pair = 1.0 / (1.0 + np.exp(-z_pair))
    p_geom = 1.0 / (1.0 + np.exp(-z_geom))
    hard_min = np.minimum(p_pair, p_geom)

    # 3) calibrated soft-min in logit space:
    #    z_final = -τ · log(exp(-z'_p / τ) + exp(-z'_g / τ))
    tau = args.soft_min_tau
    z_stack = np.stack([-z_pair / tau, -z_geom / tau], axis=1)  # (N, 2)
    soft_min = -tau * np.logaddexp(z_stack[:, 0], z_stack[:, 1])

    # ---------- eval each fusion rule ----------
    reports = {}
    for name, scores in [("RAW_AND", raw_and), ("HARD_MIN", hard_min), ("SOFT_MIN", soft_min)]:
        scores_by_tnp = {t: float(s) for t, s in zip(rep_tnps, scores)}
        profile_by_tnp = {t: profile_of_tnp[t] for t in rep_tnps}
        per_prof = per_profile_auroc(scores_by_tnp, profile_by_tnp)
        paired = paired_delta_stats(scores_by_tnp, profile_by_tnp)
        # Overall AUROC/AUPRC
        pos_mask = np.asarray([profile_by_tnp[t] == "positive" for t in rep_tnps])
        labels = pos_mask.astype(int)
        overall_auroc = _auroc(scores, labels)
        overall_auprc = _auprc(scores, labels)
        reports[name] = {
            "overall_auroc": float(overall_auroc),
            "overall_auprc": float(overall_auprc),
            "n_pos": int(pos_mask.sum()),
            "n_neg": int((~pos_mask).sum()),
            "per_profile": per_prof,
            "paired_delta": paired,
        }
        print(f"\n=== fusion rule: {name} ===", flush=True)
        print(f"  overall AUROC={overall_auroc:.4f}  AUPRC={overall_auprc:.4f}", flush=True)
        print(f"  per-profile AUROC:")
        for p, m in per_prof.items():
            print(f"    {p:<28} auroc={m['auroc']:.4f}  auprc={m['auprc']:.4f}", flush=True)
        print(f"  paired Δ:")
        for p, s in paired.items():
            print(f"    {p:<28} median={s['median']:+.3f}  MAD={s['MAD']:.3f}  P(Δ>0)={s['p_gt_0']:.3f}", flush=True)

    # ---------- also emit calibrated-only per-head diagnostics ----------
    # p_pair alone (calibrated): AUROC per profile against pair-property labels
    def _head_auroc(head_scores):
        s_by_t = {t: float(v) for t, v in zip(rep_tnps, head_scores)}
        return per_profile_auroc(s_by_t, {t: profile_of_tnp[t] for t in rep_tnps})
    reports["p_pair_calibrated_per_profile"] = _head_auroc(z_pair)
    reports["p_geom_calibrated_per_profile"] = _head_auroc(z_geom)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "calib_frac": args.calib_frac,
            "n_calib_parents": len(calib_parents),
            "n_report_parents": len(report_parents),
            "platt": {"pair": {"a": a_p, "b": b_p}, "geom": {"a": a_g, "b": b_g}},
            "soft_min_tau": tau,
            "reports": reports,
        }, f, indent=2)
    print(f"\n[out] {out_path}", flush=True)


if __name__ == "__main__":
    main()
