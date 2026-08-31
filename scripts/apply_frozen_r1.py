"""R1 frozen-panel eval: apply 48C1h-B synthetic Platt calibration to real data.

Loads a 48C1h-A/B multi-branch checkpoint, runs val inference to get
s_pair_aux and s_geom_aux, then applies FROZEN Platt coefficients from a
JSON file (from a prior calibrated_fusion.py run) and evaluates three
fusion rules on the entire eval set:

  1. RAW_AND      : logit = log σ(s_p) + log σ(s_g)
  2. HARD_MIN     : p     = min(σ(z'_p), σ(z'_g))  with z' from frozen Platt
  3. SOFT_MIN     : logit = -τ · logsumexp([-z'_p, -z'_g]/τ)

Reports per-profile AUROC + paired Δ (median, MAD, P(Δ>0)) for each head
and each fusion rule, matched by parent tnp via a suffix map.

Success criterion for R1: this is a FROZEN benchmark. NO tuning on this
output. R2 must apply the same script + same Platt json + same panel.
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


def paired_delta_by_suffix(scores_by_tnp: dict, neg_suffix: str) -> dict:
    """Δ = s(POS) - s(NEG) matched by parent_tnp_id (POS_tnp + neg_suffix == NEG_tnp)."""
    deltas = []
    n_matched = 0
    for tnp, s in scores_by_tnp.items():
        # Assume this is a POS: look for the paired NEG.
        neg_tnp = tnp + neg_suffix
        if neg_tnp in scores_by_tnp:
            deltas.append(s - scores_by_tnp[neg_tnp])
            n_matched += 1
    if not deltas:
        return {}
    a = np.asarray(deltas, dtype=np.float32)
    return {
        "n": int(n_matched),
        "median": float(np.median(a)),
        "MAD": float(np.median(np.abs(a - np.median(a)))),
        "std": float(a.std()),
        "p_gt_0": float((a > 0).mean()),
        "q10": float(np.quantile(a, 0.10)),
        "q90": float(np.quantile(a, 0.90)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--frozen-platt-json", required=True,
                    help="JSON from a prior calibrated_fusion.py run (48C1h-B).")
    ap.add_argument("--neg-suffix", required=True,
                    help="Suffix that maps POS tnp_id -> NEG tnp_id for paired Δ "
                         "(e.g. '__neg_durrant_shuffled').")
    ap.add_argument("--out", required=True)
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
    use_orient = any(k.startswith("orient_mlp.") or k.startswith("h_orient_aux.")
                       for k in state.keys())
    v1_cfg = V1Config(
        use_multi_branch=True,
        use_explicit_geom_stats=True,
        use_additive_fusion=use_additive,
        normalize_aux_logits=normalize_aux,
        use_orient_branch=use_orient,
    )
    model = V1Model(v1_cfg).to(device)
    r = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}", flush=True)
    model.eval()

    # ---------- load frozen Platt ----------
    platt_obj = json.loads(Path(args.frozen_platt_json).read_text())
    p_p = platt_obj["platt"]["pair"]
    p_g = platt_obj["platt"]["geom"]
    a_p, b_p = float(p_p["a"]), float(p_p["b"])
    a_g, b_g = float(p_g["a"]), float(p_g["b"])
    print(f"[frozen Platt] pair: z' = {a_p:+.3f} · s + {b_p:+.3f}", flush=True)
    print(f"[frozen Platt] geom: z' = {a_g:+.3f} · s + {b_g:+.3f}", flush=True)

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
    all_labels: list[int] = []
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
            all_labels.extend(batch["is_positive"].numpy().astype(int).tolist())
            n_batches += 1
            if n_batches % 50 == 0:
                print(f"  [batch {n_batches}] done", flush=True)

    tnp_ids = np.asarray(all_tnp_ids)
    s_pair = np.asarray(all_s_pair, dtype=np.float32)
    s_geom = np.asarray(all_s_geom, dtype=np.float32)
    labels = np.asarray(all_labels, dtype=np.int32)

    # ---------- apply frozen Platt ----------
    z_pair = a_p * s_pair + b_p
    z_geom = a_g * s_geom + b_g

    # ---------- fusion rules ----------
    def _logsig(x):
        return -np.logaddexp(0.0, -x)

    raw_and = _logsig(s_pair) + _logsig(s_geom)  # baseline (48C1f original)
    p_pair = 1.0 / (1.0 + np.exp(-z_pair))
    p_geom = 1.0 / (1.0 + np.exp(-z_geom))
    hard_min = np.minimum(p_pair, p_geom)
    tau = args.soft_min_tau
    z_stack = np.stack([-z_pair / tau, -z_geom / tau], axis=1)
    soft_min = -tau * np.logaddexp(z_stack[:, 0], z_stack[:, 1])

    # ---------- report ----------
    all_scores = {
        "raw_pair_aux":  s_pair,
        "raw_geom_aux":  s_geom,
        "calib_z_pair":  z_pair,
        "calib_z_geom":  z_geom,
        "RAW_AND":       raw_and,
        "HARD_MIN":      hard_min,
        "SOFT_MIN":      soft_min,
    }
    reports = {}
    print("\n=== R1 frozen panel (POS = is_positive=True) ===")
    print(f"  n_pos={int((labels==1).sum())}  n_neg={int((labels==0).sum())}")
    for name, scores in all_scores.items():
        au = _auroc(scores, labels)
        ap = _auprc(scores, labels)
        # Paired Δ matched by parent tnp
        scores_by_tnp = {t: float(s) for t, s in zip(tnp_ids, scores)}
        # For paired Δ, only include POS tnps; the pair is (POS, POS + suffix)
        pos_scores_by_tnp = {t: s for t, s in scores_by_tnp.items()
                              if not t.endswith(args.neg_suffix)}
        paired = paired_delta_by_suffix({**pos_scores_by_tnp, **scores_by_tnp}, args.neg_suffix)
        reports[name] = {
            "auroc": float(au) if au == au else None,
            "auprc": float(ap) if ap == ap else None,
            "paired_delta": paired,
        }
        pd_str = ""
        if paired:
            pd_str = f"  Δ median={paired['median']:+.3f}  MAD={paired['MAD']:.3f}  P(Δ>0)={paired['p_gt_0']:.3f}  n={paired['n']}"
        print(f"  {name:<16} AUROC={au:.4f}  AUPRC={ap:.4f}{pd_str}")

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "jsonl": args.jsonl,
            "frozen_platt_json": args.frozen_platt_json,
            "neg_suffix": args.neg_suffix,
            "soft_min_tau": tau,
            "n_pos": int((labels==1).sum()),
            "n_neg": int((labels==0).sum()),
            "reports": reports,
        }, f, indent=2)
    print(f"\n[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
