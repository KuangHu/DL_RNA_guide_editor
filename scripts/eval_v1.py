"""Evaluate a V1 checkpoint on any split, with the V4 guided-fraction curve.

Generalises scripts/eval_v1_on_v3_test.py (which hardcodes the V3 paths) and
adds the metric V4 is built to make meaningful:

    x = fraction of the bag's sites that are truly `guided`
    y = median model score, recall @ 0.5, and AUROC against the negatives
        in the SAME guided-fraction bin

Under the V3 distribution that curve is meaningless, because guided fraction
is itself a label (positives are noisy, negatives are not). Under V4 the
negatives carry a matched guided fraction, so a rising curve is evidence the
model rewards shared grammar rather than counting anomalies.

Usage:
    python -m scripts.eval_v1 \
        --ckpt      checkpoints/v1_on_v3/best.pt \
        --test-jsonl /.../splits/test_v4.jsonl \
        --test-cache /.../structure/test_v4_u16.index.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.v1 import V1Config, V1Model
from preprocess.site import StructureCache
from preprocess.tnp_dataset import (TnpGroupedDataset, collate_tnp_batch,
                                     make_torch_tnp_dataset)
from training.metrics import (_auprc, _auroc, candidate_recall,
                               nc_selection_accuracy, stratified_auroc,
                               tnp_metrics)
from training.train_v1 import (EASY_PROFILES, _tnp_strength_by_tnp,
                                _violation_profile_by_tnp)


def guided_fraction_by_tnp(path: str) -> dict[str, float]:
    """Fraction of each bag's sites labelled `guided`.

    Defined for negatives too: V4 stamps the matched composition onto every
    negative record, which is what makes the binned comparison legitimate.
    """
    n_sites: dict[str, int] = defaultdict(int)
    n_guided: dict[str, int] = defaultdict(int)
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            t = r["transposase_id"]
            n_sites[t] += 1
            if r["labels"].get("site_class", "guided") == "guided":
                n_guided[t] += 1
    return {t: n_guided[t] / n for t, n in n_sites.items() if n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--test-jsonl", required=True, type=Path)
    ap.add_argument("--test-cache", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--sites", type=int, default=50)
    ap.add_argument("--curve-negatives", default="level3_paired_counterfactual",
                    help="comma-separated profiles used as the in-bin negatives "
                         "for the guided-fraction curve; 'all' uses every negative")
    ap.add_argument("--out-json", type=Path, default=None)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    model = V1Model(V1Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {a.ckpt}: epoch {ck['epoch']}, val AUPRC={ck['auprc']:.4f}")

    cache = StructureCache(str(a.test_cache))
    ds = TnpGroupedDataset(str(a.test_jsonl), cache,
                           site_subsample_size=a.sites, rng_seed=0)
    print(f"test tnps={len(ds)}")
    dl = DataLoader(make_torch_tnp_dataset(ds), batch_size=a.batch_size, shuffle=False,
                    num_workers=a.num_workers,
                    collate_fn=lambda x: collate_tnp_batch(x, to_torch=True),
                    persistent_workers=a.num_workers > 0, pin_memory=True)

    gmap = _violation_profile_by_tnp(str(a.test_jsonl))
    smap = _tnp_strength_by_tnp(str(a.test_jsonl))
    fmap = guided_fraction_by_tnp(str(a.test_jsonl))

    t0 = time.time()
    scores, labels, tnp_ids = [], [], []
    cand_at_active, true_slot_all, active_all, nc_attn_all = [], [], [], []
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for b in dl:
            b = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in b.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=use_amp):
                out = model(b["candidate_patches"], b["candidate_features"],
                            b["candidate_mask"], b["nc_region_mask"])
            scores.append(torch.sigmoid(out["logit"]).float().cpu().numpy())
            labels.append(b["is_positive"].cpu().numpy())
            tnp_ids.extend(list(b["tnp_id"]))
            cr = out["cand_raw"].float().cpu().numpy()
            na = out["nc_attn"].float().cpu().numpy()
            ac = b["active_nc_index"].cpu().numpy()
            ts = b["true_slot_idx"].cpu().numpy()
            B, S = cr.shape[0], cr.shape[1]
            for bi in range(B):
                for si in range(S):
                    if int(ac[bi, si]) < 0 or int(ts[bi, si]) < 0:
                        continue
                    cand_at_active.append(cr[bi, si, int(ac[bi, si])])
                    true_slot_all.append(int(ts[bi, si]))
                    active_all.append(int(ac[bi, si]))
                    nc_attn_all.append(na[bi, si])

    scores = np.concatenate(scores)
    labels = np.concatenate(labels).astype(bool)
    groups = np.asarray([gmap[t] for t in tnp_ids])
    strengths = np.asarray([smap.get(t, "unknown") for t in tnp_ids])
    gfrac = np.asarray([fmap.get(t, np.nan) for t in tnp_ids])

    m = tnp_metrics(scores, labels)
    strat = stratified_auroc(scores, labels, groups)
    hard_mask = labels | np.array([g not in EASY_PROFILES for g in groups])
    hard_auroc = _auroc(scores[hard_mask], labels[hard_mask])
    hard_auprc = _auprc(scores[hard_mask], labels[hard_mask])

    print(f"eval done in {time.time()-t0:.1f}s\n")
    print(f"  n_tnp_pos={m['n_pos']}  n_tnp_neg={m['n_neg']}")
    print(f"  AUROC={m['auroc']:.4f}    AUPRC={m['auprc']:.4f}")
    print(f"  HARD_AUROC={hard_auroc:.4f}    HARD_AUPRC={hard_auprc:.4f}")
    if cand_at_active:
        cand = candidate_recall(np.stack(cand_at_active, 0), true_slot_all, ks=(1, 5, 10))
        nc = nc_selection_accuracy(np.stack(nc_attn_all, 0), active_all)
        print(f"  R@1={cand['recall@1']:.3f}  R@5={cand['recall@5']:.3f} "
              f"R@10={cand['recall@10']:.3f}   (n={cand['n']})")
        print(f"  NC top-1: {nc['nc_top1']:.3f}   (n={nc['n']})")

    print("\n  Per-profile AUROC:")
    for k in sorted(strat):
        print(f"    {k[6:-1]:<45} {strat[k]:.4f}")

    # ---- strength ordering: the V3 inversion check -----------------------
    print("\n  AUROC by positive strength (V3 had weak >> strong, which is backwards):")
    for prof in sorted(set(groups[~labels])):
        neg = scores[(~labels) & (groups == prof)]
        line = f"    vs {prof:<42}"
        for lvl in ("strong", "moderate", "weak"):
            mask = labels & (strengths == lvl)
            if mask.any():
                line += f"  {lvl}={_auroc(np.concatenate([scores[mask], neg]), np.concatenate([np.ones(mask.sum(), bool), np.zeros(neg.size, bool)])):.3f}"
        print(line)

    # ---- guided-fraction curve ------------------------------------------
    if a.curve_negatives == "all":
        curve_neg_mask = ~labels
    else:
        wanted = set(a.curve_negatives.split(","))
        curve_neg_mask = (~labels) & np.isin(groups, list(wanted))
    edges = np.arange(0.3, 1.0001, 0.1)
    print(f"\n  Guided-fraction curve (in-bin negatives: {a.curve_negatives}, "
          f"n={int(curve_neg_mask.sum())}):")
    print(f"    {'bin':<12}{'n_pos':>7}{'n_neg':>7}{'median score':>14}"
          f"{'recall@0.5':>12}{'AUROC in bin':>14}")
    curve = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        pm = labels & (gfrac >= lo) & (gfrac < hi + (1e-9 if hi >= 1.0 else 0))
        nm = curve_neg_mask & (gfrac >= lo) & (gfrac < hi + (1e-9 if hi >= 1.0 else 0))
        if not pm.any():
            continue
        sp, sn = scores[pm], scores[nm]
        au = (_auroc(np.concatenate([sp, sn]),
                     np.concatenate([np.ones(sp.size, bool), np.zeros(sn.size, bool)]))
              if sn.size else float("nan"))
        print(f"    {f'{lo:.0%}-{hi:.0%}':<12}{sp.size:>7}{sn.size:>7}"
              f"{np.median(sp):>14.4f}{(sp > 0.5).mean():>12.4f}{au:>14.4f}")
        curve.append({"lo": float(lo), "hi": float(hi), "n_pos": int(sp.size),
                      "n_neg": int(sn.size), "median_score": float(np.median(sp)),
                      "recall": float((sp > 0.5).mean()), "auroc": float(au)})

    if a.out_json:
        a.out_json.parent.mkdir(parents=True, exist_ok=True)
        a.out_json.write_text(json.dumps({
            "ckpt": str(a.ckpt), "split": str(a.test_jsonl),
            "auroc": float(m["auroc"]), "auprc": float(m["auprc"]),
            "hard_auroc": float(hard_auroc), "hard_auprc": float(hard_auprc),
            "per_profile_auroc": {k: float(v) for k, v in strat.items()},
            "guided_fraction_curve": curve,
            "scores": scores.tolist(), "labels": labels.tolist(),
            "groups": groups.tolist(), "strengths": strengths.tolist(),
            "guided_fraction": gfrac.tolist(), "tnp_ids": tnp_ids,
        }, indent=2))
        print(f"\nwrote {a.out_json}")


if __name__ == "__main__":
    main()
