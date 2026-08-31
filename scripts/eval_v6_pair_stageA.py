"""V6 Stage A formal val-time pair diagnostic.

Loads V6 Stage A ep0 best.pt, runs val_v4_no_l3 with swap augmentation,
and computes the pair-discrimination metrics on GUIDED sites of positive
bags.

Reports:
  - Δ_pair distribution: mean, median, Q10/25/75/90
  - P(Δ > 0), P(Δ > 1), P(Δ > 2)
  - Cognate-vs-swap AUROC (pooled q values, cognate labeled +, swap labeled −)
  - Same metrics restricted to strong / moderate / weak positive strengths

Passage thresholds:
  AUROC(pair) > 0.80
  P(Δ > 0)     > 0.75
  median(Δ)    > 1.0
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.v1 import V1Config, V1Model
from preprocess.site import StructureCache
from preprocess.tnp_dataset import TnpGroupedDataset, collate_tnp_batch, make_torch_tnp_dataset

BASE = '/global/scratch/users/kh36969/DL_novel_guide_editor'
CKPT = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/checkpoints/v6_stageA_from_v52/best.pt'
VAL_JSONL = f'{BASE}/splits/val_v4_no_l3.jsonl'
VAL_CACHE = f'{BASE}/structure/val_v4_u16.index.json'


def _auroc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if labels.sum() == 0 or (~labels).sum() == 0:
        return float('nan')
    order = np.argsort(-scores, kind='mergesort')
    y = labels[order]
    tps = np.cumsum(y); fps = np.cumsum(~y)
    tps = np.concatenate([[0], tps]); fps = np.concatenate([[0], fps])
    tpr = tps / max(1, tps[-1]); fpr = fps / max(1, fps[-1])
    return float(np.trapezoid(tpr, fpr))


def _tnp_strength_by_tnp(jsonl_path):
    out = {}
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            tnp = r['transposase_id']
            if tnp in out: continue
            meta = r.get('generator_metadata') or {}
            out[tnp] = meta.get('tnp_strength', 'unknown')
    return out


def main():
    device = torch.device('cuda')
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = V1Config(**ckpt['cfg'])
    model = V1Model(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'[ckpt] {CKPT}', flush=True)
    print(f'         epoch={ckpt["epoch"]}, val AUPRC (bag) = {ckpt["auprc"]:.4f}')
    print(f'         pair_beta = {model.pair_beta.item():.4f}')
    print(f'         use_pairing = {cfg.use_pairing}, dispersion_mode = {cfg.dispersion_mode}', flush=True)

    cache = StructureCache(VAL_CACHE)
    ds = TnpGroupedDataset(VAL_JSONL, cache, site_subsample_size=50, rng_seed=0,
                            generate_swap=True)
    dl = DataLoader(make_torch_tnp_dataset(ds), batch_size=8, shuffle=False,
                    num_workers=4,
                    collate_fn=lambda x: collate_tnp_batch(x, to_torch=True),
                    persistent_workers=True, pin_memory=True)
    strengths_map = _tnp_strength_by_tnp(VAL_JSONL)

    print(f'\n[data] {len(ds)} val tnps, swap augmentation enabled', flush=True)

    t0 = time.time()
    all_q_cog: list[float] = []
    all_q_swp: list[float] = []
    all_strength: list[str] = []
    all_tnp_id: list[str] = []
    n_bags_pos = 0
    with torch.no_grad():
        for b in dl:
            b_dev = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in b.items()}
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out_cog = model(b_dev['candidate_patches'], b_dev['candidate_features'],
                                  b_dev['candidate_mask'], b_dev['nc_region_mask'])
                out_swp = model(b_dev['candidate_patches_swap'], b_dev['candidate_features_swap'],
                                  b_dev['candidate_mask_swap'], b_dev['nc_region_mask_swap'])
            q_cog = out_cog['q_nc'].float().cpu().numpy()   # (B, S, N)
            q_swp = out_swp['q_nc'].float().cpu().numpy()
            anc = b['active_nc_index'].cpu().numpy()
            pm = b['pair_mask'].cpu().numpy()
            B, S = pm.shape
            for bi in range(B):
                tid = b['tnp_id'][bi]
                strength = strengths_map.get(tid, 'unknown')
                if b['is_positive'][bi].item():
                    n_bags_pos += 1
                for si in range(S):
                    if not pm[bi, si]:
                        continue
                    slot = int(anc[bi, si])
                    if slot < 0:
                        continue
                    all_q_cog.append(float(q_cog[bi, si, slot]))
                    all_q_swp.append(float(q_swp[bi, si, slot]))
                    all_strength.append(strength)
                    all_tnp_id.append(tid)

    q_cog = np.asarray(all_q_cog)
    q_swp = np.asarray(all_q_swp)
    delta = q_cog - q_swp
    strengths = np.asarray(all_strength)
    print(f'[data] scored {n_bags_pos} positive bags, {len(delta)} guided-site pair samples '
          f'in {time.time()-t0:.1f}s', flush=True)

    # ================ Overall ================
    print()
    print('=' * 92)
    print('  V6 Stage A — pair-discrimination metrics on val_v4_no_l3 (guided sites of positives)')
    print('=' * 92)
    def _stats(d, tag='overall', n_expected=None):
        n = len(d)
        med = float(np.median(d))
        mean = float(d.mean())
        q10, q25, q75, q90 = np.quantile(d, [0.10, 0.25, 0.75, 0.90]).tolist()
        p0 = float((d > 0).mean())
        p1 = float((d > 1).mean())
        p2 = float((d > 2).mean())
        print(f'\n  [{tag}] n = {n}')
        print(f'    Δ_pair mean = {mean:+.3f}   median = {med:+.3f}   '
              f'Q10 = {q10:+.3f}  Q25 = {q25:+.3f}  Q75 = {q75:+.3f}  Q90 = {q90:+.3f}')
        print(f'    P(Δ > 0) = {p0:.4f}   P(Δ > 1) = {p1:.4f}   P(Δ > 2) = {p2:.4f}')
        return dict(n=n, mean=mean, median=med, q10=q10, q75=q75, q90=q90,
                     p_gt_0=p0, p_gt_1=p1, p_gt_2=p2)

    overall = _stats(delta, 'overall')

    # AUROC pooled
    pooled_scores = np.concatenate([q_cog, q_swp])
    pooled_labels = np.concatenate([np.ones(len(q_cog), dtype=bool),
                                      np.zeros(len(q_swp), dtype=bool)])
    auroc_pair = _auroc(pooled_scores, pooled_labels)
    print(f'\n  AUROC(cognate vs swap, pooled) = {auroc_pair:.4f}')

    # ================ Per-strength ================
    print()
    print('=' * 92)
    print(f'  Per positive-strength (strong/moderate/weak)')
    print('=' * 92)
    strat_results = {}
    for lvl in ('strong', 'moderate', 'weak'):
        m = (strengths == lvl)
        if not m.any(): continue
        _stats(delta[m], f'POS-{lvl}')
        pooled_s = np.concatenate([q_cog[m], q_swp[m]])
        pooled_l = np.concatenate([np.ones(int(m.sum()), dtype=bool),
                                     np.zeros(int(m.sum()), dtype=bool)])
        strat_results[lvl] = _auroc(pooled_s, pooled_l)
    print(f'\n  AUROC(cognate vs swap) per strength:')
    for lvl, a in strat_results.items():
        print(f'    {lvl:<10} {a:.4f}')

    # ================ Passage check ================
    print()
    print('=' * 92)
    print(f'  Stage A passage criteria')
    print('=' * 92)
    print(f'  AUROC(pair)    > 0.80 : {auroc_pair:.4f}   {"✅ PASS" if auroc_pair > 0.80 else "✗ FAIL"}')
    print(f'  P(Δ > 0)       > 0.75 : {overall["p_gt_0"]:.4f}   {"✅ PASS" if overall["p_gt_0"] > 0.75 else "✗ FAIL"}')
    print(f'  median(Δ)      > 1.00 : {overall["median"]:+.3f}   {"✅ PASS" if overall["median"] > 1.0 else "✗ FAIL"}')
    passed = (auroc_pair > 0.80) and (overall["p_gt_0"] > 0.75) and (overall["median"] > 1.0)
    print()
    print(f'  Stage A verdict: {"✅ PASSED — proceed to Stage B" if passed else "✗ FAILED — need to retrain"}')

    # Save
    outp = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/logs/v6_stageA_pair_val_diag.json'
    with open(outp, 'w') as fh:
        json.dump({
            'auroc_pair': auroc_pair,
            'overall': overall,
            'strat': strat_results,
            'passed': passed,
        }, fh, indent=2)
    print(f'\n  saved {outp}')


if __name__ == '__main__':
    main()
