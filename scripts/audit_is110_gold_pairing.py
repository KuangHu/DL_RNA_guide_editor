"""Cognate vs shuffled TBL + DBL pairing benchmark on IS110_gold_v0.

For each of two arms (TBL, DBL) and two flavors:
  A) cognate:   (this row's bRNA spec) vs (this row's genome-actual site)
  B) shuffled:  (this row's bRNA spec) vs (another row's genome-actual site)
                — where the "other" row uses a DIFFERENT bRNA (cross-bRNA shuffle)

Also emits a DUAL-ARM score:
  dual(row) = 11 - hamming(TBL) + 11 - hamming(DBL) = matches_TBL + matches_DBL
    (higher = better)

Reports:
  Hamming distribution (median, mean, %(0), %(<=1), %(<=3))
  AUROC of cognate vs cross-bRNA shuffled  (label 1 = cognate, 0 = shuffled)
  Dual-arm AUROC

The critical test is: do cognate DUAL-ARM matches beat cross-bRNA shuffled ones
by a wide, biologically-plausible margin? If YES → the LTG/RTG rule is real,
extractable, and directly usable for a V7 pair_head training signal.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def hamming(a, b):
    if not a or not b or len(a) != len(b):
        return None
    return sum(1 for x, y in zip(a, b) if x != y)


def auroc(scores, labels):
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


def load_gold(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _trim11(s):
    if not s: return None
    return s[:11]


def run_arm(rows, arm, seed=42, pool_rows=None):
    """arm: 'TBL' or 'DBL'. Returns cognate + shuffled Hamming lists.

    Both spec and genome target are trimmed to 11 bp so length matches.
    `pool_rows` is the source of shuffle candidates (defaults to `rows`);
    when `rows` is a single-bRNA split, pass the full corpus as pool_rows
    to get a meaningful cross-bRNA shuffle.
    """
    if pool_rows is None: pool_rows = rows
    spec_key = f'{arm.lower()}_spec_11bp'   # from Table 2; may be 11 or 14 chars
    tgt_key = 'genome_target_11bp'
    ok = [r for r in rows
          if r.get(spec_key) and r.get(tgt_key)
          and len(_trim11(r[spec_key])) == 11
          and len(_trim11(r[tgt_key])) == 11]
    cognate = [(r, hamming(_trim11(r[spec_key]), _trim11(r[tgt_key]))) for r in ok]
    cognate = [(r, h) for r, h in cognate if h is not None]

    # Build shuffle pool: (bRNA, genome_target_11bp) pairs from pool_rows
    pool = [(x['ortholog_id'], _trim11(x[tgt_key]))
            for x in pool_rows
            if x.get(tgt_key) and len(_trim11(x[tgt_key])) == 11]

    rng = random.Random(seed)
    shuf = []
    for r, _ in cognate:
        # pick a pool entry from a DIFFERENT bRNA
        others = [p for p in pool if p[0] != r['ortholog_id']]
        if not others: continue
        other_bR, other_tgt = rng.choice(others)
        h = hamming(_trim11(r[spec_key]), other_tgt)
        if h is not None: shuf.append((r, h, other_bR))

    return cognate, shuf


def summarize(name, cognate, shuf):
    cog_h = np.asarray([h for _, h in cognate])
    shu_h = np.asarray([h for _, h, _ in shuf])
    print(f'\n  {name}:')
    print(f'    n cognate={len(cog_h)}, n shuffled={len(shu_h)}')
    print(f'    cognate Hamming : median={np.median(cog_h):.1f}  mean={cog_h.mean():.2f}  '
          f'%(0)={100*(cog_h==0).mean():.1f}  %(<=1)={100*(cog_h<=1).mean():.1f}  '
          f'%(<=3)={100*(cog_h<=3).mean():.1f}')
    print(f'    shuffled Hamming: median={np.median(shu_h):.1f}  mean={shu_h.mean():.2f}  '
          f'%(0)={100*(shu_h==0).mean():.1f}  %(<=1)={100*(shu_h<=1).mean():.1f}  '
          f'%(<=3)={100*(shu_h<=3).mean():.1f}')
    # AUROC: higher score = better cognate. Convert distance → matches.
    scores = np.concatenate([11 - cog_h, 11 - shu_h])
    labels = np.concatenate([np.ones(len(cog_h)), np.zeros(len(shu_h))])
    au = auroc(scores, labels)
    print(f'    AUROC (11 - hamming, label 1=cognate): {au:.4f}')
    return au


def run_dual_arm(rows, seed=42, pool_rows=None):
    """Combine TBL + DBL arms into a single dual-arm score per row."""
    cog_tbl, shu_tbl = run_arm(rows, 'TBL', seed=seed, pool_rows=pool_rows)
    cog_dbl, shu_dbl = run_arm(rows, 'DBL', seed=seed + 1, pool_rows=pool_rows)
    # Index by system_id
    tbl_by = {r['system_id']: h for r, h in cog_tbl}
    dbl_by = {r['system_id']: h for r, h in cog_dbl}
    # Cognate dual
    dual_cog = [22 - tbl_by[k] - dbl_by[k] for k in tbl_by if k in dbl_by]
    # Shuffled dual (align by system_id where available)
    shu_tbl_by = {r['system_id']: h for r, h, _ in shu_tbl}
    shu_dbl_by = {r['system_id']: h for r, h, _ in shu_dbl}
    dual_shu = [22 - shu_tbl_by[k] - shu_dbl_by[k]
                for k in shu_tbl_by if k in shu_dbl_by]
    dual_cog = np.asarray(dual_cog); dual_shu = np.asarray(dual_shu)
    print(f'\n  DUAL-ARM (TBL matches + DBL matches, /22):')
    print(f'    n cognate={len(dual_cog)}, n shuffled={len(dual_shu)}')
    print(f'    cognate matches : median={np.median(dual_cog):.1f}  mean={dual_cog.mean():.2f}  '
          f'%(>=18)={100*(dual_cog>=18).mean():.1f}  %(>=20)={100*(dual_cog>=20).mean():.1f}')
    print(f'    shuffled matches: median={np.median(dual_shu):.1f}  mean={dual_shu.mean():.2f}  '
          f'%(>=18)={100*(dual_shu>=18).mean():.1f}  %(>=20)={100*(dual_shu>=20).mean():.1f}')
    scores = np.concatenate([dual_cog, dual_shu])
    labels = np.concatenate([np.ones(len(dual_cog)), np.zeros(len(dual_shu))])
    au = auroc(scores, labels)
    print(f'    AUROC dual-arm : {au:.4f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gold', default='/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/curated/is110_gold_v0.jsonl')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    rows = load_gold(args.gold)
    print(f'[load] {len(rows)} rows from {args.gold}')

    # Split by source: WT vs Programmed have different characteristics
    for split_name, subset in (
        ('ALL',                       rows),
        ('WT (n=173, 1 bRNA)',         [r for r in rows if r['source'] == 'Durrant2024_WT']),
        ('Programmed (n=168, 8 bRNAs)', [r for r in rows if r['source'] == 'Durrant2024_Programmed']),
        ('Programmed On-Target only',   [r for r in rows if r['source'] == 'Durrant2024_Programmed' and r.get('on_target')]),
    ):
        print(f'\n{"="*90}')
        print(f'  SPLIT: {split_name}   (n={len(subset)})')
        print(f'{"="*90}')
        cog_tbl, shu_tbl = run_arm(subset, 'TBL', seed=args.seed, pool_rows=rows)
        _ = summarize('TBL arm', cog_tbl, shu_tbl)
        cog_dbl, shu_dbl = run_arm(subset, 'DBL', seed=args.seed + 1, pool_rows=rows)
        _ = summarize('DBL arm', cog_dbl, shu_dbl)
        run_dual_arm(subset, seed=args.seed, pool_rows=rows)


if __name__ == '__main__':
    main()
