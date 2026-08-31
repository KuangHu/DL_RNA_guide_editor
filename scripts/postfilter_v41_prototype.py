"""Post-filter prototype on V4.1 assembled sample.

Simple rule (for prototype only — NOT the final generator design):
  keep if rank <= threshold
  else keep with probability tail_keep_prob

Reports BEFORE/AFTER:
  overall rank median/Q75/Q90
  overall R@4/R@8/R@20
  identity distribution
  L distribution (critical — verify short-L not being scrubbed)
  per-L rank_med/Q90/R@4/R@20/%bad

Purpose: verify that removing pathological tail moves calibration in the
right direction, WITHOUT dropping so many short-L sites that L support is
distorted.

The final generator design (post-verification) will use conditional
difficulty resampling at fixed L/background — not blind rejection.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')

import numpy as np

from sim2real_rank_margin import (
    find_true_candidate, rank_true_and_max_decoy,
)


def compute_rank_per_record(records, max_records=2000, seed=42):
    """Return list of {rec_idx, L, identity, rank, s_true, s_decoy}."""
    if len(records) > max_records:
        rng = random.Random(seed)
        records = rng.sample(records, max_records)
    out = []
    for idx, rec in enumerate(records):
        if not rec['labels'].get('is_positive'): continue
        acn = rec['labels'].get('active_noncoding_index', 0)
        ncs = rec['inputs']['noncoding_regions']
        if acn >= len(ncs): continue
        nc = ncs[acn]; flank = rec['inputs']['flank']
        true = find_true_candidate(rec)
        if true is None: continue
        res = rank_true_and_max_decoy(nc, flank, true)
        if res is None: continue
        rank, s_true, s_dec = res
        out.append({'rec_idx': idx, 'rec': rec,
                    'L': true['L'], 'identity': s_true / true['L'],
                    'rank': rank, 's_true': s_true, 's_decoy': s_dec})
    return out


def apply_filter(rows, rank_threshold=100, tail_keep_prob=0.1, seed=1234):
    rng = random.Random(seed)
    kept = []
    for r in rows:
        if r['rank'] <= rank_threshold:
            kept.append(r)
        elif rng.random() < tail_keep_prob:
            kept.append(r)
    return kept


def summarize(rows, label):
    print(f'\n  {label}   n={len(rows)}')
    if not rows: return
    r = np.asarray([x['rank'] for x in rows])
    idt = np.asarray([x['identity'] for x in rows])
    Ls = np.asarray([x['L'] for x in rows])
    print(f'    rank : median={int(np.median(r))}  Q75={int(np.quantile(r,.75))}  '
          f'Q90={int(np.quantile(r,.90))}   max={int(r.max())}')
    print(f'    R@K  : R@4={np.mean(r<=4):.3f}  R@8={np.mean(r<=8):.3f}  '
          f'R@20={np.mean(r<=20):.3f}   P(rank>100)={np.mean(r>100):.3f}')
    print(f'    id   : median={np.median(idt):.3f}  Q10={np.quantile(idt,.10):.3f}  '
          f'Q25={np.quantile(idt,.25):.3f}  Q75={np.quantile(idt,.75):.3f}')

    print(f'    L distribution:')
    cnt = Counter(Ls.tolist())
    total = sum(cnt.values())
    print(f'      L: ' + '  '.join(f'{L}({cnt[L]}/{100*cnt[L]/total:.0f}%)'
                                    for L in sorted(cnt)))


def per_L_table(rows, label):
    print(f'\n  {label} per-L breakdown:')
    print(f'    {"L":>3} {"n":>4}  {"rank_med":>9} {"rank_Q90":>9}  '
          f'{"R@4":>5} {"R@20":>5}  {"%_bad":>6}')
    by_L = defaultdict(list)
    for r in rows:
        by_L[r['L']].append(r['rank'])
    for L in sorted(by_L.keys()):
        r = np.asarray(by_L[L])
        if len(r) < 3: continue
        print(f'    {L:>3} {len(r):>4}  {int(np.median(r)):>9} '
              f'{int(np.quantile(r,.90)):>9}  '
              f'{np.mean(r<=4):>5.2f} {np.mean(r<=20):>5.2f}  '
              f'{100*np.mean(r>100):>5.1f}%')


def L_retention(before, after):
    """Show per-L retention rate."""
    print(f'\n  L distribution retention (before -> after):')
    print(f'    {"L":>3} {"before":>7} {"after":>7} {"retained_%":>10}')
    b_cnt = Counter(r['L'] for r in before)
    a_cnt = Counter(r['L'] for r in after)
    for L in sorted(b_cnt.keys()):
        b = b_cnt[L]; a = a_cnt.get(L, 0)
        pct = 100 * a / max(1, b)
        print(f'    {L:>3} {b:>7} {a:>7} {pct:>10.1f}%')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v41', default='/global/scratch/users/kh36969/tmp/v41_cal/assembled_v41.jsonl')
    ap.add_argument('--rank-threshold', type=int, default=100)
    ap.add_argument('--tail-keep-prob', type=float, default=0.1)
    ap.add_argument('--max-records', type=int, default=2000)
    args = ap.parse_args()

    print(f'[load] {args.v41}')
    with open(args.v41) as f: recs = [json.loads(l) for l in f]

    print(f'[compute] per-record rank on {min(args.max_records, len(recs))} records ...',
          flush=True)
    rows = compute_rank_per_record(recs, max_records=args.max_records)
    print(f'  processed {len(rows)}')

    print(f'\n{"="*100}')
    print(f'  POST-FILTER PROTOTYPE — rank_threshold={args.rank_threshold}, '
          f'tail_keep_prob={args.tail_keep_prob}')
    print(f'{"="*100}')

    kept = apply_filter(rows, rank_threshold=args.rank_threshold,
                        tail_keep_prob=args.tail_keep_prob)

    summarize(rows, 'BEFORE (all V4.1 sites)')
    summarize(kept, 'AFTER post-filter')

    per_L_table(rows, 'BEFORE')
    per_L_table(kept, 'AFTER')

    L_retention(rows, kept)

    # Check L distortion — is L distribution proportional after filter?
    print(f'\n  L distortion check (χ² test if we had it — for now just visual):')
    b_cnt = Counter(r['L'] for r in rows); b_total = sum(b_cnt.values())
    a_cnt = Counter(r['L'] for r in kept); a_total = sum(a_cnt.values())
    max_shift = 0
    for L in sorted(b_cnt):
        b_frac = b_cnt[L] / b_total
        a_frac = a_cnt.get(L, 0) / max(1, a_total)
        shift = abs(a_frac - b_frac)
        if shift > max_shift: max_shift = shift
    print(f'    max L-fraction shift: {max_shift:.3f}')
    print(f'    (If > 0.03, L distribution is materially distorted — need difficulty ')
    print(f'     resampling instead of drop.)')


if __name__ == '__main__':
    main()
