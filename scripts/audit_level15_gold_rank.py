"""Level 1.5 — gold candidate RANK distribution + junction-pool proposal test.

Two questions:

  Q1. For each cognate Durrant site, what is the rank of the "gold" candidate
      (the junction-anchored L=11 fwd/rc alignment) in the current proposal
      score (matches, per (orient, L))?

      -> Recall@K for K = 4, 8, 20, 50, 100
      -> Rank distribution median / Q75 / Q90

  Q2. Does the proposed hybrid pool "global_topK=4 ∪ junction_topK=4"
      recover the gold candidate at K = 8 total?

The "gold" alignment for a cognate site is defined as: the L=11 candidate at
flank_start in [0, 5] with the HIGHEST match count (over both orientations).
This is our best proxy for the true junction-anchored LtG/RtG cognate hit.

Both files are per-site oriented — no NC MIL, no model. Pure candidate
proposal analysis.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')

import numpy as np

BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
RC = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}


def seq_to_arr(s):
    return np.asarray([BASE_MAP.get(c, 4) for c in s.upper()], dtype=np.int8)


def rc_seq(s):
    return ''.join(RC.get(c, 'N') for c in s[::-1].upper())


def enumerate_all_L11(nc: str, flank: str):
    """Return a list of dicts: {orient, nc_start, flank_start, matches}
    for every possible ungapped L=11 alignment."""
    L = 11
    nc_a = seq_to_arr(nc)
    fk_a = seq_to_arr(flank)
    fk_rc_a = seq_to_arr(rc_seq(flank))
    if len(nc_a) < L or len(fk_a) < L:
        return []
    nc_win = np.lib.stride_tricks.sliding_window_view(nc_a, L)  # (n_nc, L)
    a_oh = np.eye(5, dtype=np.int8)[nc_win]
    out = []
    for orient, fw in (('fwd', fk_a), ('rc', fk_rc_a)):
        fw_win = np.lib.stride_tricks.sliding_window_view(fw, L)  # (n_fk, L)
        b_oh = np.eye(5, dtype=np.int8)[fw_win]
        M = np.einsum('nlc,mlc->nm', a_oh, b_oh)   # (n_nc, n_fk)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                # for orient='rc', j indexes into flank_rc; convert to fwd flank coord
                if orient == 'fwd':
                    flank_start = j
                else:
                    flank_start = len(fk_a) - j - L
                out.append({'orient': orient, 'nc_start': i,
                            'flank_start': flank_start, 'matches': int(M[i, j])})
    return out


def rank_of_junction_candidate(all_cands, junction_bound=5):
    """Given the exhaustive candidate list, find the junction-anchored 'gold'
    candidate (max matches at flank_start ≤ junction_bound) and compute:
      - its global rank when sorted by (matches desc, index asc)
      - its rank restricted to its own orient
    """
    if not all_cands:
        return None
    # Gold = best match at flank_start ≤ junction_bound
    at_junction = [c for c in all_cands if c['flank_start'] <= junction_bound]
    if not at_junction:
        return None
    gold = max(at_junction, key=lambda c: c['matches'])

    # Rank by matches descending among ALL L=11 candidates
    sorted_all = sorted(all_cands, key=lambda c: -c['matches'])
    # position of gold: first index where matches == gold['matches'] AND
    # nc_start,flank_start,orient identity match
    gold_rank_all = None
    for r, c in enumerate(sorted_all):
        if (c['matches'] == gold['matches']
            and c['nc_start'] == gold['nc_start']
            and c['flank_start'] == gold['flank_start']
            and c['orient'] == gold['orient']):
            gold_rank_all = r + 1
            break

    # Also: rank restricted to gold's orient only (the current top-K
    # is per (orient, L) — so this is the meaningful rank for K=top_k_per_combo)
    per_orient = [c for c in all_cands if c['orient'] == gold['orient']]
    per_orient_sorted = sorted(per_orient, key=lambda c: -c['matches'])
    gold_rank_per_orient = None
    for r, c in enumerate(per_orient_sorted):
        if (c['matches'] == gold['matches']
            and c['nc_start'] == gold['nc_start']
            and c['flank_start'] == gold['flank_start']):
            gold_rank_per_orient = r + 1
            break

    # Ties: how many candidates have matches ≥ gold['matches']?
    n_ties_ge = sum(1 for c in all_cands if c['matches'] >= gold['matches'])
    n_ties_gt = sum(1 for c in all_cands if c['matches'] > gold['matches'])

    return {
        'gold_matches': gold['matches'],
        'gold_orient': gold['orient'],
        'gold_nc_start': gold['nc_start'],
        'gold_flank_start': gold['flank_start'],
        'rank_all': gold_rank_all,
        'rank_per_orient': gold_rank_per_orient,
        'n_ties_ge_gold': n_ties_ge,
        'n_ties_gt_gold': n_ties_gt,
        'n_all_L11': len(all_cands),
    }


def audit(jsonl_path, label):
    records = [json.loads(l) for l in Path(jsonl_path).read_text().splitlines()]
    print(f'[{label}] {len(records)} sites')
    rows = []
    for i, rec in enumerate(records):
        nc = rec['inputs']['noncoding_regions'][rec['labels']['active_noncoding_index']]
        flank = rec['inputs']['flank']
        cands = enumerate_all_L11(nc, flank)
        info = rank_of_junction_candidate(cands, junction_bound=5)
        if info: rows.append({'site_id': rec['site_id'], 'bag': rec['transposase_id'],
                              'brna': rec['generator_metadata']['is_id'], **info})
        if (i + 1) % 100 == 0:
            print(f'  processed {i + 1}/{len(records)}')
    return rows


def report_ranks(rows, label):
    print(f'\n{"="*90}')
    print(f'  {label}: gold-candidate rank distribution (per-orient sort, current top-K basis)')
    print(f'{"="*90}\n')
    r_orient = np.asarray([r['rank_per_orient'] for r in rows])
    r_all = np.asarray([r['rank_all'] for r in rows])
    gold_m = np.asarray([r['gold_matches'] for r in rows])

    print(f'  gold matches: median={int(np.median(gold_m))}, mean={gold_m.mean():.2f}, '
          f'Q10={int(np.quantile(gold_m,.1))}, Q90={int(np.quantile(gold_m,.9))}')

    print(f'\n  Rank within its own orient (this is the rank that matters for top_k_per_combo):')
    print(f'    median={int(np.median(r_orient))}  Q75={int(np.quantile(r_orient,.75))}  '
          f'Q90={int(np.quantile(r_orient,.9))}  Q95={int(np.quantile(r_orient,.95))}  '
          f'max={int(r_orient.max())}')
    for K in (1, 2, 4, 8, 12, 20, 50, 100):
        pct = 100 * (r_orient <= K).mean()
        print(f'    Recall@K={K:>3}: {pct:>5.1f}%')

    print(f'\n  Rank across BOTH orients (relevant if K applies globally):')
    print(f'    median={int(np.median(r_all))}  Q75={int(np.quantile(r_all,.75))}  '
          f'Q90={int(np.quantile(r_all,.9))}')

    # Fraction of candidates with matches > gold's:
    n_gt = np.asarray([r['n_ties_gt_gold'] for r in rows])
    print(f'\n  # candidates with matches STRICTLY GREATER than gold (per site):')
    print(f'    median={int(np.median(n_gt))}  mean={n_gt.mean():.1f}  Q90={int(np.quantile(n_gt,.9))}')


def report_junction_pool(cog_rows, shu_rows):
    """The proposed hybrid: global top-K=4 ∪ junction top-K=4 = up to K=8.
    Does the gold candidate survive?"""
    print(f'\n{"="*90}')
    print(f'  Hybrid proposal: global_topK=4 ∪ junction_topK=4 → up to K=8 combined')
    print(f'{"="*90}\n')
    for label, rows in [('cognate', cog_rows), ('shuffled', shu_rows)]:
        # For each site: is the gold candidate in top-4 global (per orient)? Or in top-4 junction (per orient)?
        # We already have rank_per_orient. Junction top-K needs a per-orient junction rank.
        # But if rank_per_orient ≤ 4 → survives global. If flank_start ≤ 5 (definition), it's a junction candidate,
        # so it survives the junction pool at rank 1 within-junction.
        n = len(rows)
        n_global4 = sum(1 for r in rows if r['rank_per_orient'] <= 4)
        # Junction pool: since gold IS by definition at flank_start ≤ 5, it's always in
        # the junction candidate set. Its rank in junction-only is 1 (it's defined as max at junction).
        # So the junction pool guarantees recall = 100%.
        # The COMBINED (union) has recall = 100% because gold is a junction candidate.
        n_combined = n
        print(f'  {label:<10}  n={n}')
        print(f'    Recall@K=4 (global proposal):     {100*n_global4/n:>5.1f}%')
        print(f'    Recall@K=8 (global 4 ∪ junction 4):  100.0%   (guaranteed by construction)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cog', default='/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/inference/durrant_cognate.jsonl')
    ap.add_argument('--shu', default='/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/inference/durrant_shuffled.jsonl')
    args = ap.parse_args()
    cog_rows = audit(args.cog, 'cognate')
    shu_rows = audit(args.shu, 'shuffled')
    report_ranks(cog_rows, 'COGNATE')
    report_ranks(shu_rows, 'SHUFFLED  (control — should look similar rank-wise since shuffled targets are also short DNA)')
    report_junction_pool(cog_rows, shu_rows)

    # Additional: compare gold match COUNT distributions between cognate and shuffled
    print(f'\n{"="*90}')
    print(f'  Gold junction candidate MATCH COUNT: cognate vs shuffled')
    print(f'{"="*90}\n')
    cog_m = np.asarray([r['gold_matches'] for r in cog_rows])
    shu_m = np.asarray([r['gold_matches'] for r in shu_rows])
    print(f'  cognate  gold_matches: median={int(np.median(cog_m))}  mean={cog_m.mean():.2f}')
    print(f'  shuffled gold_matches: median={int(np.median(shu_m))}  mean={shu_m.mean():.2f}')
    # AUROC using gold matches directly as score
    scores = np.concatenate([cog_m, shu_m])
    labels = np.concatenate([np.ones(len(cog_m)), np.zeros(len(shu_m))])
    order = np.argsort(-scores, kind='mergesort')
    y = labels[order].astype(bool)
    tps = np.cumsum(y); fps = np.cumsum(~y)
    tps = np.concatenate([[0], tps]); fps = np.concatenate([[0], fps])
    tpr = tps / max(1, tps[-1]); fpr = fps / max(1, fps[-1])
    au = float(np.trapezoid(tpr, fpr))
    print(f'  AUROC using gold junction matches as score: {au:.4f}')


if __name__ == '__main__':
    main()
