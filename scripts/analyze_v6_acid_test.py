"""Discovery-oriented analysis of the V6 acid-test 6×6 matrix output.

Reads the JSONL from test_c_cross_family_matrix.py and reports for each family:

  1. Level-1 (cognate pairing sensitivity for IS110):
     - Δ_real distribution (Q10/Q25/median/Q75/Q90/mean)
     - P(Δ_real > 0)
     - Native-vs-swap AUROC (per family)

  2. Score distribution (specificity for controls):
     - median, Q75, Q90, P(score > 0.5 / 0.7 / 0.8)

  3. High-confidence subset analysis (discovery):
     - Top 10% and top 20% by native score, per family
     - Their score AND Δ_real distributions

Definitions:
  For each bag (unique tnp_id) with nc_family = i:
    native_score = final_score in cell (nc=i, flank=i)
    swap_score   = median final_score across flank donors j != i (5 samples per bag)
    Δ_real       = native_score − swap_score

  Native-vs-swap AUROC per family i:
    pooled: native scores (label=1) + swap scores (label=0), same bags
    ↓ measures whether within-family, cognate consistently beats mismatched flanks
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

FAMILIES = ('IS110', 'IS30', 'IS903', 'IS10-R', 'ISLdl1', 'ISAjo2')


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--jsonl', required=True,
                    help='Output JSONL from test_c_cross_family_matrix.py')
    p.add_argument('--label', default='V6',
                    help='Model label for table headers.')
    args = p.parse_args()

    rows = []
    with open(args.jsonl) as f:
        for line in f:
            rows.append(json.loads(line))

    # Group by (tnp_id) → collect (nc_family, flank_family) -> logit (base_logit, final_score).
    per_bag = defaultdict(dict)
    for r in rows:
        per_bag[r['tnp_id']][(r['nc_family'], r['flank_family'])] = {
            'base_logit': r['base_logit'],
            'base_score': r['base_score'],
            'final_score': r['final_score'],
            'logit': r['logit'],
        }

    # Now compute per-bag native/swap/Δreal, grouped by NC family (bag's own family)
    fam_bags = defaultdict(list)   # fam -> list of dicts
    for tnp, cells in per_bag.items():
        nc_fam = list(cells.keys())[0][0]  # cells keyed by (nc, flank); nc is same for all
        native = cells.get((nc_fam, nc_fam))
        if native is None:
            continue
        swap_scores = [c['final_score'] for (nc, fk), c in cells.items() if fk != nc_fam]
        swap_logits = [c['logit']       for (nc, fk), c in cells.items() if fk != nc_fam]
        if not swap_scores:
            continue
        fam_bags[nc_fam].append({
            'tnp_id': tnp,
            'native_score': native['final_score'],
            'swap_score_med': float(np.median(swap_scores)),
            'swap_scores_all': swap_scores,
            'delta_real_score': native['final_score'] - float(np.median(swap_scores)),
            'delta_real_logit': native['logit']       - float(np.median(swap_logits)),
            'native_logit': native['logit'],
        })

    print(f'\n{"="*100}')
    print(f'  {args.label} — Level-1 discovery-oriented metrics per family')
    print(f'{"="*100}')

    # === Level 1: Δ_real distribution + native-vs-swap AUROC ===
    print(f'\n  Δ_real (logit units, native_flank vs median off-family swap) per family:')
    print(f'  {"family":<10} {"n":>4}  {"mean":>7} {"med":>7} '
          f'{"Q10":>7} {"Q25":>7} {"Q75":>7} {"Q90":>7}   {"P(>0)":>7}   {"AUROC":>7}')
    print(f'  {"-"*10} {"-"*4}  {"-"*7} {"-"*7} {"-"*7} {"-"*7} {"-"*7} {"-"*7}   {"-"*7}   {"-"*7}')
    for fam in FAMILIES:
        bags = fam_bags.get(fam, [])
        if not bags:
            continue
        d = np.asarray([b['delta_real_logit'] for b in bags])
        native_l = np.asarray([b['native_logit'] for b in bags])
        # Native-vs-swap AUROC: pool native logits (label=1) + all swap logits (label=0)
        native_pool, swap_pool = [], []
        for b in bags:
            native_pool.append(b['native_logit'])
            for s in [c['logit'] for (nc, fk), c in per_bag[b['tnp_id']].items() if fk != fam]:
                swap_pool.append(s)
        pool_s = np.concatenate([np.asarray(native_pool), np.asarray(swap_pool)])
        pool_y = np.concatenate([np.ones(len(native_pool), dtype=bool),
                                    np.zeros(len(swap_pool), dtype=bool)])
        auroc = _auroc(pool_s, pool_y)
        q = np.quantile(d, [.10, .25, .5, .75, .90])
        print(f'  {fam:<10} {len(bags):>4}  {d.mean():>+7.3f} {q[2]:>+7.3f} '
              f'{q[0]:>+7.3f} {q[1]:>+7.3f} {q[3]:>+7.3f} {q[4]:>+7.3f}   '
              f'{(d > 0).mean():>7.3f}   {auroc:>7.4f}')

    # === Level 2: native-score specificity distribution ===
    print(f'\n{"="*100}')
    print(f'  {args.label} — Level-2 specificity: native-score distribution per family')
    print(f'{"="*100}')
    print(f'\n  {"family":<10} {"n":>4}  '
          f'{"mean":>6} {"med":>6} {"Q75":>6} {"Q90":>6}   '
          f'{"P>0.5":>7} {"P>0.7":>7} {"P>0.8":>7}')
    print(f'  {"-"*10} {"-"*4}  {"-"*6} {"-"*6} {"-"*6} {"-"*6}   '
          f'{"-"*7} {"-"*7} {"-"*7}')
    for fam in FAMILIES:
        bags = fam_bags.get(fam, [])
        if not bags:
            continue
        s = np.asarray([b['native_score'] for b in bags])
        q = np.quantile(s, [.5, .75, .9])
        print(f'  {fam:<10} {len(bags):>4}  '
              f'{s.mean():>6.3f} {q[0]:>6.3f} {q[1]:>6.3f} {q[2]:>6.3f}   '
              f'{(s > 0.5).mean():>7.3f} {(s > 0.7).mean():>7.3f} {(s > 0.8).mean():>7.3f}')

    # === Discovery subset analysis: top-quantile IS110 (and others) ===
    print(f'\n{"="*100}')
    print(f'  {args.label} — Discovery subsets: top-K native-score bags per family')
    print(f'{"="*100}')
    for pct, label in ((0.20, 'top 20%'), (0.10, 'top 10%')):
        print(f'\n  {label} of each family by native score:')
        print(f'  {"family":<10} {"n_top":>5}  '
              f'{"med(score)":>10} {"med(Δreal)":>10} {"P(Δ>0)":>7}')
        print(f'  {"-"*10} {"-"*5}  {"-"*10} {"-"*10} {"-"*7}')
        for fam in FAMILIES:
            bags = fam_bags.get(fam, [])
            if not bags:
                continue
            n_top = max(1, int(np.ceil(len(bags) * pct)))
            top = sorted(bags, key=lambda b: -b['native_score'])[:n_top]
            s = np.asarray([b['native_score'] for b in top])
            d = np.asarray([b['delta_real_logit'] for b in top])
            print(f'  {fam:<10} {len(top):>5}  '
                  f'{float(np.median(s)):>10.3f} {float(np.median(d)):>+10.3f} '
                  f'{float((d > 0).mean()):>7.3f}')

    # === Enrichment view: what fraction of "high-score" bags are IS110? ===
    print(f'\n{"="*100}')
    print(f'  {args.label} — Enrichment: family composition among high-score bags')
    print(f'{"="*100}')
    all_bags = []
    for fam, bags in fam_bags.items():
        for b in bags:
            all_bags.append({'family': fam, **b})
    n_total = len(all_bags)
    # Total IS110 fraction
    n_is110 = sum(1 for b in all_bags if b['family'] == 'IS110')
    baseline = n_is110 / n_total if n_total else 0
    print(f'\n  Baseline IS110 fraction: {n_is110}/{n_total} = {baseline:.3f}')
    print(f'  {"threshold":<12} {"n_bags":>7} {"n_IS110":>8} {"%_IS110":>8}  '
          f'{"enrichment":>11}')
    for th in (0.5, 0.6, 0.7, 0.8):
        subset = [b for b in all_bags if b['native_score'] > th]
        n = len(subset)
        n_pos = sum(1 for b in subset if b['family'] == 'IS110')
        if n > 0:
            frac = n_pos / n
            enrich = frac / baseline if baseline > 0 else float('nan')
        else:
            frac = float('nan'); enrich = float('nan')
        print(f'  score > {th:<5}  {n:>7} {n_pos:>8} {frac:>8.3f}   '
              f'{enrich:>10.2f}×')


if __name__ == '__main__':
    main()
