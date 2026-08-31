"""LEVEL 4 — empirical substitution-matrix pairing scorers, strict LOBO.

Question: Where does the 0.82 → higher gap live?
  Is it recoverable via a LEARNED substitution grammar
  (position/arm-specific), or is the raw ATCG match ceiling already the
  natural upper bound of raw-sequence pairing?

Design:
  For each held-out bRNA i, train on the other bRNAs' cognate + shuffled
  (RNA_base, DNA_base, position) triplets. Estimate log-enrichment:

      S(r, d, p) = log((P(d|r,p,cog) + eps) / (P(d|r,p,shuf) + eps))

  Score a test pair = sum over 11 positions of S(RNA[p], DNA[p]).
  Higher = more like cognate.

Four scorers under identical LOBO split:
  M0  binary match         : score = sum(1[RNA[p] == DNA[p]])                    (Level 3 A baseline)
  M1  global substitution  : one S(r,d) matrix, positions/arms pooled
  M2  arm-specific         : S_LTG(r,d), S_core(r,d), S_RTG(r,d)
  M3  position-specific    : S_p(r,d) for p in [0..10]

Metrics per scorer: pooled AUROC, macro AUROC, per-bRNA AUROC.

Also: dump the arm-specific matrix from an ALL-DATA fit for inspection —
useful for reading off wobble tendencies etc., but NOT used in the LOBO metric.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')
sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/scripts')

import numpy as np
from audit_level2c_gold_inject import find_ltg_position, load_ltg_specs

GOLD_JSONL = Path('/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/curated/is110_gold_v0.jsonl')

# Arm split: LTG = 0..6, core = 7..8, RTG = 9..10 (verified on WT IS621)
ARM_LTG = list(range(0, 7))
ARM_CORE = list(range(7, 9))
ARM_RTG = list(range(9, 11))
ARM_OF_POS = {}
for p in ARM_LTG: ARM_OF_POS[p] = 'LTG'
for p in ARM_CORE: ARM_OF_POS[p] = 'core'
for p in ARM_RTG: ARM_OF_POS[p] = 'RTG'

BASES = 'ACGT'
B2I = {b: i for i, b in enumerate(BASES)}
N_BASES = 4
EPS = 1.0  # pseudocount for smoothing


def load_curated():
    rows = [json.loads(l) for l in GOLD_JSONL.read_text().splitlines()]
    curated = []
    for r in rows:
        b = r.get('ortholog_id')
        brna_seq = r.get('brna_sequence')
        tgt = r.get('genome_target_11bp')
        spec = r.get('tbl_spec_11bp')
        if not b or not brna_seq or not tgt or not spec:
            continue
        if len(tgt) != 11 or len(spec) != 11:
            continue
        brna_dna = brna_seq.replace('U', 'T').replace('u', 't').upper()
        ltg_pos, _ = find_ltg_position(brna_dna, spec)
        if ltg_pos < 0 or ltg_pos + 11 > len(brna_dna):
            continue
        guide = brna_dna[ltg_pos:ltg_pos + 11]
        curated.append({'bR': b, 'guide': guide, 'cognate_target': tgt})
    return curated


def build_examples(curated, seed=42):
    """Returns list of (bR, guide, target, label). One cognate + one shuffled per row."""
    rng = random.Random(seed)
    all_targets_by_brna = defaultdict(list)
    for r in curated:
        all_targets_by_brna[r['bR']].append(r['cognate_target'])
    examples = []
    for r in curated:
        examples.append({'bR': r['bR'], 'guide': r['guide'],
                         'target': r['cognate_target'], 'label': 1})
        others = [b for b in all_targets_by_brna if b != r['bR']]
        if not others: continue
        picked_brna = rng.choice(others)
        picked_tgt = rng.choice(all_targets_by_brna[picked_brna])
        examples.append({'bR': r['bR'], 'guide': r['guide'],
                         'target': picked_tgt, 'label': 0})
    return examples


def count_pairs(examples, pos_bucket='global'):
    """Count (RNA_base, DNA_base) at pos_bucket, separately for cognate/shuffled.

    pos_bucket ∈ {'global', 'LTG', 'core', 'RTG'} OR an int in [0..10].
    Returns (cog_counts, shuf_counts): each is (N_BASES, N_BASES) uint32 array.
    """
    cog = np.zeros((N_BASES, N_BASES), dtype=np.int64)
    shu = np.zeros((N_BASES, N_BASES), dtype=np.int64)
    for ex in examples:
        g = ex['guide']; t = ex['target']
        if len(g) < 11 or len(t) < 11: continue
        for p in range(11):
            r = g[p]; d = t[p]
            if r not in B2I or d not in B2I: continue
            if pos_bucket == 'global':
                pass
            elif pos_bucket == 'LTG':
                if p not in ARM_LTG: continue
            elif pos_bucket == 'core':
                if p not in ARM_CORE: continue
            elif pos_bucket == 'RTG':
                if p not in ARM_RTG: continue
            elif isinstance(pos_bucket, int):
                if p != pos_bucket: continue
            else:
                raise ValueError(pos_bucket)
            if ex['label'] == 1:
                cog[B2I[r], B2I[d]] += 1
            else:
                shu[B2I[r], B2I[d]] += 1
    return cog, shu


def build_S(cog, shu, eps=EPS):
    """Log-enrichment matrix. Convert to conditional probabilities per row."""
    P_cog = (cog + eps) / (cog.sum(axis=1, keepdims=True) + eps * N_BASES)
    P_shu = (shu + eps) / (shu.sum(axis=1, keepdims=True) + eps * N_BASES)
    return np.log(P_cog / P_shu)


def score_pair(guide, target, scorer_type, params):
    """Score a pair with a substitution scorer."""
    s = 0.0
    for p in range(11):
        r = guide[p]; d = target[p]
        if r not in B2I or d not in B2I: continue
        ri = B2I[r]; di = B2I[d]
        if scorer_type == 'M0':
            s += 1.0 if ri == di else 0.0
        elif scorer_type == 'M1':
            s += params['S_global'][ri, di]
        elif scorer_type == 'M2':
            arm = ARM_OF_POS[p]
            s += params[f'S_{arm}'][ri, di]
        elif scorer_type == 'M3':
            s += params[f'S_pos{p}'][ri, di]
    return s


def _auroc_ties(scores, labels):
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg): return float('nan')
    P, N = len(pos), len(neg)
    all_s = np.concatenate([pos, neg])
    order = np.argsort(all_s, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    while i < len(all_s):
        j = i
        while j+1 < len(all_s) and all_s[order[j+1]] == all_s[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j+1): ranks[order[k]] = avg
        i = j + 1
    U = ranks[:P].sum() - P*(P+1)/2
    return float(U / (P * N))


def leave_one_bRNA_out(examples, scorer_type, min_test=4):
    bRs = sorted(set(ex['bR'] for ex in examples))
    per_bR = {}
    pooled_scores, pooled_labels = [], []
    for held in bRs:
        train = [ex for ex in examples if ex['bR'] != held]
        test = [ex for ex in examples if ex['bR'] == held]
        if len(test) < min_test:
            continue
        # Fit params on train
        params = {}
        if scorer_type == 'M0':
            pass
        elif scorer_type == 'M1':
            cog, shu = count_pairs(train, 'global')
            params['S_global'] = build_S(cog, shu)
        elif scorer_type == 'M2':
            for arm in ('LTG', 'core', 'RTG'):
                cog, shu = count_pairs(train, arm)
                params[f'S_{arm}'] = build_S(cog, shu)
        elif scorer_type == 'M3':
            for p in range(11):
                cog, shu = count_pairs(train, p)
                params[f'S_pos{p}'] = build_S(cog, shu)
        # Score test
        test_scores = [score_pair(ex['guide'], ex['target'], scorer_type, params) for ex in test]
        test_labels = [ex['label'] for ex in test]
        au = _auroc_ties(test_scores, test_labels)
        per_bR[held] = {'n_test': len(test), 'auroc': au}
        pooled_scores.extend(test_scores)
        pooled_labels.extend(test_labels)
    pooled_auroc = _auroc_ties(np.asarray(pooled_scores), np.asarray(pooled_labels))
    valid_aurocs = [v['auroc'] for v in per_bR.values() if not np.isnan(v['auroc'])]
    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else float('nan')
    return per_bR, pooled_auroc, macro_auroc


def dump_all_data_matrices(examples):
    """Fit substitution matrices on ALL data (not for LOBO metric — inspection only)."""
    print(f'\n{"="*100}')
    print(f'  Substitution matrices fit on ALL data (INSPECTION ONLY — not the LOBO metric)')
    print(f'{"="*100}')
    for bucket in ('LTG', 'core', 'RTG'):
        cog, shu = count_pairs(examples, bucket)
        S = build_S(cog, shu)
        print(f'\n  Arm {bucket}: log-enrichment S(r,d) rows=RNA, cols=DNA')
        print(f'      {"":<4}' + ''.join(f'{d:>7}' for d in BASES))
        for i, r in enumerate(BASES):
            row = ''.join(f'{S[i,j]:>+7.2f}' for j in range(N_BASES))
            print(f'      {r:<4}{row}')
        # Also print row sums / dominant entries
        # Dominant pairing per RNA base
        print(f'      dominant DNA per RNA base:')
        for i, r in enumerate(BASES):
            best_d = BASES[int(S[i].argmax())]
            print(f'        {r}->{best_d}  (S={S[i, S[i].argmax()]:+.2f})')


def main():
    curated = load_curated()
    print(f'[data] {len(curated)} usable rows across {len(set(r["bR"] for r in curated))} bRNAs')
    examples = build_examples(curated, seed=42)
    n_bR = len(set(ex['bR'] for ex in examples))
    print(f'[data] {len(examples)} examples ({sum(ex["label"]==1 for ex in examples)} cognate + '
          f'{sum(ex["label"]==0 for ex in examples)} shuffled), n_bRNAs={n_bR}')

    print(f'\n{"="*100}')
    print(f'  LEVEL 4 — Substitution matrix pairing scorers, strict leave-one-bRNA-out')
    print(f'{"="*100}\n')
    print(f'  {"scorer":<48} {"pooled":>7} {"macro":>7} {"n_folds":>7}')
    print('  ' + '-' * 74)
    results = {}
    for tag, desc in [
        ('M0', 'binary match (Level 3 A baseline)'),
        ('M1', 'global substitution S(r,d)'),
        ('M2', 'arm-specific S_LTG / S_core / S_RTG (r,d)'),
        ('M3', 'position-specific S_p(r,d), p in [0..10]'),
    ]:
        per_bR, pooled, macro = leave_one_bRNA_out(examples, tag)
        results[tag] = {'per_bR': per_bR, 'pooled': pooled, 'macro': macro, 'desc': desc}
        print(f'  {tag}  {desc:<44} {pooled:>7.4f} {macro:>7.4f} {len(per_bR):>7}')

    # Per-bRNA breakdown
    print(f'\n  Per-bRNA AUROC breakdown:')
    all_brs = sorted(set(bR for r in results.values() for bR in r['per_bR']))
    print(f'  {"scorer":<10} ', end='')
    for bR in all_brs:
        print(f'{bR[-14:]:>15}', end='')
    print()
    for tag, r in results.items():
        print(f'  {tag:<10} ', end='')
        for bR in all_brs:
            v = r['per_bR'].get(bR, {}).get('auroc', float('nan'))
            print(f'{v:>15.3f}' if not np.isnan(v) else f'{"n/a":>15}', end='')
        print()

    # Inspection: ALL-DATA arm matrix
    dump_all_data_matrices(examples)

    print(f'\n  Reference:')
    print(f'    Oracle spec-vs-target Hamming (annotation ceiling) = 0.9801')
    print(f'    Oracle raw junction matches                        = 0.8207')
    print(f'    V6 baseline / gold-injected                         = 0.5498 / 0.5527')
    print(f'    Level 3 A  (single-arm linear on match counts)     = 0.812 pooled / 0.768 macro')
    print(f'    Level 3 B  (dual-arm linear)                       = 0.807 / 0.745')
    print(f'    Level 3 C  (dual-arm tiny MLP)                     = 0.861 / 0.838')
    print(f'    Level 3 D  (position-specific linear)              = 0.779 / 0.743')

    print(f'\n  Interpretation:')
    print(f'    M0 ≈ M1 ≈ M2 ≈ M3        → substitution grammar adds nothing new; the 0.82 ceiling is real')
    print(f'    M2 ≫ M0                  → arm-level substitution tolerance is the missing piece')
    print(f'    M3 ≫ M2                  → position-level tolerance matters (needs more data to generalize)')
    print(f'    M3 ≪ M2                  → position-level overfits at 5-bRNA scale (consistent with Level 3 D)')


if __name__ == '__main__':
    main()
