"""LEVEL 3 — IS110-specific diagnostic on IS110_gold_v0.

Runs four hand-feature classifiers with strict leave-one-bRNA-out:

  A  single-arm linear      : total_matches, total_mismatches
  B  dual-arm linear        : LTG_matches, core_matches, RTG_matches (+ mismatches)
  C  dual-arm tiny MLP      : same features as B, 2-layer with tanh
  D  position-specific linear : per-position match vector m[0..10]

Question this answers:
  Is explicit LTG+core+RTG decomposition necessary to explain IS110 gold,
  or does aggregate/position-agnostic already do the job?

  This ONLY informs the IS110-expert branch of a future V7. The generic
  topology-agnostic branch is out of scope here.

Data:
  IS110_gold_v0.jsonl → 341 (bRNA, cognate_11bp_target) pairs.
  Per row we build:
    Cognate  : (bRNA_guide, this_row_genome_target_11bp)
    Shuffled : (bRNA_guide, another_row_genome_target_11bp from a DIFFERENT bRNA)
  where bRNA_guide := bRNA_sequence[ltg_pos:ltg_pos+11] with ltg_pos found by
  max sliding L=11 fwd match to TBL_spec.

Split:
  Leave-one-bRNA-out. All examples using bRNA_i in test; all other bRNAs
  form the training set. Report:
    per-fold AUROC
    macro AUROC (mean across bRNAs, unweighted)
    pooled AUROC (all folds concatenated)
    paired P(s_cog > s_shuf) within held-out bag
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from audit_level2c_gold_inject import find_ltg_position, load_ltg_specs

GOLD_JSONL = Path('/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/curated/is110_gold_v0.jsonl')

# Arm split: IS621 has core at positions 7-8 for TBL target 11bp (verified from
# WT bRNA specificity `ATCGGGCCTAC` — 'CT' at positions 7-8). Default LTG =
# positions 0-6 (7 bp), core = 7-8 (2 bp), RTG = 9-10 (2 bp).
# Try also the alternate 5+2+4 split for comparison.
DEFAULT_ARM = 'ltg7_core2_rtg2'   # positions [0..6] | [7..8] | [9..10]
ALT_ARM = 'ltg5_core2_rtg4'       # positions [0..4] | [5..6] | [7..10]


def arm_split(name):
    if name == 'ltg7_core2_rtg2':
        return list(range(0, 7)), list(range(7, 9)), list(range(9, 11))
    if name == 'ltg5_core2_rtg4':
        return list(range(0, 5)), list(range(5, 7)), list(range(7, 11))
    raise ValueError(name)


def _match_vector(brna_guide: str, target: str) -> np.ndarray:
    """0/1 length-11 vector; N or length mismatch → 0."""
    v = np.zeros(11, dtype=np.float32)
    if len(brna_guide) < 11 or len(target) < 11:
        return v
    for i in range(11):
        a = brna_guide[i].upper(); b = target[i].upper()
        v[i] = 1.0 if (a == b and a != 'N') else 0.0
    return v


def _features(brna_guide: str, target: str, arm: str) -> dict:
    m = _match_vector(brna_guide, target)
    ltg_idx, core_idx, rtg_idx = arm_split(arm)
    return {
        'position_matches': m,                              # (11,)
        'total_matches': float(m.sum()),
        'total_mismatches': float(11 - m.sum()),
        'ltg_matches': float(m[ltg_idx].sum()),
        'ltg_mismatches': float(len(ltg_idx) - m[ltg_idx].sum()),
        'core_matches': float(m[core_idx].sum()),
        'core_mismatches': float(len(core_idx) - m[core_idx].sum()),
        'rtg_matches': float(m[rtg_idx].sum()),
        'rtg_mismatches': float(len(rtg_idx) - m[rtg_idx].sum()),
    }


def build_dataset(seed=42, arm=DEFAULT_ARM):
    rows = [json.loads(l) for l in GOLD_JSONL.read_text().splitlines()]
    ltg_specs = load_ltg_specs()

    # Build (bRNA, guide_seq, genome_target_11bp) per row
    curated = []
    for r in rows:
        b = r.get('ortholog_id')
        brna_seq = r.get('brna_sequence')  # RNA form (U)
        tgt = r.get('genome_target_11bp')
        spec = r.get('tbl_spec_11bp')
        if not b or not brna_seq or not tgt or not spec:
            continue
        if len(tgt) != 11 or len(spec) != 11:
            continue
        # RNA → DNA
        brna_dna = brna_seq.replace('U', 'T').replace('u', 't').upper()
        ltg_pos, _ = find_ltg_position(brna_dna, spec)
        if ltg_pos < 0 or ltg_pos + 11 > len(brna_dna):
            continue
        guide = brna_dna[ltg_pos:ltg_pos + 11]
        curated.append({'bR': b, 'guide': guide, 'cognate_target': tgt})

    print(f'[data] {len(curated)} usable rows across '
          f'{len(set(r["bR"] for r in curated))} bRNAs')

    # Cognate examples: use each row's own guide + cognate target
    cog_examples = []
    for r in curated:
        f = _features(r['guide'], r['cognate_target'], arm)
        cog_examples.append({'bR': r['bR'], 'label': 1, **f})

    # Shuffled examples: for each row, pair its guide with a genome_target from
    # a DIFFERENT bRNA (chosen randomly). Same seed → reproducible.
    rng = random.Random(seed)
    all_targets_by_brna = defaultdict(list)
    for r in curated:
        all_targets_by_brna[r['bR']].append(r['cognate_target'])
    shu_examples = []
    for r in curated:
        others_brnas = [b for b in all_targets_by_brna if b != r['bR']]
        if not others_brnas:
            continue
        pick_brna = rng.choice(others_brnas)
        pick_target = rng.choice(all_targets_by_brna[pick_brna])
        f = _features(r['guide'], pick_target, arm)
        shu_examples.append({'bR': r['bR'], 'label': 0, **f})

    print(f'[data] cognate={len(cog_examples)}  shuffled={len(shu_examples)}')
    return cog_examples + shu_examples


def to_matrices(examples, feature_set):
    X = []
    y = []
    b = []
    for ex in examples:
        if feature_set == 'A':   # single-arm linear
            xi = [ex['total_matches'], ex['total_mismatches']]
        elif feature_set == 'B':  # dual-arm linear
            xi = [ex['ltg_matches'], ex['ltg_mismatches'],
                  ex['core_matches'], ex['core_mismatches'],
                  ex['rtg_matches'], ex['rtg_mismatches']]
        elif feature_set == 'D':  # position-specific linear
            xi = list(ex['position_matches'])
        else:
            raise ValueError(feature_set)
        X.append(xi)
        y.append(ex['label'])
        b.append(ex['bR'])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), np.asarray(b)


class LinearModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)
    def forward(self, x):
        return self.lin(x).squeeze(-1)


class TinyMLP(nn.Module):
    def __init__(self, in_dim, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_and_predict(X_train, y_train, X_test, model_cls, epochs=200, lr=0.05,
                      wd=1e-3, verbose=False):
    """All configs use sklearn — much faster on CPU for tiny data."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    if model_cls is LinearModel:
        clf = LogisticRegression(C=1.0 / max(wd, 1e-9), max_iter=1000,
                                  solver='lbfgs')
    elif model_cls is TinyMLP:
        clf = MLPClassifier(hidden_layer_sizes=(16, 16), activation='tanh',
                             solver='lbfgs', alpha=wd, max_iter=500,
                             random_state=42)
    else:
        raise ValueError(model_cls)
    clf.fit(X_train, y_train.astype(int))
    return clf.predict_proba(X_test)[:, 1]


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


def leave_one_brna_out(all_examples, feature_set, model_name, min_test=4):
    """Return {bR: AUROC} + pooled AUROC + macro AUROC."""
    X, y, b = to_matrices(all_examples, feature_set)
    bRs = sorted(set(b))
    per_bR = {}
    pooled_scores, pooled_labels = [], []
    for held in bRs:
        train_mask = b != held
        test_mask = b == held
        n_test = int(test_mask.sum())
        if n_test < min_test:
            continue
        # Model
        if model_name == 'linear':
            cls = LinearModel
        elif model_name == 'mlp':
            cls = TinyMLP
        else:
            raise ValueError(model_name)
        pred = train_and_predict(X[train_mask], y[train_mask], X[test_mask], cls)
        au = _auroc_ties(pred, y[test_mask])
        per_bR[held] = {'n_test': n_test, 'auroc': au}
        pooled_scores.extend(list(pred))
        pooled_labels.extend(list(y[test_mask]))
    pooled_auroc = _auroc_ties(np.asarray(pooled_scores), np.asarray(pooled_labels))
    macro_auroc = float(np.mean([v['auroc'] for v in per_bR.values() if not np.isnan(v['auroc'])]))
    return per_bR, pooled_auroc, macro_auroc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default=DEFAULT_ARM,
                    choices=[DEFAULT_ARM, ALT_ARM])
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    examples = build_dataset(seed=args.seed, arm=args.arm)
    print(f'[arm split] {args.arm}: '
          f'LTG={arm_split(args.arm)[0]}, core={arm_split(args.arm)[1]}, RTG={arm_split(args.arm)[2]}')

    print(f'\n{"="*100}')
    print(f'  LEVEL 3 — Leave-one-bRNA-out on IS110_gold_v0  (n_bRNAs = '
          f'{len(set(e["bR"] for e in examples))})')
    print(f'{"="*100}\n')

    configs = [
        ('A  single-arm linear (aggregate)', 'A', 'linear'),
        ('B  dual-arm linear (LTG+core+RTG)', 'B', 'linear'),
        ('C  dual-arm tiny MLP  (same features as B, hidden=16)', 'B', 'mlp'),
        ('D  position-specific linear (11 weights)', 'D', 'linear'),
    ]
    print(f'  {"model":<50} {"pooled":>7} {"macro":>7} {"n_folds":>7}')
    print('  ' + '-' * 78)
    results = {}
    for label, feat, mtype in configs:
        per_bR, pooled, macro = leave_one_brna_out(examples, feat, mtype)
        results[label] = per_bR
        print(f'  {label:<50} {pooled:>7.4f} {macro:>7.4f} {len(per_bR):>7}')

    print(f'\n  Per-bRNA AUROC breakdown:')
    print(f'  {"model":<50} ', end='')
    all_brs = sorted(set(bR for res in results.values() for bR in res))
    for bR in all_brs:
        print(f'{bR[-14:]:>15}', end='')
    print()
    for label, per_bR in results.items():
        print(f'  {label:<50} ', end='')
        for bR in all_brs:
            v = per_bR.get(bR, {}).get('auroc', float('nan'))
            print(f'{v:>15.3f}' if not np.isnan(v) else f'{"n/a":>15}', end='')
        print()

    print(f'\n  Reference (from earlier tests):')
    print(f'    Oracle Hamming spec-vs-target AUROC (pooled) = 0.9801')
    print(f'    Oracle raw junction matches AUROC (pooled)   = 0.8207')
    print(f'    V6 on Durrant baseline                        = 0.5498')
    print(f'    V6 with gold candidate injected               = 0.5527')

    print(f'\n  Interpretation guide (for IS110-EXPERT branch only, not V7 generic detector):')
    print(f'    A ≈ B ≈ C     → no arm decomposition needed for IS110 expert')
    print(f'    A ≪ B ≈ C     → explicit LTG/core/RTG decomposition IS the key for IS110 expert')
    print(f'    A ≪ B ≪ C     → additionally requires nonlinear arm interaction')
    print(f'    D ≫ A         → position-specific weights carry the signal (some positions dominant)')


if __name__ == '__main__':
    main()
