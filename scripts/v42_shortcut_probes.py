"""V4.2 shortcut probe suite.

Purpose: verify that no simple summary of the model-visible inputs
(inputs.flank + inputs.noncoding_regions) can predict positive vs negative
with high AUROC. In other words: force RNA-DNA interaction to be the ONLY
discriminative signal.

Structural audit (Layer 1) — reported at end:
  - Presence-of-metadata differential between V4.2 positives and V4 negatives.
    Fields that ONLY appear on one side are 100% label-predictive if a loader
    accidentally reads them. This is a schema audit, not a training probe.

Feature-group probes (Layer 2) — the actual AUROC test:

  P1  layout_only
      Features: [flank_len, n_ncs, active_nc_len, ncs_max_len]
      Pass:  AUROC < 0.60
  P2  flank_marginal
      Features: [gc_flank, entropy_flank, 4 mononuc freqs, 16 dinuc freqs]
      Pass:  AUROC < 0.60
  P3  nc_active_marginal
      Features: same 21 stats on the active NC sequence
      Pass:  AUROC < 0.60
  P4  flank_plus_nc_marginal
      Features: concat of P2 + P3 (42 stats)
      Pass:  AUROC < 0.60
  P5  best_candidate_oracle
      Features: [max_match, max_identity, best_flank_start, best_nc_start,
                  best_L, best_orient_is_fwd] — computed from an oracle sweep
      Pass:  AUROC < 0.80  (positives DO have real pairing signal — a small
             gap here is EXPECTED. If very high, that's a shortcut.)

Each probe fits (a) sklearn LogisticRegression (baseline linear) and
(b) RandomForest (nonlinear, small). Report both AUROCs.

Shuffle-invariance test (bonus, run only on P4/P5):
  Within each positive bag, permute the flank across sites (keep NCs fixed).
  Rebuild P5 features on shuffled positives. If a probe is truly picking up
  NC↔flank interaction, its AUROC should DROP substantially. If it stays
  high, the probe was picking up marginal-only signal.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')

import numpy as np

DNA = 'ACGT'
BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
RC = {'A':'T','T':'A','C':'G','G':'C','N':'N'}


def seq_to_arr(s):
    return np.asarray([BASE_MAP.get(c, 4) for c in s.upper()], dtype=np.int8)


def rc_seq(s):
    return ''.join(RC.get(c,'N') for c in s[::-1].upper())


def _load(path, n_max, seed=42, positive=True):
    recs = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            recs.append(r)
            if len(recs) >= n_max * 3:
                break
    rng = random.Random(seed)
    rng.shuffle(recs)
    kept = []
    for r in recs:
        y = 1 if r['labels']['is_positive'] else 0
        if positive and y != 1: continue
        if (not positive) and y != 0: continue
        kept.append(r)
        if len(kept) >= n_max: break
    return kept


# ============  Feature extractors  ============

def feat_layout(rec):
    flank = rec['inputs']['flank']
    ncs = rec['inputs']['noncoding_regions']
    active = rec['labels'].get('active_noncoding_index', 0)
    active_nc_len = len(ncs[active]) if active < len(ncs) else 0
    ncs_max_len = max((len(x) for x in ncs), default=0)
    return np.asarray([len(flank), len(ncs), active_nc_len, ncs_max_len],
                       dtype=np.float32)


def _seq_stats(s):
    s = s.upper()
    n = len(s)
    if n == 0: return np.zeros(21, dtype=np.float32)
    # Mononuc freqs (4) — A, C, G, T (ignore N)
    counts = np.zeros(4, dtype=np.float64)
    for c in s:
        if c in BASE_MAP and BASE_MAP[c] < 4:
            counts[BASE_MAP[c]] += 1
    mono = counts / max(1, counts.sum())
    # GC
    gc = float(mono[1] + mono[2])
    # Shannon entropy over the 4 non-N bases
    p = mono[mono > 0]
    ent = float(-(p * np.log2(p)).sum()) if len(p) else 0.0
    # Dinuc freqs (16)
    dinuc = np.zeros(16, dtype=np.float64)
    for i in range(n - 1):
        a = s[i]; b = s[i + 1]
        if a in BASE_MAP and b in BASE_MAP and BASE_MAP[a] < 4 and BASE_MAP[b] < 4:
            dinuc[BASE_MAP[a] * 4 + BASE_MAP[b]] += 1
    dinuc = dinuc / max(1, dinuc.sum())
    return np.concatenate([[gc, ent], mono, dinuc]).astype(np.float32)


def feat_flank_marginal(rec):
    return _seq_stats(rec['inputs']['flank'])


def feat_nc_marginal(rec):
    ncs = rec['inputs']['noncoding_regions']
    active = rec['labels'].get('active_noncoding_index', 0)
    if active < len(ncs):
        return _seq_stats(ncs[active])
    return np.zeros(22, dtype=np.float32)


def feat_flank_plus_nc(rec):
    return np.concatenate([feat_flank_marginal(rec), feat_nc_marginal(rec)])


def _best_ungapped(nc, flank, lengths=(8, 10, 12, 14, 16)):
    """Oracle best ungapped alignment across L, orient, positions."""
    nc_a = seq_to_arr(nc); fk_a = seq_to_arr(flank); fk_rc_a = seq_to_arr(rc_seq(flank))
    best = {'matches': -1, 'L': -1, 'orient': 'fwd',
            'nc_start': 0, 'flank_start': 0}
    for L in lengths:
        if len(nc_a) < L or len(fk_a) < L: continue
        nc_win = np.lib.stride_tricks.sliding_window_view(nc_a, L)
        a_oh = np.eye(5, dtype=np.int8)[nc_win]
        for orient, fw in (('fwd', fk_a), ('rc', fk_rc_a)):
            fw_win = np.lib.stride_tricks.sliding_window_view(fw, L)
            b_oh = np.eye(5, dtype=np.int8)[fw_win]
            M = np.einsum('nlc,mlc->nm', a_oh, b_oh)
            idx = np.unravel_index(np.argmax(M), M.shape)
            m = int(M[idx])
            if m > best['matches']:
                fs = int(idx[1]) if orient == 'fwd' else len(fk_a) - int(idx[1]) - L
                best = {'matches': m, 'L': L, 'orient': orient,
                        'nc_start': int(idx[0]), 'flank_start': fs}
    return best


def feat_best_candidate(rec):
    ncs = rec['inputs']['noncoding_regions']
    active = rec['labels'].get('active_noncoding_index', 0)
    nc = ncs[active] if active < len(ncs) else ''
    flank = rec['inputs']['flank']
    if len(nc) < 8 or len(flank) < 8:
        return np.zeros(6, dtype=np.float32)
    b = _best_ungapped(nc, flank)
    return np.asarray([b['matches'], b['matches'] / max(1, b['L']),
                        b['flank_start'], b['nc_start'], b['L'],
                        1.0 if b['orient'] == 'fwd' else 0.0], dtype=np.float32)


# ============  Probes  ============

def train_and_eval(X, y, name, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    # Standardize
    mu = X_tr.mean(axis=0); sd = X_tr.std(axis=0); sd[sd == 0] = 1
    Xt = (X_tr - mu) / sd
    Xe = (X_te - mu) / sd
    lr = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr.fit(Xt, y_tr)
    au_lr = roc_auc_score(y_te, lr.predict_proba(Xe)[:, 1])
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed, n_jobs=1)
    rf.fit(X_tr, y_tr)
    au_rf = roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1])
    print(f'  {name:<28} AUROC (linear LR): {au_lr:.4f}   AUROC (RF): {au_rf:.4f}')
    return au_lr, au_rf


def build_feature_matrix(recs, extractor):
    X = []
    for r in recs:
        X.append(extractor(r))
    return np.stack(X, axis=0)


def structural_audit(pos_path, neg_path):
    """Verify presence-of-field differences between positives and negatives.
    A field that appears on one side only would be 100% label-predictive if
    a loader accidentally reads it."""
    pos_r = json.loads(next(open(pos_path)))
    neg_r = json.loads(next(open(neg_path)))

    def _keys(obj, prefix=''):
        out = set()
        if isinstance(obj, dict):
            for k, v in obj.items():
                out.add(prefix + k)
                out |= _keys(v, prefix + k + '.')
        return out
    pk = _keys(pos_r); nk = _keys(neg_r)
    only_pos = pk - nk
    only_neg = nk - pk
    print(f'\n  Fields ONLY on positives: {sorted(only_pos)}')
    print(f'  Fields ONLY on negatives: {sorted(only_neg)}')
    print(f'\n  These MUST NOT be present in what the model reads (inputs.*).')
    input_pos = set(pos_r.get('inputs', {}).keys())
    input_neg = set(neg_r.get('inputs', {}).keys())
    print(f'  inputs.* on positives : {sorted(input_pos)}')
    print(f'  inputs.* on negatives : {sorted(input_neg)}')
    if input_pos == input_neg:
        print(f'  → inputs.* schema MATCHES. No structural leakage via inputs.')
    else:
        print(f'  ! inputs.* differs between pos/neg — STRUCTURAL LEAKAGE risk')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pos', default='/global/scratch/users/kh36969/DL_novel_guide_editor/data/positives_v42.jsonl')
    ap.add_argument('--neg', default='/global/scratch/users/kh36969/DL_novel_guide_editor/data/negatives_v4.jsonl')
    ap.add_argument('--n-per-class', type=int, default=5000)
    args = ap.parse_args()

    print(f'\n{"="*100}')
    print(f'  V4.2 SHORTCUT PROBE SUITE')
    print(f'{"="*100}')

    print(f'\n[layer 1] structural audit')
    structural_audit(args.pos, args.neg)

    print(f'\n[load] {args.n_per_class} positives from {args.pos}')
    pos_recs = _load(args.pos, args.n_per_class, seed=42, positive=True)
    print(f'[load] {args.n_per_class} negatives from {args.neg}')
    neg_recs = _load(args.neg, args.n_per_class, seed=42, positive=False)
    print(f'[loaded] pos={len(pos_recs)}  neg={len(neg_recs)}')

    y_pos = np.ones(len(pos_recs), dtype=np.int64)
    y_neg = np.zeros(len(neg_recs), dtype=np.int64)
    y = np.concatenate([y_pos, y_neg])
    recs = pos_recs + neg_recs

    print(f'\n[layer 2] probe AUROCs (target: <0.60 for marginal probes)')
    print(f'  {"probe":<28} results')
    print(f'  ' + '-' * 78)

    probes = [
        ('P1 layout_only',            feat_layout,          0.60),
        ('P2 flank_marginal',          feat_flank_marginal,  0.60),
        ('P3 nc_active_marginal',      feat_nc_marginal,     0.60),
        ('P4 flank_plus_nc_marginal',  feat_flank_plus_nc,   0.60),
        ('P5 best_candidate_oracle',   feat_best_candidate,  0.80),
    ]
    results = {}
    for name, extractor, threshold in probes:
        print(f'\n  extracting features for {name} ...', flush=True)
        X = build_feature_matrix(recs, extractor)
        print(f'    shape {X.shape}', flush=True)
        au_lr, au_rf = train_and_eval(X, y, name)
        results[name] = (au_lr, au_rf, threshold)

    print(f'\n{"="*100}')
    print(f'  VERDICT')
    print(f'{"="*100}\n')
    fail = 0
    for name, (au_lr, au_rf, thr) in results.items():
        au_max = max(au_lr, au_rf)
        verdict = 'PASS' if au_max < thr else 'FAIL'
        marker = '  ' if verdict == 'PASS' else '⚠️'
        print(f'  {marker} {name:<28} best AUROC={au_max:.4f}  threshold <{thr}   {verdict}')
        if verdict == 'FAIL': fail += 1
    print(f'\n  {fail} probe(s) fail. ' +
          ('These features are ACTIVE shortcuts.' if fail else 'No marginal shortcuts detected.'))


if __name__ == '__main__':
    main()
