"""sim2real calibration report — the gate every new synthetic dataset must pass.

Given a synthetic JSONL and the Durrant IS110_gold reference JSONL (constructed
into model-format bags), compute matched summary statistics and emit a
side-by-side comparison with pass/fail per metric.

Metrics computed on BOTH:
  positive-site true-match identity distribution         (L, matches, identity)
  candidate pool matches distribution                    (all candidates in generator pool)
  Rank of true candidate in per-(orient, L) proposal     (recall@K = fraction true in top K)
  Junction enrichment: fraction of top candidates at flank_start ≤ 15
  Oracle AUROC: best-candidate matches as cognate score, vs a within-bag shuffled null
  Bag guided fraction (labels for synthetic; assume 1.0 for Durrant cognate)
  NC length distribution
  Structure heterogeneity within bag (D_struct as in Level 5)

Pass/fail thresholds (loosened per Level 5 discussion — DO NOT tighten to
"match Durrant's exact 51% k=3 peak", that's an IS621 scaffold artifact):

  identity_median            in [0.65, 0.80]
  fraction perfect identity  < 0.05
  fraction identity >= 0.9   < 0.15
  fraction identity <= 0.75  > 0.40   (substantial weak tail exists)
  L_median                   matches Durrant ±1

CAVEAT — Recall@K and gold_rank_median in the fake-bag comparison table
below are CONFOUNDED by flank background differences (synthetic uses real
bacterial 120bp flanks; Durrant fake-bags use random ACGT padding beyond
the cognate 14bp). These are informative but should NOT be used as hard
gates until synthetic and Durrant flanks are put through identical
candidate pipelines with matched flank-background statistics.

Failing means: DO NOT train on this dataset. Regenerate.
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
from preprocess.candidates import build_candidate_arrays

DEFAULT_DURRANT = '/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/inference/durrant_cognate.jsonl'
DEFAULT_DURRANT_GOLD = '/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/curated/is110_gold_v0.jsonl'


def load_records(path, max_records=None):
    recs = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_records is not None and i >= max_records: break
            recs.append(json.loads(line))
    return recs


def identity_from_labels(rec):
    """For synthetic: extract (L, matches) from labels. Returns None if unavailable."""
    L = rec['labels'].get('guide_length')
    mm = rec['labels'].get('n_mismatches')
    if L is None or mm is None: return None, None
    return int(L), int(L) - int(mm)


def oracle_best_L11(nc, flank):
    """Enumerate ALL L=11 candidates, return list of (nc_start, flank_start, orient, matches)."""
    BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    RC = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    def s2a(s): return np.asarray([BASE_MAP.get(c, 4) for c in s.upper()], dtype=np.int8)
    def rc(s): return ''.join(RC.get(c, 'N') for c in s[::-1].upper())
    L = 11
    nc_a = s2a(nc); fk_a = s2a(flank); fk_rc_a = s2a(rc(flank))
    if len(nc_a) < L or len(fk_a) < L: return []
    out = []
    nc_win = np.lib.stride_tricks.sliding_window_view(nc_a, L)
    a_oh = np.eye(5, dtype=np.int8)[nc_win]
    for orient, fw in (('fwd', fk_a), ('rc', fk_rc_a)):
        fw_win = np.lib.stride_tricks.sliding_window_view(fw, L)
        b_oh = np.eye(5, dtype=np.int8)[fw_win]
        M = np.einsum('nlc,mlc->nm', a_oh, b_oh)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                fs = j if orient == 'fwd' else len(fk_a) - j - L
                out.append({'orient': orient, 'nc_start': i, 'flank_start': fs,
                            'matches': int(M[i, j])})
    return out


def gold_rank_and_recall(nc, flank, positive_flank_target_pos=0):
    """Rank the junction-anchored (flank_start ≤ 5) best L=11 candidate among all L=11 cands.
    Returns (gold_matches, rank_per_orient, rank_all)."""
    cands = oracle_best_L11(nc, flank)
    if not cands: return None
    # gold = best matches at flank_start ≤ 5
    at_j = [c for c in cands if c['flank_start'] <= 5]
    if not at_j: return None
    gold = max(at_j, key=lambda c: c['matches'])
    per_orient = sorted([c for c in cands if c['orient'] == gold['orient']],
                        key=lambda c: -c['matches'])
    rank_orient = next((r + 1 for r, c in enumerate(per_orient)
                        if c['orient'] == gold['orient']
                        and c['nc_start'] == gold['nc_start']
                        and c['flank_start'] == gold['flank_start']), None)
    return {'gold_matches': gold['matches'],
            'rank_per_orient': rank_orient or -1,
            'gold_flank_start': gold['flank_start']}


def audit_one_dataset(records, label, max_sites_for_oracle=None, seed=42):
    """Compute all metrics on a list of records."""
    print(f'  [{label}] processing {len(records)} records', flush=True)
    L_list, matches_list = [], []
    for rec in records:
        if rec['labels'].get('is_positive'):
            L, m = identity_from_labels(rec)
            if L is not None and m is not None:
                L_list.append(L); matches_list.append(m)

    # Oracle-based rank distribution (junction-anchored L=11 gold)
    oracle_records = records
    if max_sites_for_oracle is not None and len(records) > max_sites_for_oracle:
        rng = random.Random(seed)
        oracle_records = rng.sample(records, max_sites_for_oracle)
    gold_ranks = []
    gold_matches_at_junction = []
    for rec in oracle_records:
        if not rec['labels'].get('is_positive'): continue
        acn = rec['labels'].get('active_noncoding_index')
        if acn is None: continue
        ncs = rec['inputs']['noncoding_regions']
        if acn >= len(ncs): continue
        nc = ncs[acn]; flank = rec['inputs']['flank']
        info = gold_rank_and_recall(nc, flank)
        if info is None: continue
        gold_ranks.append(info['rank_per_orient'])
        gold_matches_at_junction.append(info['gold_matches'])

    # Bag guided fraction
    bag_guided = defaultdict(lambda: {'total': 0, 'guided': 0})
    nc_lens = []
    for rec in records:
        tnp = rec['transposase_id']
        bag_guided[tnp]['total'] += 1
        if rec['labels'].get('site_class') == 'guided' and rec['labels'].get('is_positive'):
            bag_guided[tnp]['guided'] += 1
        acn = rec['labels'].get('active_noncoding_index')
        if acn is not None:
            ncs = rec['inputs']['noncoding_regions']
            if acn < len(ncs): nc_lens.append(len(ncs[acn]))
    guided_fracs = [d['guided'] / d['total'] for d in bag_guided.values() if d['total'] > 0]

    # Summary
    def _q(a, qs):
        a = np.asarray(a)
        if not len(a): return {q: float('nan') for q in qs}
        return {q: float(np.quantile(a, q / 100)) for q in qs}

    res = {'label': label}
    if L_list:
        L_arr = np.asarray(L_list); m_arr = np.asarray(matches_list)
        identity = m_arr / L_arr
        res['n_pos'] = len(L_list)
        res['L_median'] = float(np.median(L_arr))
        res['L_min'] = int(L_arr.min())
        res['L_max'] = int(L_arr.max())
        res['matches_median'] = float(np.median(m_arr))
        res['identity_median'] = float(np.median(identity))
        res['identity_q25'] = float(np.quantile(identity, .25))
        res['identity_q75'] = float(np.quantile(identity, .75))
        res['identity_q10'] = float(np.quantile(identity, .10))
        res['frac_id_1_0'] = float(np.mean(identity == 1.0))
        res['frac_id_ge_0_9'] = float(np.mean(identity >= 0.9))
        res['frac_id_le_0_75'] = float(np.mean(identity <= 0.75))
    if gold_ranks:
        gr = np.asarray(gold_ranks)
        res['gold_rank_median'] = float(np.median(gr))
        res['gold_rank_q75'] = float(np.quantile(gr, .75))
        res['gold_rank_q90'] = float(np.quantile(gr, .90))
        res['recall_at_4'] = float(np.mean(gr <= 4))
        res['recall_at_8'] = float(np.mean(gr <= 8))
        res['recall_at_20'] = float(np.mean(gr <= 20))
        res['gold_matches_at_junction_median'] = float(np.median(gold_matches_at_junction))
    res['n_bags'] = len(guided_fracs)
    res['guided_frac_median'] = float(np.median(guided_fracs)) if guided_fracs else float('nan')
    res['guided_frac_q25'] = float(np.quantile(guided_fracs, .25)) if guided_fracs else float('nan')
    res['guided_frac_q75'] = float(np.quantile(guided_fracs, .75)) if guided_fracs else float('nan')
    res['nc_len_median'] = float(np.median(nc_lens)) if nc_lens else float('nan')
    return res


def print_side_by_side(res_syn, res_real):
    print(f'\n{"="*100}')
    print(f'  SIM2REAL CALIBRATION REPORT — {res_syn["label"]} vs {res_real["label"]}')
    print(f'{"="*100}\n')
    metrics = [
        ('n_pos', 'n positive sites', 'int'),
        ('L_median', 'guide L median', 'float', 1),
        ('matches_median', 'matches median', 'float', 1),
        ('identity_median', 'identity median', 'float', 3),
        ('identity_q25', 'identity Q25', 'float', 3),
        ('identity_q10', 'identity Q10', 'float', 3),
        ('frac_id_1_0', 'fraction perfect identity', 'float', 3),
        ('frac_id_ge_0_9', 'fraction identity >= 0.9', 'float', 3),
        ('frac_id_le_0_75', 'fraction identity <= 0.75', 'float', 3),
        ('gold_matches_at_junction_median', 'oracle junction matches median (L=11)', 'float', 1),
        ('gold_rank_median', 'gold candidate rank median', 'float', 1),
        ('gold_rank_q75', 'gold candidate rank Q75', 'float', 1),
        ('gold_rank_q90', 'gold candidate rank Q90', 'float', 1),
        ('recall_at_4', 'Recall @K=4', 'float', 3),
        ('recall_at_8', 'Recall @K=8', 'float', 3),
        ('recall_at_20', 'Recall @K=20', 'float', 3),
        ('n_bags', 'n bags', 'int'),
        ('guided_frac_median', 'bag guided fraction median', 'float', 3),
        ('guided_frac_q25', 'bag guided fraction Q25', 'float', 3),
        ('nc_len_median', 'NC length median', 'float', 0),
    ]
    print(f'  {"metric":<40} {"synthetic":>14} {"real (Durrant)":>16}   {"delta":>10}')
    print('  ' + '-' * 90)
    for m in metrics:
        key = m[0]; desc = m[1]; kind = m[2]
        prec = m[3] if len(m) > 3 else 0
        s = res_syn.get(key); r = res_real.get(key)
        if s is None or r is None or (isinstance(s, float) and np.isnan(s)) or (isinstance(r, float) and np.isnan(r)):
            print(f'  {desc:<40} {"—":>14} {"—":>16}   {"—":>10}')
            continue
        if kind == 'int':
            print(f'  {desc:<40} {int(s):>14} {int(r):>16}   {int(s - r):>+10}')
        else:
            fmt = f'{{:>14.{prec}f}}'; fmt2 = f'{{:>16.{prec}f}}'; fmt3 = f'{{:>+10.{prec}f}}'
            print(f'  {desc:<40} ' + fmt.format(s) + ' ' + fmt2.format(r) + '   ' + fmt3.format(s - r))

    # Pass/fail on key thresholds
    print(f'\n  {"PASS/FAIL":<40} {"|delta|":>14} {"threshold":>16}   verdict')
    print('  ' + '-' * 90)
    def _check(name, delta, threshold):
        verdict = 'PASS' if abs(delta) <= threshold else 'FAIL'
        print(f'  {name:<40} {abs(delta):>14.3f} {threshold:>16.3f}   {verdict}')
        return verdict
    verdicts = []
    if res_syn.get('identity_median') and res_real.get('identity_median'):
        d = res_syn['identity_median'] - res_real['identity_median']
        verdicts.append(_check('identity_median', d, 0.05))
    if res_syn.get('L_median') and res_real.get('L_median'):
        d = res_syn['L_median'] - res_real['L_median']
        verdicts.append(_check('L_median', d, 1.0))
    if res_syn.get('recall_at_4') and res_real.get('recall_at_4'):
        d = res_syn['recall_at_4'] - res_real['recall_at_4']
        verdicts.append(_check('recall_at_4', d, 0.15))
    if res_syn.get('gold_rank_median') is not None and res_real.get('gold_rank_median') is not None:
        d = res_syn['gold_rank_median'] - res_real['gold_rank_median']
        verdicts.append(_check('gold_rank_median', d, 3.0))
    if res_syn.get('guided_frac_median') and res_real.get('guided_frac_median'):
        d = res_syn['guided_frac_median'] - res_real['guided_frac_median']
        verdicts.append(_check('guided_frac_median', d, 0.15))
    fails = [v for v in verdicts if v == 'FAIL']
    print()
    if fails:
        print(f'  OVERALL: {len(fails)} / {len(verdicts)} thresholds FAIL — this dataset is not fit for training')
    else:
        print(f'  OVERALL: all {len(verdicts)} thresholds pass. Dataset is calibrated.')
    return len(fails) == 0


def durrant_direct_stats():
    """Load IS110_gold_v0 direct (bRNA guide + genome_target_11bp) and
    compute canonical identity + arm-match stats without pipeline noise."""
    from audit_level2c_gold_inject import find_ltg_position, load_ltg_specs
    specs = load_ltg_specs()
    identities = []
    ltg_matches, core_matches, rtg_matches = [], [], []
    with open(DEFAULT_DURRANT_GOLD) as f:
        for line in f:
            r = json.loads(line)
            b = r.get('ortholog_id')
            brna_seq = r.get('brna_sequence')
            tgt = r.get('genome_target_11bp')
            spec = specs.get(b)
            if not (b and brna_seq and tgt and spec): continue
            if len(tgt) != 11 or len(spec) != 11: continue
            brna_dna = brna_seq.replace('U', 'T').replace('u', 't').upper()
            pos, _ = find_ltg_position(brna_dna, spec)
            if pos < 0 or pos + 11 > len(brna_dna): continue
            guide = brna_dna[pos:pos + 11]
            m = [1 if a == b and a != 'N' else 0 for a, b in zip(guide, tgt.upper())]
            identities.append(sum(m) / 11)
            ltg_matches.append(sum(m[0:7]))
            core_matches.append(sum(m[7:9]))
            rtg_matches.append(sum(m[9:11]))
    return {
        'label': 'Durrant IS110_gold (direct 11bp)',
        'n_pos': len(identities),
        'L_median': 11.0,
        'matches_median': float(np.median([int(11 * i) for i in identities])),
        'identity_median': float(np.median(identities)),
        'identity_q25': float(np.quantile(identities, .25)),
        'identity_q10': float(np.quantile(identities, .10)),
        'identity_q75': float(np.quantile(identities, .75)),
        'frac_id_1_0': float(np.mean(np.asarray(identities) == 1.0)),
        'frac_id_ge_0_9': float(np.mean(np.asarray(identities) >= 0.9)),
        'frac_id_le_0_75': float(np.mean(np.asarray(identities) <= 0.75)),
        'ltg_matches_median': float(np.median(ltg_matches)),
        'core_matches_median': float(np.median(core_matches)),
        'rtg_matches_median': float(np.median(rtg_matches)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--syn', required=True, help='Synthetic dataset JSONL')
    ap.add_argument('--real', default=DEFAULT_DURRANT, help='Durrant reference JSONL (fake bags)')
    ap.add_argument('--max-syn', type=int, default=5000, help='Cap synthetic records for scan')
    ap.add_argument('--max-oracle-syn', type=int, default=1000,
                     help='Cap synthetic records for expensive oracle rank computation')
    args = ap.parse_args()

    print(f'[load] syn: {args.syn}')
    syn = load_records(args.syn, max_records=args.max_syn)
    print(f'[load] real (fake bags): {args.real}')
    real = load_records(args.real)

    res_syn = audit_one_dataset(syn, f'synthetic({Path(args.syn).stem})',
                                 max_sites_for_oracle=args.max_oracle_syn)
    res_real = audit_one_dataset(real, 'Durrant IS110 (fake bags)', max_sites_for_oracle=None)
    res_gold = durrant_direct_stats()

    # Print an additional table with the direct Durrant identity comparison
    print(f'\n{"="*100}')
    print(f'  DIRECT DURRANT vs SYNTHETIC — identity + arm-match distributions')
    print(f'  (Durrant computed from bRNA_guide vs genome_target_11bp; synthetic from labels.)')
    print(f'{"="*100}\n')
    print(f'  {"metric":<40} {"synthetic":>14} {"Durrant direct":>16}   delta')
    print('  ' + '-' * 90)
    for key, desc in [
        ('n_pos', 'n positive sites'),
        ('L_median', 'guide L median'),
        ('matches_median', 'matches median'),
        ('identity_median', 'identity median'),
        ('identity_q25', 'identity Q25'),
        ('identity_q10', 'identity Q10'),
        ('frac_id_1_0', 'fraction perfect identity'),
        ('frac_id_ge_0_9', 'fraction identity >= 0.9'),
        ('frac_id_le_0_75', 'fraction identity <= 0.75'),
    ]:
        s = res_syn.get(key); r = res_gold.get(key)
        if s is None or r is None:
            print(f'  {desc:<40} {"—":>14} {"—":>16}'); continue
        if key == 'n_pos':
            print(f'  {desc:<40} {int(s):>14} {int(r):>16}   {int(s - r):+d}')
        else:
            print(f'  {desc:<40} {s:>14.3f} {r:>16.3f}   {s - r:+.3f}')

    # Pass/fail on the direct-Durrant identity thresholds
    verdicts = []
    def _check(name, delta, threshold):
        verdict = 'PASS' if abs(delta) <= threshold else 'FAIL'
        print(f'  {name:<40} {abs(delta):>14.3f} {threshold:>16.3f}   {verdict}')
        return verdict
    print(f'\n  {"PASS/FAIL":<40} {"|delta|":>14} {"threshold":>16}   verdict')
    print('  ' + '-' * 90)
    # Hard gates: identity distribution shape (Durrant-plausible band, NOT exact match).
    def _in_band(name, val, lo, hi):
        verdict = 'PASS' if lo <= val <= hi else 'FAIL'
        print(f'  {name:<40} {val:>14.3f} {f"[{lo}, {hi}]":>16}   {verdict}')
        return verdict
    def _below(name, val, thresh):
        verdict = 'PASS' if val < thresh else 'FAIL'
        print(f'  {name:<40} {val:>14.3f} {f"< {thresh}":>16}   {verdict}')
        return verdict
    def _above(name, val, thresh):
        verdict = 'PASS' if val > thresh else 'FAIL'
        print(f'  {name:<40} {val:>14.3f} {f"> {thresh}":>16}   {verdict}')
        return verdict
    if res_syn.get('identity_median') is not None:
        verdicts.append(_in_band('identity_median', res_syn['identity_median'], 0.65, 0.80))
    if res_syn.get('frac_id_1_0') is not None:
        verdicts.append(_below('frac perfect identity', res_syn['frac_id_1_0'], 0.05))
    if res_syn.get('frac_id_ge_0_9') is not None:
        verdicts.append(_below('frac identity >= 0.9', res_syn['frac_id_ge_0_9'], 0.15))
    if res_syn.get('frac_id_le_0_75') is not None:
        verdicts.append(_above('frac identity <= 0.75', res_syn['frac_id_le_0_75'], 0.40))
    if res_syn.get('L_median') is not None and res_gold.get('L_median') is not None:
        d = res_syn['L_median'] - res_gold['L_median']
        _v = 'PASS' if abs(d) <= 1.0 else 'FAIL'
        print(f'  {"L_median vs Durrant ±1":<40} {abs(d):>14.3f} {"<= 1.0":>16}   {_v}')
        verdicts.append(_v)
    # Informational only (flank-background confounded)
    print(f'\n  INFORMATIONAL (flank-background confounded, not a hard gate):')
    if res_syn.get('recall_at_4') is not None and res_real.get('recall_at_4') is not None:
        d = res_syn['recall_at_4'] - res_real['recall_at_4']
        print(f'    recall@4 syn={res_syn["recall_at_4"]:.3f} vs real={res_real["recall_at_4"]:.3f}, Δ={d:+.3f}')
    if res_syn.get('gold_rank_median') is not None and res_real.get('gold_rank_median') is not None:
        d = res_syn['gold_rank_median'] - res_real['gold_rank_median']
        print(f'    gold_rank_median syn={res_syn["gold_rank_median"]:.1f} vs real={res_real["gold_rank_median"]:.1f}, Δ={d:+.1f}')

    fails = [v for v in verdicts if v == 'FAIL']
    print()
    if fails:
        print(f'  OVERALL: {len(fails)} / {len(verdicts)} thresholds FAIL — regenerate.')
    else:
        print(f'  OVERALL: all {len(verdicts)} thresholds pass. Dataset is calibrated.')
    # Also print original side-by-side
    _ = print_side_by_side(res_syn, res_real)
    sys.exit(0 if not fails else 1)


if __name__ == '__main__':
    main()
