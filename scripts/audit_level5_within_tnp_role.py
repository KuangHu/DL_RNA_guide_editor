"""LEVEL 5 — Generic within-Tnp role reproducibility on real bags.

Question: Does the generic (topology-agnostic) hypothesis actually hold on
real IS110 bags? Are the sites within one Tnp reusing the same RNA↔DNA
interaction role, above what a shuffle-control would give?

For each real Tnp bag (n_sites >= min_sites), per site:
  Take TOP-1 junction-protected candidate. Junction-protected = if the best
  candidate (any orient, L in [7..12], anywhere in flank) lies OUTSIDE the
  15 bp junction window, we also consider the BEST candidate INSIDE the
  junction window and pick whichever has higher matches — but with a strong
  junction-priority bias: if any junction candidate has matches >= L-4, we
  take the best junction one. This mimics the `global_top4 ∪ junction_top4`
  proposal that Level 1.5 justified.

Record per site: (nc_start_norm, jdist, orient, L, matches, unp_profile at guide start).

Then per bag, compute five within-bag consistency metrics:

  D_RNApos  = MAD of nc_start_norm
  D_DNApos  = MAD of jdist                             (junction-relative flank pos)
  H_orient  = 1 - dominant_orient_fraction             (0 = all same orient)
  H_L       = 1 - modal_L_fraction                     (0 = all same guide length)
  D_struct  = median pairwise Euclidean distance between per-site 16-D unpaired profiles

Then the "best coherent subset" version — for each metric, find the k most
consistent sites and re-compute the metric on that subset for k in
{full, 70%, 50%, 30%}. This handles the case where a Tnp has genuine
biological sites plus noise sites; the coherent subset finds the biological
subgroup.

Shuffle control: within-bag, permute the (NC, flank) pairing across sites,
recompute all metrics. Real signal should show consistency_real < consistency_shuffled
(smaller MAD / entropy).

For each family (IS110, IS30, IS903, IS10-R) with n_bags >= 5, report the
distribution of (real - shuffled) deltas per metric. Significant negative
delta on IS110 vs zero on controls would support the generic hypothesis.
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

from preprocess.site import StructureCache

REAL_JSONL = '/global/scratch/users/kh36969/DL_novel_guide_editor/real_data/inference/real_all.jsonl'
CACHE_INDEX = '/global/scratch/users/kh36969/DL_novel_guide_editor/real_data/inference/real_all_u16.index.json'

BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
RC = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
LENGTHS = (7, 8, 9, 10, 11, 12)
FLANK_LEN = 120
JUNCTION_WIN = 15


def seq_to_arr(s):
    return np.asarray([BASE_MAP.get(c, 4) for c in s.upper()], dtype=np.int8)


def rc_seq(s):
    return ''.join(RC.get(c, 'N') for c in s[::-1].upper())


def best_alignment(nc, flank, restrict_junction=False, side='downstream'):
    """Best ungapped alignment across L in LENGTHS × both orients × all positions.
    If restrict_junction=True, only consider flank_start within 5 bp of junction
    (0 for downstream, len-L-5..len-L for upstream). Returns dict or None."""
    nc_a = seq_to_arr(nc); fk_a = seq_to_arr(flank)
    fk_rc_a = seq_to_arr(rc_seq(flank))
    best = None
    for L in LENGTHS:
        if len(nc_a) < L or len(fk_a) < L: continue
        nc_win = np.lib.stride_tricks.sliding_window_view(nc_a, L)
        a_oh = np.eye(5, dtype=np.int8)[nc_win]
        for orient, fw in (('fwd', fk_a), ('rc', fk_rc_a)):
            fw_win = np.lib.stride_tricks.sliding_window_view(fw, L)
            b_oh = np.eye(5, dtype=np.int8)[fw_win]
            M = np.einsum('nlc,mlc->nm', a_oh, b_oh)
            # For each nc_start, find the flank_start with max matches (junction filter if needed)
            if restrict_junction:
                # Compute allowed flank_start range in fwd coords
                if side == 'downstream':
                    lo, hi = 0, JUNCTION_WIN
                else:  # upstream
                    lo, hi = FLANK_LEN - JUNCTION_WIN - L, FLANK_LEN - L
                # For orient='rc', j indexes into flank_rc; convert requirement
                if orient == 'rc':
                    # rc flank position j maps to fwd flank position = len(fk) - j - L
                    # want lo <= (len - j - L) <= hi → len-L-hi <= j <= len-L-lo
                    j_lo = max(0, len(fk_a) - L - hi)
                    j_hi = min(M.shape[1] - 1, len(fk_a) - L - lo)
                else:
                    j_lo = max(0, lo)
                    j_hi = min(M.shape[1] - 1, hi)
                if j_lo > j_hi: continue
                M_slice = M[:, j_lo:j_hi + 1]
                if M_slice.size == 0: continue
                idx = np.unravel_index(np.argmax(M_slice), M_slice.shape)
                m = int(M_slice[idx])
                j_local = j_lo + idx[1]
                nc_start = int(idx[0])
            else:
                idx = np.unravel_index(np.argmax(M), M.shape)
                m = int(M[idx])
                nc_start = int(idx[0])
                j_local = int(idx[1])
            if orient == 'rc':
                flank_start = len(fk_a) - j_local - L
            else:
                flank_start = j_local
            cand = {'L': L, 'orient': orient, 'matches': m,
                    'nc_start': nc_start, 'flank_start': flank_start}
            if best is None or m > best['matches']:
                best = cand
    return best


def junction_protected_pick(nc, flank, side):
    """Pick top-1 candidate under junction-protected proposal:
    if best junction candidate has matches >= (L - 4), use it;
    otherwise use the global best."""
    j_cand = best_alignment(nc, flank, restrict_junction=True, side=side)
    g_cand = best_alignment(nc, flank, restrict_junction=False, side=side)
    if j_cand is None: return g_cand
    if g_cand is None: return j_cand
    # Junction-priority rule: accept junction candidate if it hits >= L-4 matches
    if j_cand['matches'] >= (j_cand['L'] - 4):
        return j_cand
    return g_cand


def load_bags():
    by_tnp = defaultdict(list)
    fams = {}
    with open(REAL_JSONL) as f:
        for line in f:
            r = json.loads(line)
            by_tnp[r['transposase_id']].append(r)
            fams[r['transposase_id']] = r['generator_metadata']['is_family']
    return by_tnp, fams


def audit_bag(recs, cache):
    """Per-site candidate + structure features."""
    sites = []
    for r in recs:
        acn = r['labels'].get('active_noncoding_index')
        if acn is None: continue
        ncs = r['inputs']['noncoding_regions']
        if acn >= len(ncs): continue
        nc = ncs[acn]; flank = r['inputs']['flank']
        side = r['generator_metadata']['flank_side']
        if len(nc) < 15 or len(flank) < 15: continue
        cand = junction_protected_pick(nc, flank, side)
        if cand is None: continue
        # Junction distance
        if side == 'downstream':
            jdist = cand['flank_start']
        else:
            jdist = FLANK_LEN - cand['flank_start'] - cand['L']
        # Structure at guide start position (16-dim unp profile)
        try:
            struct, valid = cache.get(r['site_id'], slot=acn, nc_len=len(nc))
            # Take the profile at cand['nc_start']
            if cand['nc_start'] < struct.shape[0]:
                unp_profile = struct[cand['nc_start']].astype(np.float32)
            else:
                unp_profile = np.zeros(16, dtype=np.float32)
        except Exception:
            unp_profile = np.zeros(16, dtype=np.float32)
        sites.append({
            'nc_start': cand['nc_start'],
            'nc_start_norm': cand['nc_start'] / max(1, len(nc)),
            'jdist': jdist,
            'orient': cand['orient'],
            'L': cand['L'],
            'matches': cand['matches'],
            'unp_profile': unp_profile,
            'side': side,
        })
    return sites


def bag_consistency(sites):
    if len(sites) < 3: return None
    nc_norm = np.asarray([s['nc_start_norm'] for s in sites])
    jd = np.asarray([s['jdist'] for s in sites])
    orients = [s['orient'] for s in sites]
    Ls = np.asarray([s['L'] for s in sites])
    unps = np.stack([s['unp_profile'] for s in sites], axis=0)

    def _mad(x):
        m = float(np.median(x))
        return float(np.median(np.abs(x - m)))

    dom_orient = Counter(orients).most_common(1)[0][1] / len(orients)
    modal_L = Counter(Ls.tolist()).most_common(1)[0][1] / len(Ls)
    # Pairwise Euclidean between unp profiles
    if unps.shape[0] >= 2:
        # sample max 100 pairs to keep this cheap
        rng = np.random.default_rng(0)
        n = min(100, unps.shape[0])
        idxs = rng.choice(unps.shape[0], size=n, replace=False)
        sub = unps[idxs]
        diff = sub[:, None, :] - sub[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        d = d[np.triu_indices(d.shape[0], k=1)]
        d_struct = float(np.median(d)) if len(d) else float('nan')
    else:
        d_struct = float('nan')
    return {
        'D_RNApos': _mad(nc_norm),
        'D_DNApos': _mad(jd),
        'H_orient': 1.0 - dom_orient,
        'H_L': 1.0 - modal_L,
        'D_struct': d_struct,
        'n_sites': len(sites),
    }


def coherent_subset_stats(sites, k_frac):
    """Compute per-metric consistency on the top-k_frac most-consistent subset."""
    if len(sites) < 3: return None
    k = max(3, int(np.ceil(len(sites) * k_frac)))
    if k >= len(sites): return bag_consistency(sites)

    def _keep_around_median(values, k):
        m = np.median(values)
        d = np.abs(values - m)
        order = np.argsort(d)
        return set(order[:k].tolist())

    # For each metric, find best-k subset separately and compute the metric on it.
    nc_norm = np.asarray([s['nc_start_norm'] for s in sites])
    jd = np.asarray([s['jdist'] for s in sites])
    orients = [s['orient'] for s in sites]
    Ls = np.asarray([s['L'] for s in sites])
    unps = np.stack([s['unp_profile'] for s in sites], axis=0)

    def _mad(x):
        m = float(np.median(x))
        return float(np.median(np.abs(x - m)))

    # RNApos: keep k closest to median
    keep_rna = _keep_around_median(nc_norm, k)
    D_RNApos = _mad(nc_norm[list(keep_rna)])
    keep_dna = _keep_around_median(jd, k)
    D_DNApos = _mad(jd[list(keep_dna)])
    # Orient: pick k sites from dominant orient (if enough), else all dominant + fillers
    dom_o = Counter(orients).most_common(1)[0][0]
    dom_indices = [i for i, o in enumerate(orients) if o == dom_o][:k]
    H_orient = 0.0 if len(dom_indices) >= k else 1.0 - len(dom_indices) / k
    # L: same
    modal_l = Counter(Ls.tolist()).most_common(1)[0][0]
    dom_L_indices = [i for i, L in enumerate(Ls) if L == modal_l][:k]
    H_L = 0.0 if len(dom_L_indices) >= k else 1.0 - len(dom_L_indices) / k
    # Struct: find k unps closest to centroid
    centroid = unps.mean(axis=0)
    d_from_c = np.sqrt(((unps - centroid) ** 2).sum(axis=1))
    idx = np.argsort(d_from_c)[:k]
    sub = unps[idx]
    if sub.shape[0] >= 2:
        diff = sub[:, None, :] - sub[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        d = d[np.triu_indices(d.shape[0], k=1)]
        d_struct = float(np.median(d))
    else:
        d_struct = float('nan')
    return {'D_RNApos': D_RNApos, 'D_DNApos': D_DNApos,
            'H_orient': H_orient, 'H_L': H_L, 'D_struct': d_struct,
            'n_sites': k}


def shuffle_flanks(recs, seed=1234):
    rng = random.Random(seed)
    flanks = [r['inputs']['flank'] for r in recs]
    idx = list(range(len(recs)))
    for _ in range(50):
        rng.shuffle(idx)
        if all(i != idx[i] for i in range(len(recs))): break
    out = []
    for i, r in enumerate(recs):
        r2 = json.loads(json.dumps(r))
        r2['inputs']['flank'] = flanks[idx[i]]
        out.append(r2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-sites', type=int, default=20)
    ap.add_argument('--n-per-family', type=int, default=20)
    ap.add_argument('--max-sites-per-bag', type=int, default=25)
    ap.add_argument('--controls', nargs='+', default=['IS30', 'IS903', 'IS10-R'])
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print(f'[load] real_all.jsonl + structure cache')
    by_tnp, fams = load_bags()
    cache = StructureCache(CACHE_INDEX)

    rng = random.Random(args.seed)
    picks = {}
    for fam in ['IS110'] + args.controls:
        cands = [t for t, recs in by_tnp.items()
                 if fams[t] == fam and len(recs) >= args.min_sites]
        pool = list(cands); rng.shuffle(pool)
        picks[fam] = pool[:args.n_per_family]
        print(f'  {fam:<10}: {len(cands)} bags with n>={args.min_sites}, picked {len(picks[fam])}')

    # Compute per-bag stats
    all_bags = []
    site_rng = random.Random(args.seed + 100)
    for fam, tnps in picks.items():
        for tnp in tnps:
            recs = by_tnp[tnp]
            if len(recs) > args.max_sites_per_bag:
                idxs = site_rng.sample(range(len(recs)), args.max_sites_per_bag)
                recs = [recs[i] for i in idxs]
            real_sites = audit_bag(recs, cache)
            if len(real_sites) < 3: continue
            shuf_sites = audit_bag(shuffle_flanks(recs, seed=1234), cache)
            if len(shuf_sites) < 3: continue

            entry = {'tnp': tnp, 'family': fam, 'n_sites_used': len(real_sites)}
            for k_frac, tag in [(1.0, 'full'), (0.7, 'top70'), (0.5, 'top50'), (0.3, 'top30')]:
                real_c = coherent_subset_stats(real_sites, k_frac) if k_frac < 1.0 else bag_consistency(real_sites)
                shuf_c = coherent_subset_stats(shuf_sites, k_frac) if k_frac < 1.0 else bag_consistency(shuf_sites)
                if real_c is None or shuf_c is None: continue
                for metric in ('D_RNApos', 'D_DNApos', 'H_orient', 'H_L', 'D_struct'):
                    entry[f'{metric}_{tag}_real'] = real_c[metric]
                    entry[f'{metric}_{tag}_shuf'] = shuf_c[metric]
                    entry[f'{metric}_{tag}_delta'] = real_c[metric] - shuf_c[metric]
            all_bags.append(entry)

    # Family-level report: for each metric × subset, median delta across bags
    print(f'\n{"="*110}')
    print(f'  LEVEL 5 — Within-Tnp role reproducibility: real minus shuffled')
    print(f'  (negative delta = real MORE consistent than shuffled)')
    print(f'{"="*110}\n')

    metrics = [
        ('D_RNApos',  'nc_start_norm MAD'),
        ('D_DNApos',  'junction-dist MAD'),
        ('H_orient',  'orient inconsistency (1 - dom frac)'),
        ('H_L',       'L inconsistency (1 - modal frac)'),
        ('D_struct',  'unp-profile pairwise Euclidean median'),
    ]
    for subset_tag in ('full', 'top70', 'top50', 'top30'):
        print(f'\n  --- subset: {subset_tag} ---')
        print(f'  {"family":<10} {"n_bags":>7}  ', end='')
        for m, _ in metrics: print(f'{m:>12}', end='')
        print()
        for fam in ['IS110'] + args.controls:
            fam_bags = [b for b in all_bags if b['family'] == fam]
            if not fam_bags: continue
            print(f'  {fam:<10} {len(fam_bags):>7}  ', end='')
            for m, _ in metrics:
                deltas = [b.get(f'{m}_{subset_tag}_delta') for b in fam_bags
                          if b.get(f'{m}_{subset_tag}_delta') is not None
                          and not np.isnan(b[f'{m}_{subset_tag}_delta'])]
                if not deltas:
                    print(f'{"":>12}', end='')
                else:
                    med = float(np.median(deltas))
                    print(f'{med:>+12.3f}', end='')
            print()

    # Also show per-family, per-metric fraction of bags with delta < 0
    print(f'\n{"="*110}')
    print(f'  Fraction of bags with delta < 0  (higher = more bags show real MORE consistent than shuffled)')
    print(f'{"="*110}\n')
    for subset_tag in ('full', 'top70', 'top50', 'top30'):
        print(f'\n  --- subset: {subset_tag} ---')
        print(f'  {"family":<10} {"n_bags":>7}  ', end='')
        for m, _ in metrics: print(f'{m:>12}', end='')
        print()
        for fam in ['IS110'] + args.controls:
            fam_bags = [b for b in all_bags if b['family'] == fam]
            if not fam_bags: continue
            print(f'  {fam:<10} {len(fam_bags):>7}  ', end='')
            for m, _ in metrics:
                deltas = [b.get(f'{m}_{subset_tag}_delta') for b in fam_bags
                          if b.get(f'{m}_{subset_tag}_delta') is not None
                          and not np.isnan(b[f'{m}_{subset_tag}_delta'])]
                if not deltas:
                    print(f'{"":>12}', end='')
                else:
                    frac_neg = np.mean(np.asarray(deltas) < 0)
                    print(f'{frac_neg:>12.2f}', end='')
            print()

    print(f'\n  DECISION KEY:')
    print(f'    IS110 has consistently NEGATIVE deltas at some subset level → generic hypothesis holds')
    print(f'    IS110 deltas ≈ controls or ≈ 0 → generic hypothesis unsupported by current representation')


if __name__ == '__main__':
    main()
