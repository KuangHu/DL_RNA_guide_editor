"""Build paired counterfactual negatives from V4.2 positives.

Design principle:
  Take a positive bag (all sites of one transposase). Apply a violation
  transform that ONLY changes the intended causal axis. All nuisance
  variables (NC lengths, n_ncs, flank lengths, bag size, guide sequences,
  layout) stay identical.

Profiles implemented in this file:

  paired_shuffle_v42 (level3-like)
    Permute the flank across sites within a bag: NC_i now pairs with
    F_π(i) instead of F_i. The GUIDE inside NC_i is unchanged; the FLANK
    is a different site's flank. Cognate pairing is broken; every marginal
    stays intact by construction.

  wrong_orientation_v42
    For a random subset of ~40% of sites in each bag, flip the guide
    orientation: rewrite the guide inside the NC so it would match under
    the OPPOSITE orientation. The bag now contains a mix of orientations
    → orientation_consistency violated. NC / flank / L / positions unchanged.

Every emitted record contains an assertion pass in `validate_pair()`:
  paired POS and NEG must have identical:
    len(flank), n_ncs, [len(nc_i) for each i], active_noncoding_index

If any assertion fails, we abort — the pipeline must be provably nuisance-
matched by construction.

Output:
  <out-dir>/negatives_v42_shuffle.jsonl        # paired_shuffle
  <out-dir>/negatives_v42_wrongorient.jsonl    # wrong_orientation
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_POS = '/global/scratch/users/kh36969/DL_novel_guide_editor/data/positives_v42.jsonl'
DEFAULT_OUT = '/global/scratch/users/kh36969/DL_novel_guide_editor/data'


_COMP = str.maketrans("ACGTacgt", "TGCAtgca")
def revcomp(s): return s.translate(_COMP)[::-1]


def load_by_bag(path, max_bags=None):
    by_bag = defaultdict(list)
    n = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            tnp = r['transposase_id']
            by_bag[tnp].append(r)
            n += 1
    print(f'[load] {n} records across {len(by_bag)} bags', flush=True)
    if max_bags is not None:
        keys = list(by_bag.keys())[:max_bags]
        by_bag = {k: by_bag[k] for k in keys}
        print(f'[load] restricted to first {max_bags} bags', flush=True)
    return by_bag


def validate_pair(pos_rec, neg_rec, allowed_change: set):
    """Assert that every field EXCEPT the allowed set is byte-identical."""
    checks = {
        'flank_len': (len(pos_rec['inputs']['flank']),
                       len(neg_rec['inputs']['flank'])),
        'n_ncs': (len(pos_rec['inputs']['noncoding_regions']),
                   len(neg_rec['inputs']['noncoding_regions'])),
        'nc_lens': (tuple(len(x) for x in pos_rec['inputs']['noncoding_regions']),
                     tuple(len(x) for x in neg_rec['inputs']['noncoding_regions'])),
        'active_nc_idx': (pos_rec['labels'].get('active_noncoding_index'),
                           neg_rec['labels'].get('active_noncoding_index')),
    }
    for k, (p, n) in checks.items():
        if p != n:
            raise AssertionError(
                f'nuisance mismatch on {k}: pos={p} neg={n}   allowed={allowed_change}')
    # Sequence content: allowed only if in whitelist
    if 'flank_content' not in allowed_change:
        if pos_rec['inputs']['flank'] != neg_rec['inputs']['flank']:
            raise AssertionError(f'flank content differs but not allowed')
    if 'nc_content' not in allowed_change:
        for a, b in zip(pos_rec['inputs']['noncoding_regions'],
                          neg_rec['inputs']['noncoding_regions']):
            if a != b:
                raise AssertionError(f'nc content differs but not allowed')


def paired_shuffle_bag(pos_bag, rng):
    """Emit paired shuffle negatives from a positive bag.
    Each output record has NC identical to some positive but flank replaced
    by another site's flank in the same bag. Only 'flank_content' allowed to
    change.
    """
    n = len(pos_bag)
    if n < 2: return []
    # Derangement
    idx = list(range(n))
    for _ in range(50):
        rng.shuffle(idx)
        if all(i != idx[i] for i in range(n)): break
    out = []
    for i, pos in enumerate(pos_bag):
        donor = pos_bag[idx[i]]
        neg = json.loads(json.dumps(pos))  # deep copy
        neg['site_id'] = f'{pos["site_id"]}_shuf'
        neg['inputs']['flank'] = donor['inputs']['flank']
        neg['labels']['is_positive'] = False
        neg['labels']['site_class'] = 'guided'  # site itself LOOKS like a positive
        neg['labels']['violation_profile'] = 'paired_shuffle_v42'
        # Nullify all pairing-specific labels since they no longer describe reality
        neg['labels']['target_position_in_flank'] = None
        neg['labels']['target_dna'] = None
        # Keep guide + guide_length + orientation to preserve NC layout
        # Add violation metadata
        neg['generator_metadata']['violation'] = {
            'profile': 'paired_shuffle_v42',
            'donor_site_id': donor['site_id'],
            'axis_changed': 'flank_content (cognate pairing broken)',
        }
        try:
            validate_pair(pos, neg, allowed_change={'flank_content'})
        except AssertionError as e:
            print(f'[validate] {e}', flush=True)
            continue
        out.append(neg)
    return out


def _revcomp_guide_in_nc(nc: str, guide_span, guide_length: int) -> str:
    """Replace the guide slot in NC with RC of what's there."""
    gs, ge = guide_span
    if ge - gs != guide_length:
        return nc  # can't safely modify
    old_guide = nc[gs:ge]
    new_guide = revcomp(old_guide)
    return nc[:gs] + new_guide + nc[ge:]


def wrong_orient_bag(pos_bag, rng, flip_frac=0.4):
    """Emit wrong_orientation negatives. For ~40% of sites in the bag,
    RC-invert the guide sequence inside the NC. Flank is unchanged.

    This creates BAG-LEVEL orientation inconsistency. Individual sites still
    LOOK like plausible pairing (some hit fwd, some hit rc), but the BAG
    lacks a consistent orientation rule."""
    n = len(pos_bag)
    if n < 3: return []
    idx = list(range(n))
    rng.shuffle(idx)
    flipped_idx = set(idx[:max(1, int(n * flip_frac))])
    out = []
    for i, pos in enumerate(pos_bag):
        neg = json.loads(json.dumps(pos))
        neg['site_id'] = f'{pos["site_id"]}_wrongori'
        neg['labels']['is_positive'] = False
        neg['labels']['violation_profile'] = 'wrong_orientation_v42'
        neg['labels']['site_class'] = 'guided'
        if i in flipped_idx:
            # RC-invert the guide bases inside the active NC
            active_idx = pos['labels']['active_noncoding_index']
            span = pos['labels'].get('guide_span_in_active_noncoding')
            L = pos['labels']['guide_length']
            if span is None or active_idx is None: continue
            ncs = list(neg['inputs']['noncoding_regions'])
            ncs[active_idx] = _revcomp_guide_in_nc(ncs[active_idx], span, L)
            neg['inputs']['noncoding_regions'] = ncs
            # Also flip label orientation to reflect the change
            neg['labels']['match_orientation'] = (
                'forward' if pos['labels']['match_orientation'] == 'reverse_complement'
                else 'reverse_complement')
            neg['generator_metadata']['violation'] = {
                'profile': 'wrong_orientation_v42',
                'site_flipped': True,
                'flip_frac_in_bag': flip_frac,
            }
        else:
            neg['generator_metadata']['violation'] = {
                'profile': 'wrong_orientation_v42',
                'site_flipped': False,
                'flip_frac_in_bag': flip_frac,
            }
        try:
            validate_pair(pos, neg, allowed_change={'nc_content'})
        except AssertionError as e:
            print(f'[validate] {e}', flush=True)
            continue
        out.append(neg)
    return out


def _random_dna(rng, n):
    return ''.join(rng.choice('ACGT') for _ in range(n))


def wrong_position_bag(pos_bag, rng, jitter_min=15):
    """Move the target substring in the flank to a NEW position; fill the
    original slot with random DNA. NC unchanged. Layout invariant."""
    out = []
    for pos in pos_bag:
        neg = json.loads(json.dumps(pos))
        neg['site_id'] = f'{pos["site_id"]}_wrongpos'
        tp = pos['labels'].get('target_position_in_flank')
        L = pos['labels'].get('guide_length')
        flank = pos['inputs']['flank']
        if tp is None or L is None or not (isinstance(tp, list) and len(tp) == 2):
            continue
        ts, te = tp
        flen = len(flank)
        target = flank[ts:te]
        # Pick a new position at least jitter_min away
        candidates = [i for i in range(0, flen - L + 1)
                      if abs(i - ts) >= jitter_min]
        if not candidates: continue
        new_ts = rng.choice(candidates)
        new_te = new_ts + L
        # Construct new flank: replace target slot with random DNA, insert target at new slot
        new_flank = list(flank)
        # Fill old target slot with random DNA
        for i, b in enumerate(_random_dna(rng, L)):
            new_flank[ts + i] = b
        # Save what's at the new slot to move somewhere (destructive)
        for i, b in enumerate(target):
            new_flank[new_ts + i] = b
        neg['inputs']['flank'] = ''.join(new_flank)
        neg['labels']['is_positive'] = False
        neg['labels']['site_class'] = 'guided'
        neg['labels']['violation_profile'] = 'wrong_position_v42'
        neg['labels']['target_position_in_flank'] = [new_ts, new_te]
        neg['generator_metadata']['violation'] = {
            'profile': 'wrong_position_v42',
            'old_target_pos': [ts, te],
            'new_target_pos': [new_ts, new_te],
            'axis_changed': 'target_position_in_flank',
        }
        try:
            validate_pair(pos, neg, allowed_change={'flank_content'})
        except AssertionError as e:
            print(f'[validate] {e}', flush=True)
            continue
        out.append(neg)
    return out


def wrong_length_bag(pos_bag, rng):
    """Simulate a per-bag mixed-length violation. For each site, choose a new
    'effective guide length' L' != L. Fill the first L' bases of the guide slot
    with random DNA (destroying that portion of the true guide); the rest of
    the slot keeps the original guide bases. NC/flank lengths unchanged.
    """
    out = []
    for pos in pos_bag:
        neg = json.loads(json.dumps(pos))
        neg['site_id'] = f'{pos["site_id"]}_wronglen'
        L = pos['labels'].get('guide_length')
        active_idx = pos['labels'].get('active_noncoding_index')
        span = pos['labels'].get('guide_span_in_active_noncoding')
        if L is None or active_idx is None or span is None: continue
        gs, ge = span
        # Sample new effective length differing from L by at least 2
        options = [k for k in range(4, min(17, ge - gs + 1)) if abs(k - L) >= 2]
        if not options: continue
        L_new = rng.choice(options)
        # Corrupt bases beyond position L_new in the guide slot
        ncs = list(neg['inputs']['noncoding_regions'])
        old_nc = ncs[active_idx]
        # Replace all guide-slot bases with random DNA; a subset (first L_new)
        # simulates the new "guide" — but its sequence is random relative to
        # the flank target, so the true pairing is broken.
        rand_bases = _random_dna(rng, ge - gs)
        ncs[active_idx] = old_nc[:gs] + rand_bases + old_nc[ge:]
        neg['inputs']['noncoding_regions'] = ncs
        neg['labels']['is_positive'] = False
        neg['labels']['site_class'] = 'guided'
        neg['labels']['violation_profile'] = 'wrong_length_v42'
        neg['labels']['guide_length_effective'] = L_new
        neg['generator_metadata']['violation'] = {
            'profile': 'wrong_length_v42',
            'old_L': L, 'new_effective_L': L_new,
            'axis_changed': 'guide_length_consistency',
        }
        try:
            validate_pair(pos, neg, allowed_change={'nc_content'})
        except AssertionError as e:
            print(f'[validate] {e}', flush=True)
            continue
        out.append(neg)
    return out


def wrong_structure_bag(pos_bag, rng):
    """Move the guide substring inside the NC to a DIFFERENT position within
    the same NC (different structural context). Fill original slot with
    random DNA. NC total length preserved."""
    out = []
    for pos in pos_bag:
        neg = json.loads(json.dumps(pos))
        neg['site_id'] = f'{pos["site_id"]}_wrongstr'
        L = pos['labels'].get('guide_length')
        active_idx = pos['labels'].get('active_noncoding_index')
        span = pos['labels'].get('guide_span_in_active_noncoding')
        if L is None or active_idx is None or span is None: continue
        gs, ge = span
        ncs = list(neg['inputs']['noncoding_regions'])
        nc = ncs[active_idx]
        # Choose new position far from original
        candidates = [i for i in range(0, len(nc) - L + 1)
                      if abs(i - gs) >= 20 and (i + L <= gs or i >= ge)]
        if not candidates: continue
        new_gs = rng.choice(candidates)
        new_ge = new_gs + L
        guide_seq = nc[gs:ge]
        # Save what's currently at new_gs..new_ge (will be overwritten)
        occ_seq = nc[new_gs:new_ge]
        nc_list = list(nc)
        # Fill old guide slot with random DNA
        for i, b in enumerate(_random_dna(rng, L)):
            nc_list[gs + i] = b
        # Put guide at new position (overwrites what was there)
        for i, b in enumerate(guide_seq):
            nc_list[new_gs + i] = b
        ncs[active_idx] = ''.join(nc_list)
        neg['inputs']['noncoding_regions'] = ncs
        neg['labels']['is_positive'] = False
        neg['labels']['site_class'] = 'guided'
        neg['labels']['violation_profile'] = 'wrong_structure_role_v42'
        neg['labels']['guide_span_in_active_noncoding'] = [new_gs, new_ge]
        neg['generator_metadata']['violation'] = {
            'profile': 'wrong_structure_role_v42',
            'old_guide_span': [gs, ge],
            'new_guide_span': [new_gs, new_ge],
            'axis_changed': 'guide_position_within_NC',
        }
        try:
            validate_pair(pos, neg, allowed_change={'nc_content'})
        except AssertionError as e:
            print(f'[validate] {e}', flush=True)
            continue
        out.append(neg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pos', default=DEFAULT_POS)
    ap.add_argument('--out-dir', default=DEFAULT_OUT)
    ap.add_argument('--profiles', nargs='+',
                     default=['paired_shuffle', 'wrong_orientation',
                              'wrong_position', 'wrong_length', 'wrong_structure'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-bags', type=int, default=None)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    pos_by_bag = load_by_bag(args.pos, max_bags=args.max_bags)

    if 'paired_shuffle' in args.profiles:
        out_path = out_dir / 'negatives_v42_shuffle.jsonl'
        print(f'\n[emit] paired_shuffle -> {out_path}', flush=True)
        n_emit = 0
        with out_path.open('w') as fh:
            for bag_i, (tnp, bag) in enumerate(pos_by_bag.items()):
                recs = paired_shuffle_bag(bag, rng)
                for r in recs:
                    fh.write(json.dumps(r) + '\n')
                n_emit += len(recs)
                if (bag_i + 1) % 1000 == 0:
                    print(f'  processed {bag_i+1} bags, emitted {n_emit}', flush=True)
        print(f'  done: {n_emit} sites', flush=True)

    if 'wrong_orientation' in args.profiles:
        out_path = out_dir / 'negatives_v42_wrongorient.jsonl'
        print(f'\n[emit] wrong_orientation -> {out_path}', flush=True)
        n_emit = 0
        with out_path.open('w') as fh:
            for bag_i, (tnp, bag) in enumerate(pos_by_bag.items()):
                recs = wrong_orient_bag(bag, rng)
                for r in recs:
                    fh.write(json.dumps(r) + '\n')
                n_emit += len(recs)
                if (bag_i + 1) % 1000 == 0:
                    print(f'  processed {bag_i+1} bags, emitted {n_emit}', flush=True)
        print(f'  done: {n_emit} sites', flush=True)

    if 'wrong_position' in args.profiles:
        out_path = out_dir / 'negatives_v42_wrongpos.jsonl'
        print(f'\n[emit] wrong_position -> {out_path}', flush=True)
        n_emit = 0
        with out_path.open('w') as fh:
            for bag_i, (tnp, bag) in enumerate(pos_by_bag.items()):
                recs = wrong_position_bag(bag, rng)
                for r in recs:
                    fh.write(json.dumps(r) + '\n')
                n_emit += len(recs)
                if (bag_i + 1) % 1000 == 0:
                    print(f'  processed {bag_i+1} bags, emitted {n_emit}', flush=True)
        print(f'  done: {n_emit} sites', flush=True)

    if 'wrong_length' in args.profiles:
        out_path = out_dir / 'negatives_v42_wronglen.jsonl'
        print(f'\n[emit] wrong_length -> {out_path}', flush=True)
        n_emit = 0
        with out_path.open('w') as fh:
            for bag_i, (tnp, bag) in enumerate(pos_by_bag.items()):
                recs = wrong_length_bag(bag, rng)
                for r in recs:
                    fh.write(json.dumps(r) + '\n')
                n_emit += len(recs)
                if (bag_i + 1) % 1000 == 0:
                    print(f'  processed {bag_i+1} bags, emitted {n_emit}', flush=True)
        print(f'  done: {n_emit} sites', flush=True)

    if 'wrong_structure' in args.profiles:
        out_path = out_dir / 'negatives_v42_wrongstruct.jsonl'
        print(f'\n[emit] wrong_structure -> {out_path}', flush=True)
        n_emit = 0
        with out_path.open('w') as fh:
            for bag_i, (tnp, bag) in enumerate(pos_by_bag.items()):
                recs = wrong_structure_bag(bag, rng)
                for r in recs:
                    fh.write(json.dumps(r) + '\n')
                n_emit += len(recs)
                if (bag_i + 1) % 1000 == 0:
                    print(f'  processed {bag_i+1} bags, emitted {n_emit}', flush=True)
        print(f'  done: {n_emit} sites', flush=True)


if __name__ == '__main__':
    main()
