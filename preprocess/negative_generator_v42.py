"""V4.2 counterfactual negative generator.

Contract-based negative generation: each profile declares
  * `allowed_changes` (whitelist of fields that MAY differ between paired
    POS and NEG)
  * everything else is asserted byte-identical at generation time

The five profiles collectively cover the causal axes we want the classifier
to actually learn:

  paired_shuffle_v42          break cognate NC↔flank assignment
  wrong_orientation_v42       break orientation consistency across sites
  wrong_position_v42          break target-position / junction relation
  wrong_length_v42            break guide-length consistency
  wrong_structure_role_v42    break structural-role placement

Every emitted record carries a `generator_metadata.negative_generator`
provenance block so any pipeline mistake is visible via audit but never
reaches the model tensor (loader must whitelist `inputs.*` only).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

NEG_GEN_VERSION = "v4.2_counterfactual"


_COMP = str.maketrans("ACGTacgt", "TGCAtgca")
def _revcomp(s: str) -> str: return s.translate(_COMP)[::-1]


def _random_dna(rng: random.Random, n: int) -> str:
    return ''.join(rng.choice('ACGT') for _ in range(n))


class InvariantViolation(AssertionError):
    """Raised when a paired POS/NEG violates the profile's contract."""


def _assert_layout_invariants(pos: Dict, neg: Dict) -> None:
    """Universal invariants — MUST hold for every negative profile."""
    if len(pos['inputs']['flank']) != len(neg['inputs']['flank']):
        raise InvariantViolation('flank length differs')
    p_ncs = pos['inputs']['noncoding_regions']
    n_ncs = neg['inputs']['noncoding_regions']
    if len(p_ncs) != len(n_ncs):
        raise InvariantViolation('n_ncs differs')
    for i, (a, b) in enumerate(zip(p_ncs, n_ncs)):
        if len(a) != len(b):
            raise InvariantViolation(f'NC[{i}] length differs')
    if pos['labels'].get('active_noncoding_index') != neg['labels'].get('active_noncoding_index'):
        raise InvariantViolation('active_noncoding_index differs')


def _assert_content_invariants(pos: Dict, neg: Dict, allowed: Set[str]) -> None:
    if 'flank_content' not in allowed:
        if pos['inputs']['flank'] != neg['inputs']['flank']:
            raise InvariantViolation(
                'flank content differs but "flank_content" not in allowed_changes')
    if 'nc_content' not in allowed:
        for i, (a, b) in enumerate(zip(pos['inputs']['noncoding_regions'],
                                        neg['inputs']['noncoding_regions'])):
            if a != b:
                raise InvariantViolation(
                    f'NC[{i}] content differs but "nc_content" not in allowed_changes')


@dataclass
class NegativeProfile:
    """Base class. Subclasses implement `_transform_site` OR `transform_bag`.

    Subclasses declare:
      name             — profile string used in labels + metadata
      allowed_changes  — whitelist of fields allowed to differ from POS
      params           — profile-specific kwargs (e.g. flip_frac, jitter_min)
    """
    name: str = ''
    allowed_changes: Set[str] = field(default_factory=set)
    params: Dict = field(default_factory=dict)

    def transform_bag(self, pos_bag: List[Dict],
                       rng: random.Random) -> List[Dict]:
        """Default: apply per-site transform independently."""
        out = []
        for pos in pos_bag:
            neg = self._transform_site(pos, rng)
            if neg is None: continue
            self._finalize(pos, neg)
            out.append(neg)
        return out

    def _transform_site(self, pos: Dict, rng: random.Random) -> Optional[Dict]:
        raise NotImplementedError

    def _finalize(self, pos: Dict, neg: Dict) -> None:
        """Attach provenance + run invariant checks. Raises on violation."""
        neg['labels']['is_positive'] = False
        neg['labels']['violation_profile'] = self.name
        gm = neg.setdefault('generator_metadata', {})
        gm['negative_generator'] = {
            'version': NEG_GEN_VERSION,
            'profile': self.name,
            'parent_positive_site_id': pos['site_id'],
            'allowed_changes': sorted(self.allowed_changes),
            'params': dict(self.params),
        }
        _assert_layout_invariants(pos, neg)
        _assert_content_invariants(pos, neg, self.allowed_changes)
        gm['negative_generator']['invariant_check_passed'] = True


class PairedShuffle(NegativeProfile):
    """Permute flanks across sites within a bag. Only flank_content changes.
    Cognate NC↔flank assignment is broken; every marginal invariant is
    identical by construction."""
    def __init__(self):
        super().__init__(name='paired_shuffle_v42',
                          allowed_changes={'flank_content'},
                          params={})

    def transform_bag(self, pos_bag, rng):
        n = len(pos_bag)
        if n < 2: return []
        idx = list(range(n))
        for _ in range(50):
            rng.shuffle(idx)
            if all(i != idx[i] for i in range(n)): break
        out = []
        for i, pos in enumerate(pos_bag):
            donor = pos_bag[idx[i]]
            neg = json.loads(json.dumps(pos))
            neg['site_id'] = f'{pos["site_id"]}_shuf'
            neg['inputs']['flank'] = donor['inputs']['flank']
            # Nullify labels that no longer describe reality
            neg['labels']['target_position_in_flank'] = None
            neg['labels']['target_dna'] = None
            neg['labels']['site_class'] = 'guided'
            self._finalize(pos, neg)
            neg['generator_metadata']['negative_generator']['donor_site_id'] = donor['site_id']
            out.append(neg)
        return out


class WrongOrientation(NegativeProfile):
    """For flip_frac of sites in each bag, RC-invert the guide inside the NC.
    Same flank, same layout, different guide-orient for a bag subset."""
    def __init__(self, flip_frac: float = 0.4):
        super().__init__(name='wrong_orientation_v42',
                          allowed_changes={'nc_content'},
                          params={'flip_frac': flip_frac})

    def transform_bag(self, pos_bag, rng):
        n = len(pos_bag)
        if n < 3: return []
        idx = list(range(n)); rng.shuffle(idx)
        k = max(1, int(n * self.params['flip_frac']))
        flipped = set(idx[:k])
        out = []
        for i, pos in enumerate(pos_bag):
            neg = json.loads(json.dumps(pos))
            neg['site_id'] = f'{pos["site_id"]}_wrongori'
            neg['labels']['site_class'] = 'guided'
            if i in flipped:
                active = pos['labels']['active_noncoding_index']
                span = pos['labels'].get('guide_span_in_active_noncoding')
                L = pos['labels']['guide_length']
                if span is None or active is None or (span[1] - span[0] != L):
                    continue
                gs, ge = span
                ncs = list(neg['inputs']['noncoding_regions'])
                old = ncs[active]
                ncs[active] = old[:gs] + _revcomp(old[gs:ge]) + old[ge:]
                neg['inputs']['noncoding_regions'] = ncs
                new_orient = ('forward' if pos['labels']['match_orientation']
                              == 'reverse_complement' else 'reverse_complement')
                neg['labels']['match_orientation'] = new_orient
            self._finalize(pos, neg)
            neg['generator_metadata']['negative_generator']['site_flipped'] = (i in flipped)
            out.append(neg)
        return out


class WrongPosition(NegativeProfile):
    """Move target sequence to a NEW position in the flank (>= jitter_min bp
    from the original); fill the vacated slot with random DNA.
    Only flank_content changes."""
    def __init__(self, jitter_min: int = 15):
        super().__init__(name='wrong_position_v42',
                          allowed_changes={'flank_content'},
                          params={'jitter_min': jitter_min})

    def _transform_site(self, pos, rng):
        tp = pos['labels'].get('target_position_in_flank')
        L = pos['labels'].get('guide_length')
        flank = pos['inputs']['flank']
        if tp is None or L is None or not (isinstance(tp, list) and len(tp) == 2):
            return None
        ts, te = tp
        flen = len(flank)
        target = flank[ts:te]
        cands = [i for i in range(0, flen - L + 1)
                 if abs(i - ts) >= self.params['jitter_min']]
        if not cands: return None
        new_ts = rng.choice(cands); new_te = new_ts + L
        buf = list(flank)
        # Fill old slot with random DNA
        for i, b in enumerate(_random_dna(rng, L)):
            buf[ts + i] = b
        # Overwrite new slot with the target
        for i, b in enumerate(target):
            buf[new_ts + i] = b
        neg = json.loads(json.dumps(pos))
        neg['site_id'] = f'{pos["site_id"]}_wrongpos'
        neg['inputs']['flank'] = ''.join(buf)
        neg['labels']['target_position_in_flank'] = [new_ts, new_te]
        neg['labels']['site_class'] = 'guided'
        return neg


class WrongLength(NegativeProfile):
    """Corrupt the guide slot inside the NC with random DNA. The effective
    guide length becomes ill-defined (bag-level length consistency broken).
    Only nc_content changes."""
    def __init__(self):
        super().__init__(name='wrong_length_v42',
                          allowed_changes={'nc_content'},
                          params={})

    def _transform_site(self, pos, rng):
        active = pos['labels'].get('active_noncoding_index')
        span = pos['labels'].get('guide_span_in_active_noncoding')
        L = pos['labels'].get('guide_length')
        if active is None or span is None or L is None: return None
        gs, ge = span
        options = [k for k in range(4, min(17, ge - gs + 1)) if abs(k - L) >= 2]
        if not options: return None
        L_new = rng.choice(options)
        ncs = list(pos['inputs']['noncoding_regions'])
        old_nc = ncs[active]
        random_slot = _random_dna(rng, ge - gs)
        ncs[active] = old_nc[:gs] + random_slot + old_nc[ge:]
        neg = json.loads(json.dumps(pos))
        neg['site_id'] = f'{pos["site_id"]}_wronglen'
        neg['inputs']['noncoding_regions'] = ncs
        neg['labels']['guide_length_effective'] = L_new
        neg['labels']['site_class'] = 'guided'
        return neg


class WrongStructureRole(NegativeProfile):
    """Move the guide substring inside the NC to a different position at
    least `min_shift` bp from the original — puts the guide in a different
    structural context. Only nc_content changes."""
    def __init__(self, min_shift: int = 20):
        super().__init__(name='wrong_structure_role_v42',
                          allowed_changes={'nc_content'},
                          params={'min_shift': min_shift})

    def _transform_site(self, pos, rng):
        active = pos['labels'].get('active_noncoding_index')
        span = pos['labels'].get('guide_span_in_active_noncoding')
        L = pos['labels'].get('guide_length')
        if active is None or span is None or L is None: return None
        gs, ge = span
        ncs = list(pos['inputs']['noncoding_regions'])
        nc = ncs[active]
        cands = [i for i in range(0, len(nc) - L + 1)
                 if abs(i - gs) >= self.params['min_shift']
                 and (i + L <= gs or i >= ge)]
        if not cands: return None
        new_gs = rng.choice(cands); new_ge = new_gs + L
        guide_seq = nc[gs:ge]
        buf = list(nc)
        for i, b in enumerate(_random_dna(rng, L)):
            buf[gs + i] = b
        for i, b in enumerate(guide_seq):
            buf[new_gs + i] = b
        ncs[active] = ''.join(buf)
        neg = json.loads(json.dumps(pos))
        neg['site_id'] = f'{pos["site_id"]}_wrongstr'
        neg['inputs']['noncoding_regions'] = ncs
        neg['labels']['guide_span_in_active_noncoding'] = [new_gs, new_ge]
        neg['labels']['site_class'] = 'guided'
        return neg


PROFILE_REGISTRY = {
    'paired_shuffle':      PairedShuffle,
    'wrong_orientation':    WrongOrientation,
    'wrong_position':       WrongPosition,
    'wrong_length':         WrongLength,
    'wrong_structure_role': WrongStructureRole,
}


def build_profiles(cfg: Dict) -> Dict[str, NegativeProfile]:
    """Return {config_key: instance} for each ENABLED profile from cfg."""
    out = {}
    for key, spec in cfg.get('profiles', {}).items():
        if not spec.get('enabled', False): continue
        if key not in PROFILE_REGISTRY:
            raise ValueError(f'unknown profile: {key}')
        cls = PROFILE_REGISTRY[key]
        init_kwargs = {k: v for k, v in spec.items()
                       if k not in ('enabled', 'weight')}
        out[key] = cls(**init_kwargs)
    return out
