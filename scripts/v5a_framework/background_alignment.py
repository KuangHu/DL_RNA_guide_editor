"""Shared, cached affine-gap alignment statistics for Phase D.

Paired identity excludes gaps. Indels are reported separately. Coverage
must be evaluated on BOTH input lengths before using this identity.
"""
from functools import lru_cache
from Bio.Align import PairwiseAligner


@lru_cache(maxsize=65536)
def gapped_identity(a: str, b: str, gap_open: int = -2, gap_ext: int = -1,
                    match: int = 1, mismatch: int = -1) -> tuple:
    if not a or not b:
        return (0.0, 0, 0, 0, 0, min(len(a), len(b)))
    aligner = PairwiseAligner(mode='global', match_score=match,
                             mismatch_score=mismatch,
                             open_gap_score=gap_open, extend_gap_score=gap_ext)
    alignment = aligner.align(a, b)[0]
    paired = matches = 0
    for (alo, ahi), (blo, bhi) in zip(*alignment.aligned):
        paired += int(ahi - alo)
        matches += sum(x == y and x in 'ACGT' for x, y in zip(a[alo:ahi], b[blo:bhi]))
    gaps = len(a) + len(b) - 2 * paired
    return (matches / paired if paired else 0.0, paired, matches,
            paired - matches, gaps, min(len(a), len(b)))


def side_metrics(a: str, b: str, min_bp: int, min_ratio: float) -> dict:
    ident, paired, matches, mismatches, indels, shorter = gapped_identity(a, b)
    shorter_ratio = paired / shorter if shorter else 0.0
    longer_ratio = paired / max(len(a), len(b)) if a and b else 0.0
    # Unknown bases cannot supply evidence of a distinct background.
    canonical = bool(a and b) and set(a + b) <= set('ACGT')
    return dict(identity=ident, paired_bp=paired, matches=matches,
                mismatches=mismatches, indels=indels,
                coverage_shorter=shorter_ratio, coverage_longer=longer_ratio,
                eligible=canonical and paired >= min_bp and longer_ratio >= min_ratio)


def select_side(left: dict, right: dict) -> tuple[str, dict | None]:
    eligible = [(side, metrics) for side, metrics in [('left', left), ('right', right)]
                if metrics['eligible']]
    return max(eligible, key=lambda sm: (sm[1]['identity'], sm[1]['paired_bp'])) if eligible else ('none', None)


def same_locus(a: dict, b: dict, overlap_bp: int = 100) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    keys = ('pre_event_genome', 'pre_event_contig')
    if any(not isinstance(a.get(k), str) or not a[k].strip() or
           not isinstance(b.get(k), str) or not b[k].strip() for k in keys):
        return False
    if any(a[k] != b[k] for k in keys):
        return False
    try:
        alo, ahi, blo, bhi = (int(a['start']), int(a['end']), int(b['start']), int(b['end']))
    except (KeyError, ValueError, TypeError):
        return False
    return alo < ahi and blo < bhi and min(ahi, bhi) - max(alo, blo) >= overlap_bp
