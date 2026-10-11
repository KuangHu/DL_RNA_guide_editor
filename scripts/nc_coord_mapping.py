"""R1 — per-fragment nc coord mapping, no b0+nc_pos shortcut.

For each nc fragment: locate ALL exact matches in occupied (both forward
and reverse strands). The fragment's source is:
  - UNIQUE: exactly one full-length match → (lo, hi, strand) stored.
  - AMBIGUOUS: multiple full-length matches → list of candidates stored,
    the one closest to insert start reported as primary with ambiguous=True.
  - MISSING: no full-length match → degrade to best partial (prefix 40bp,
    then 30, then 20 — in that order); if still none, strand/lo/hi = None.

For any nc_pos in the concat (= frag[0] + "N"*SPACER_N + frag[1] + ... ):
  - returns (frag_idx, offset_in_frag, projected_occupied_lo,
             projected_occupied_hi, projection_strand, ambiguous, note)
  - When nc_pos lands in the N-spacer, frag_idx=-1, offset=-1, projection
    marks it as "spacer: not real DNA".

For a FULL MOTIF INTERVAL [p, p+L) in concat:
  - If both p and p+L-1 land in the SAME fragment: project the interval
    end-to-end from that fragment's source.
  - If they cross a spacer: return "crosses_spacer" (treat as ambiguous).
  - If they cross a fragment boundary (same fragment but part of concat
    extends outside): return "crosses_fragment_boundary".

The projected occupied interval respects strand: for strand=+ a fragment's
nc offset k projects to (frag_lo + k, +); for strand=- it projects to
(frag_hi - 1 - k, -). The full L-mer projected to occupied is thus:
  strand=+: [frag_lo + k, frag_lo + k + L)
  strand=-: [frag_hi - k - L, frag_hi - k)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

SPACER_N = 13  # matches scripts.v5a_framework.multi_hit_retention.SPACER_N
COMP = str.maketrans("ACGT", "TGCA")

def _revcomp(s: str) -> str:
    return s.translate(COMP)[::-1]


def _find_all(sub: str, hay: str) -> list[int]:
    out, i = [], 0
    while True:
        j = hay.find(sub, i)
        if j < 0: break
        out.append(j); i = j + 1
    return out


@dataclass
class FragmentSource:
    frag_idx: int
    length: int
    occupied_lo: Optional[int]       # 0-based start in occupied (half-open)
    occupied_hi: Optional[int]       # 0-based end in occupied (half-open)
    strand: Optional[str]            # '+' or '-' or None
    status: str                      # 'unique', 'ambiguous', 'partial_<N>', 'missing'
    candidates: list[tuple[int, int, str]] = field(default_factory=list)
    note: str = ""


@dataclass
class CoordProjection:
    source_frag_idx: int             # -1 if in spacer
    offset_in_frag: int              # -1 if in spacer
    occupied_lo: Optional[int]
    occupied_hi: Optional[int]       # one-past-end of the projected L-mer
    strand: Optional[str]
    ambiguous: bool
    status: str                      # 'ok' | 'spacer' | 'crosses_spacer' |
                                     # 'crosses_fragment_boundary' |
                                     # 'ambiguous_source' | 'missing_source'
    note: str = ""


def map_fragments_to_occupied(nc_frags: list[str], occupied: str,
                                hint_near: Optional[int] = None,
                                partial_lens: tuple[int, ...] = (40, 30, 20),
                                ) -> list[FragmentSource]:
    """For each fragment, find all exact full-length matches in occupied on
    both strands. Return a FragmentSource per fragment."""
    sources: list[FragmentSource] = []
    for i, frag in enumerate(nc_frags):
        F = frag.upper()
        hits: list[tuple[int, int, str]] = []
        for lo in _find_all(F, occupied):
            hits.append((lo, lo + len(F), "+"))
        rF = _revcomp(F)
        for lo in _find_all(rF, occupied):
            hits.append((lo, lo + len(F), "-"))
        if len(hits) == 1:
            lo, hi, strand = hits[0]
            sources.append(FragmentSource(i, len(F), lo, hi, strand,
                                            "unique", hits, "exact full-length"))
            continue
        if len(hits) > 1:
            primary = (min(hits, key=lambda t: abs(t[0] - hint_near))
                        if hint_near is not None else hits[0])
            sources.append(FragmentSource(
                i, len(F), primary[0], primary[1], primary[2],
                "ambiguous", hits,
                f"{len(hits)} full-length matches; primary chosen by hint_near={hint_near}"))
            continue
        # Full-length missing: degrade to longest-prefix / RC
        for plen in partial_lens:
            if plen > len(F): continue
            pref = F[:plen]
            rpref = _revcomp(pref)
            cand = []
            for lo in _find_all(pref, occupied):
                cand.append((lo, lo + len(F), "+"))
            for lo in _find_all(rpref, occupied):
                cand.append((lo - (len(F) - plen), lo + plen, "-"))
            if len(cand) == 1:
                lo, hi, strand = cand[0]
                sources.append(FragmentSource(
                    i, len(F), lo, hi, strand, f"partial_{plen}", cand,
                    f"unique match on {plen}-bp prefix"))
                break
            if len(cand) > 1 and hint_near is not None:
                primary = min(cand, key=lambda t: abs(t[0] - hint_near))
                sources.append(FragmentSource(
                    i, len(F), primary[0], primary[1], primary[2],
                    f"ambiguous_partial_{plen}", cand,
                    f"{len(cand)} partial({plen}) matches; "
                    f"primary by hint_near={hint_near}"))
                break
        else:
            sources.append(FragmentSource(i, len(F), None, None, None,
                                            "missing", [], "no match"))
    return sources


def concat_pos_to_frag(nc_frags: list[str], concat_pos: int,
                        ) -> tuple[int, int, bool]:
    """(frag_idx, offset_in_frag, in_spacer). frag_idx=-1 when in_spacer."""
    off = 0
    for i, frag in enumerate(nc_frags):
        if concat_pos < off + len(frag):
            return (i, concat_pos - off, False)
        off += len(frag)
        if concat_pos < off + SPACER_N:
            return (-1, concat_pos - off, True)
        off += SPACER_N
    return (-1, -1, True)


def project_interval(nc_frags: list[str], sources: list[FragmentSource],
                       concat_lo: int, concat_hi: int,
                       occupied_len: Optional[int] = None,
                       ) -> CoordProjection:
    """Project the concat interval [concat_lo, concat_hi) into occupied.

    Returns a CoordProjection whose occupied_lo/_hi describe the
    strand-corrected projection of the FULL interval. Status is "ok" only
    when (a) the fragment source is unique and (b) the projected interval
    stays within fragment bounds and (when occupied_len is given) within
    [0, occupied_len]. Any other case yields a non-ok status so integration
    code can filter by `status == "ok"`.
    """
    L = concat_hi - concat_lo
    if L <= 0:
        return CoordProjection(-1, -1, None, None, None, True,
                                "empty_interval", f"L={L}")
    f1, o1, s1 = concat_pos_to_frag(nc_frags, concat_lo)
    f2, o2, s2 = concat_pos_to_frag(nc_frags, concat_hi - 1)
    if s1 or s2:
        return CoordProjection(-1, -1, None, None, None, True,
                                "crosses_spacer" if s1 != s2 else "spacer",
                                f"concat[{concat_lo}:{concat_hi}) touches spacer")
    if f1 != f2:
        return CoordProjection(f1, o1, None, None, None, True,
                                "crosses_fragment_boundary",
                                f"concat[{concat_lo}:{concat_hi}) spans frag {f1}→{f2}")
    src = sources[f1]
    if src.status == "missing":
        return CoordProjection(f1, o1, None, None, None, False,
                                "missing_source",
                                f"fragment {f1} not located in occupied")

    # Within-fragment bounds check: o1 + L must fit inside the fragment
    if o1 < 0 or o1 + L > src.length:
        return CoordProjection(f1, o1, None, None, None, True,
                                "out_of_fragment_bounds",
                                f"offset o1={o1} + L={L} exceeds "
                                f"fragment length {src.length}")

    if src.strand == "+":
        occ_lo = src.occupied_lo + o1
        occ_hi = occ_lo + L
    else:
        occ_hi = src.occupied_hi - o1
        occ_lo = occ_hi - L

    # Occupied-coord bounds check: projected interval must be within
    # [0, src.occupied_hi] (and within [0, occupied_len] if provided).
    hard_hi = occupied_len if occupied_len is not None else src.occupied_hi
    if occ_lo < 0 or occ_hi > hard_hi or occ_lo >= occ_hi:
        return CoordProjection(f1, o1, occ_lo, occ_hi, src.strand, True,
                                "out_of_bounds",
                                f"projected [{occ_lo}, {occ_hi}) outside "
                                f"[0, {hard_hi}]")

    ambiguous = src.status.startswith("ambiguous") or src.status.startswith("partial")
    status = "ok" if src.status == "unique" else "ambiguous_source"
    return CoordProjection(f1, o1, occ_lo, occ_hi, src.strand,
                             ambiguous, status, src.note)
