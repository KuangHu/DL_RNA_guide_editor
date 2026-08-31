"""flank_coherence — cross-site motif conservation (primary) + within-site TSD (secondary).

Motivation:
    The FP path for Channel A on non-guided multi-site elements is:
        5 sites' junction-adjacent flanks share a short motif
        -> that motif matches some ncRNA position
        -> all 5 sites hit that position -> S=5.
    The confounder is CROSS-SITE MOTIF CONSERVATION, not TSD per se. TSD is
    one measurable cause (IS10-R, IS903, ISLdl1). Site-specific conservation
    without TSD (ISAjo2 pdif) triggers the same path.

Two outputs, both per Tnp / per insertion, no filters:
    1. cross_site_motif_conservation(tnp) -> ConservationReport
       Positional consensus over N sites' junction-anchored ±15 nt windows.
       Reports overall score + POSITION DISTRIBUTION (concentrated vs dispersed).
    2. within_site_tsd(insertion) -> TSDReport | None
       Longest exact direct repeat between upstream last 15 nt and downstream
       first 15 nt of the SAME physical insertion. Returns None where both-
       junction context unavailable (e.g. Durrant target-at-start records).

Both are diagnostic. Never used as a filter — see variant.py `tsd_handling`
(`"off"` or `"partition"`; `"mask"` is deliberately not in the enum because
junction-anchored masking deletes guide target when they overlap, as in
Durrant's 73% target_flank_start=0 case).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .match_table import MatchTable, SiteRecord


WINDOW_HALF = 15         # ±15 nt around junction / target
MIN_TSD_LEN = 4          # shortest exact direct repeat to consider a TSD hit
MAX_TSD_LEN = 12         # upper bound on TSD length in the 5 negative families

UNIFORM_BG = np.array([0.25, 0.25, 0.25, 0.25])   # A, C, G, T


@dataclass(frozen=True)
class ConservationReport:
    """Positional-consensus conservation over N sites of one Tnp.

    Information content computed against a NAMED background composition —
    `bg_source` records which q was used (uniform, per-family, or per-record).
    A 100%-A position under uniform bg gives log(4)=1.39 nats; under an AT-rich
    bg (q_A=0.35) it gives log(1/0.35)=1.05 nats. Same conservation, different
    information — hence bg must be reported alongside the score.
    """
    tnp_id: str
    n_sites: int
    window_len: int
    per_position_modal_freq: list[float]   # length window_len
    per_position_information: list[float]  # sum_{b} p_b log(p_b / q_b); nats
    per_position_freqs: list[list[float]]  # window_len x 4 (A,C,G,T); needed for fp_hazard
    total_conservation: float              # sum of per-position (modal_freq - max(q_b)) clipped >=0
    concentrated_score: float              # max_over_length_9_substrings of window_mean_information
    dispersed_score: float                 # mean information over positions with information < concentrated peak
    concentration_ratio: float             # concentrated_score / max(dispersed_score, 1e-9)
    argmax_position: int                   # window offset where concentrated peak lives
    bg_source: str                         # "uniform" | "family:{name}" | "record"
    bg_freqs: list[float]                  # [A, C, G, T]


@dataclass(frozen=True)
class TSDReport:
    """Within-site direct-repeat detection at one physical insertion."""
    site_idx: int
    tsd_seq: str
    tsd_length: int
    upstream_offset: int    # position in upstream_flank[-15:] where TSD starts (0..14)
    downstream_offset: int  # position in downstream_flank[:15] where TSD starts (0..14)
    exact_match: bool


# ---------- cross-site motif conservation ----------

_BASE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


def _junction_window(site: SiteRecord, half: int = WINDOW_HALF) -> str:
    """Extract the junction-anchored window from a site.

    Durrant positive: flank[target_flank_start : target_flank_start + half]
                      (target position defines junction; 73% at position 0).
    Negative:         flank[0 : half]
                      (downstream flank starts at junction).
    """
    if site.target_flank_start is not None:
        s = int(site.target_flank_start)
        return site.flank[s : s + half]
    return site.flank[:half]


def _positional_consensus(windows: list[str], window_len: int,
                           q_bg: np.ndarray = UNIFORM_BG
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Return (modal_freq[pos], information[pos]) over windows.

    modal_freq[pos] = fraction of sites with the modal base at pos.
    information[pos] = sum_b p_b * log(p_b / q_b[b]) in nats.
                       (KL divergence to `q_bg`.)

    q_bg is the background base composition [A, C, G, T]. Uniform (0.25 each)
    treats every base as equally surprising; a per-family bg like ISLdl1's
    AT-rich composition would correctly deflate a 100%-A position's information
    from ln(4)=1.39 to ln(1/q_A)~1.05 nats.
    """
    counts = np.zeros((window_len, 4), dtype=np.float64)
    n = 0
    for w in windows:
        if len(w) < window_len:
            continue
        for pos in range(window_len):
            b = _BASE_INDEX.get(w[pos].upper())
            if b is not None:
                counts[pos, b] += 1
        n += 1
    if n == 0:
        return (np.full(window_len, q_bg.max()), np.zeros(window_len),
                np.tile(q_bg, (window_len, 1)))
    freqs = counts / max(1, n)
    modal = freqs.max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        info = (freqs * np.log(np.clip(freqs / q_bg[None, :], 1e-12, None))).sum(axis=1)
    return modal, info, freqs


def family_base_composition(mt: MatchTable, family: str) -> np.ndarray:
    """Compute background base composition [A, C, G, T] over ALL flanks of
    all Tnps in one family.

    For negatives, this uses the downstream flanks (the ones actually stored
    in SiteRecord.flank). Bases outside ACGT are excluded from the denominator.
    """
    counts = np.zeros(4, dtype=np.float64)
    total = 0
    for tnp_id in mt.tnp_ids:
        if mt.tnps[tnp_id].family != family:
            continue
        for s in mt.tnps[tnp_id].sites:
            for base in s.flank:
                b = _BASE_INDEX.get(base.upper())
                if b is not None:
                    counts[b] += 1
                    total += 1
    if total == 0:
        return UNIFORM_BG.copy()
    return counts / total


def cross_site_motif_conservation(mt: MatchTable, tnp_id: str,
                                   site_indices: list[int] | None = None,
                                   half: int = WINDOW_HALF,
                                   q_bg: np.ndarray | None = None,
                                   bg_source: str = "uniform"
                                   ) -> ConservationReport:
    """Positional-consensus conservation across sites of one Tnp.

    q_bg: background base composition [A, C, G, T]. Pass a per-family
    composition (from family_base_composition) for the ISLdl1 vs IS10-R
    acid test; leaving it None uses uniform (0.25 each) — that will
    OVERESTIMATE conservation on AT-biased families like ISLdl1.
    """
    if q_bg is None:
        q_bg = UNIFORM_BG
    tnp = mt.tnps[tnp_id]
    if site_indices is None:
        site_indices = list(range(len(tnp.sites)))
    windows = [_junction_window(tnp.sites[i], half) for i in site_indices]
    modal, info, freqs = _positional_consensus(windows, half, q_bg)
    total = float(np.clip(modal - float(q_bg.max()), 0, None).sum())

    # Concentrated vs dispersed: sliding-9-mean of information vs off-peak mean.
    kern = 9
    if len(info) >= kern:
        conv = np.convolve(info, np.ones(kern) / kern, mode="valid")
        argmax_pos = int(np.argmax(conv))
        concentrated = float(conv[argmax_pos])
        # dispersed = mean information at positions >= kern away from the peak
        far_mask = np.ones(len(info), dtype=bool)
        far_lo = max(0, argmax_pos - kern // 2)
        far_hi = min(len(info), argmax_pos + kern // 2 + 1)
        far_mask[far_lo:far_hi] = False
        dispersed = float(info[far_mask].mean()) if far_mask.any() else 0.0
    else:
        concentrated = float(info.mean()) if len(info) else 0.0
        dispersed = 0.0
        argmax_pos = 0
    ratio = concentrated / max(1e-9, abs(dispersed) if abs(dispersed) > 1e-9
                                else 1e-9)

    return ConservationReport(
        tnp_id=tnp_id, n_sites=len(site_indices), window_len=half,
        per_position_modal_freq=modal.tolist(),
        per_position_information=info.tolist(),
        per_position_freqs=freqs.tolist(),
        total_conservation=total,
        concentrated_score=concentrated,
        dispersed_score=dispersed,
        concentration_ratio=ratio,
        argmax_position=argmax_pos,
        bg_source=bg_source,
        bg_freqs=q_bg.tolist(),
    )


# ---------- within-site TSD detection ----------

def _all_kmers_in(seq: str, k: int) -> dict[str, list[int]]:
    """{kmer: [positions_in_seq]} for kmers of length k."""
    out: dict[str, list[int]] = {}
    for i in range(len(seq) - k + 1):
        km = seq[i : i + k].upper()
        if any(b not in "ACGT" for b in km):
            continue
        out.setdefault(km, []).append(i)
    return out


def within_site_tsd(site: SiteRecord,
                     half: int = WINDOW_HALF,
                     min_len: int = MIN_TSD_LEN,
                     max_len: int = MAX_TSD_LEN) -> TSDReport | None:
    """Longest exact direct repeat between upstream_flank[-half:] and
    downstream_flank[:half].

    A TSD is a direct repeat (both copies same strand, same orientation)
    flanking the insertion. Returns the longest such repeat length (>= min_len).
    Returns None if:
      - upstream_flank is None (e.g. Durrant target-at-start records)
      - no repeat of length >= min_len exists
    """
    if site.upstream_flank is None:
        return None
    up_window = site.upstream_flank[-half:]
    dn_window = site.flank[:half]
    if len(up_window) < min_len or len(dn_window) < min_len:
        return None

    upper = min(max_len, len(up_window), len(dn_window))
    best: TSDReport | None = None
    # Longest-first exact-match scan
    for k in range(upper, min_len - 1, -1):
        up_kmers = _all_kmers_in(up_window, k)
        dn_kmers = _all_kmers_in(dn_window, k)
        common = set(up_kmers) & set(dn_kmers)
        if not common:
            continue
        # Prefer the direct-repeat with smallest gap (upstream near end, downstream near start)
        best_km = None
        best_gap = None
        best_up_off = -1
        best_dn_off = -1
        for km in common:
            for uo in up_kmers[km]:
                for do in dn_kmers[km]:
                    # Gap = length(upstream after km) + do
                    up_dist_to_junction = (len(up_window) - (uo + k))
                    gap = up_dist_to_junction + do
                    if best_gap is None or gap < best_gap:
                        best_gap = gap
                        best_km = km
                        best_up_off = uo
                        best_dn_off = do
        if best_km is not None:
            best = TSDReport(
                site_idx=site.site_idx, tsd_seq=best_km, tsd_length=k,
                upstream_offset=best_up_off,
                downstream_offset=best_dn_off,
                exact_match=True,
            )
            break
    return best


# ---------- diagnostic aggregation ----------

@dataclass(frozen=True)
class FlankCoherenceReport:
    """Per-family aggregate. Used for the P2 danger-prediction table."""
    family: str
    n_tnps: int
    conservation_by_tnp: dict[str, ConservationReport] = field(default_factory=dict)
    tsd_by_insertion: dict[tuple[str, int], TSDReport | None] = field(default_factory=dict)

    def total_conservation_dist(self) -> np.ndarray:
        return np.array([r.total_conservation
                          for r in self.conservation_by_tnp.values()])

    def concentration_ratio_dist(self) -> np.ndarray:
        return np.array([r.concentration_ratio
                          for r in self.conservation_by_tnp.values()])

    def tsd_length_dist(self) -> np.ndarray:
        lengths = [r.tsd_length for r in self.tsd_by_insertion.values()
                    if r is not None]
        return np.array(lengths) if lengths else np.zeros(0)

    def tsd_present_fraction(self) -> float:
        n_total = len(self.tsd_by_insertion)
        n_hit = sum(1 for r in self.tsd_by_insertion.values() if r is not None)
        return n_hit / max(1, n_total)


def flank_coherence_family(mt: MatchTable, family: str,
                            site_indices_by_tnp: dict[str, list[int]] | None = None,
                            use_family_bg: bool = True,
                            ) -> FlankCoherenceReport:
    """Run cross-site conservation + within-site TSD across all Tnps of a family.

    use_family_bg=True (default) computes the family's background composition
    once and passes it into every conservation call. This is what separates
    "true motif conservation" (IS10-R NGCTNAGCN) from "AT-bias inflating
    modal frequency" (ISLdl1) — the acid test the danger table asks.
    """
    q_bg = family_base_composition(mt, family) if use_family_bg else UNIFORM_BG
    bg_source = f"family:{family}" if use_family_bg else "uniform"
    rep = FlankCoherenceReport(family=family, n_tnps=0)
    n = 0
    for tnp_id in mt.tnp_ids:
        if mt.tnps[tnp_id].family != family:
            continue
        sel = site_indices_by_tnp.get(tnp_id) if site_indices_by_tnp else None
        if site_indices_by_tnp is not None and sel is None:
            continue
        cons = cross_site_motif_conservation(mt, tnp_id, sel, q_bg=q_bg,
                                              bg_source=bg_source)
        rep.conservation_by_tnp[tnp_id] = cons
        indices = sel if sel is not None else list(range(len(mt.tnps[tnp_id].sites)))
        for si in indices:
            site = mt.tnps[tnp_id].sites[si]
            rep.tsd_by_insertion[(tnp_id, si)] = within_site_tsd(site)
        n += 1
    return FlankCoherenceReport(
        family=family, n_tnps=n,
        conservation_by_tnp=rep.conservation_by_tnp,
        tsd_by_insertion=rep.tsd_by_insertion,
    )


# ---------- fp_hazard: expected FP-per-nc-position given consensus x nc composition ----------

@dataclass(frozen=True)
class HazardReport:
    """Joint FP hazard: expected # nc positions where all S sites hit m>=threshold.

    joint_hazard = sum over nc windows w of g(w) ** S_threshold, where
      g(w) = P(random flank drawn from consensus has >= threshold matches
             to nc[w:w+L])
    computed as the Poisson-binomial tail with per-position match probs
      p_i(w) = f_consensus,i[nc[w+i]]
    (the probability that a random draw from the consensus distribution at
    position i produces the nc base at that position).

    marginal_hazard = sum over nc windows of g(w). Reported as a diagnostic
    only. joint_hazard = marginal_hazard iff g(w) in {0, 1} for every w
    (perfect conservation) — the two coincide there and diverge under partial
    conservation by orders of magnitude. Case 2 (partial-A consensus vs
    neutral nc) will now be several orders of magnitude LOWER than case 3
    (100%-consensus vs the same nc), matching the correlation-through-consensus
    reality.

    Relationship to motif_info (from ConservationReport): motif_info answers
    "is this a real motif" (KL vs bg); joint_hazard answers "how dangerous is
    it for Channel A" (joint expected FP count under an iid-consensus model).
    """
    tnp_id: str
    L: int                             # match window length actually used
    threshold: int                     # m_min per single-flank hit
    S_threshold: int                   # sites required at same position
    consensus_len: int
    mean_match_prob: float             # avg over positions of E[match | nc bg]
    marginal_hazard: float             # sum_w g(w)   (diagnostic)
    joint_hazard: float                # sum_w g(w)^S (the P2 stratifier)
    nc_len: int


def _nc_composition_from_tnp(mt: MatchTable, tnp_id: str) -> np.ndarray:
    counts = np.zeros(4, dtype=np.float64)
    total = 0
    for base in mt.tnps[tnp_id].nc:
        b = _BASE_INDEX.get(base.upper())
        if b is not None:
            counts[b] += 1; total += 1
    if total == 0:
        return UNIFORM_BG.copy()
    return counts / total


def _poisson_binom_tail(probs: np.ndarray, threshold: int) -> float:
    """P(sum of independent Bernoullis(probs) >= threshold), exact.

    O(L^2) dynamic programming — fine at L=9..12.
    """
    L = len(probs)
    if L == 0:
        return 0.0
    dp = np.zeros(L + 1)
    dp[0] = 1.0
    for p in probs:
        # Update in place, high to low so we do not overwrite dp[k-1] before use.
        for k in range(L, 0, -1):
            dp[k] = dp[k] * (1 - p) + dp[k - 1] * p
        dp[0] *= (1 - p)
    return float(dp[threshold:].sum())


def fp_hazard(cons: ConservationReport, nc: str, nc_freqs: np.ndarray,
              L: int = 11, threshold: int = 8, S_threshold: int = 5,
              consensus_slice: tuple[int, int] | None = None) -> HazardReport:
    """Joint FP hazard: expected # nc positions where all S sites hit m>=threshold.

    For each nc window w:
      1. Compute per-position match probability p_i(w) = f_consensus,i[nc[w+i]]
      2. g(w) = P(sum of Bernoullis(p_i) >= threshold)   (Poisson-binomial tail)
      3. add g(w) ** S_threshold to the joint sum

    joint_hazard = sum over windows of g(w)^S. This is the correlation-through-
    consensus estimate: under 100% conservation g(w) in {0,1} and joint =
    marginal; under partial conservation g^S << g.
    """
    freqs = np.array(cons.per_position_freqs)   # (window_len, 4)
    W = freqs.shape[0]
    if consensus_slice is None:
        half_L = L // 2
        c = cons.argmax_position
        lo = max(0, c - half_L)
        hi = min(W, lo + L)
        lo = max(0, hi - L)
    else:
        lo, hi = consensus_slice
    consensus_L = freqs[lo:hi]                  # (L_eff, 4)
    L_eff = consensus_L.shape[0]
    if L_eff == 0 or len(nc) < L_eff:
        return HazardReport(tnp_id=cons.tnp_id, L=L, threshold=threshold,
                             S_threshold=S_threshold, consensus_len=0,
                             mean_match_prob=0.0, marginal_hazard=0.0,
                             joint_hazard=0.0, nc_len=len(nc))
    # Encode nc as integer array once.
    nc_int = np.array([_BASE_INDEX.get(b.upper(), -1) for b in nc], dtype=np.int32)

    marginal = 0.0
    joint = 0.0
    total_mean_p = 0.0
    n_windows = 0
    for w_start in range(len(nc) - L_eff + 1):
        # p_i(w) = 0 if nc[w+i] is N (unknown), else consensus_L[i, nc_base_idx]
        base_ids = nc_int[w_start : w_start + L_eff]
        p_per_pos = np.zeros(L_eff)
        valid = base_ids >= 0
        p_per_pos[valid] = consensus_L[np.arange(L_eff)[valid], base_ids[valid]]
        g_w = _poisson_binom_tail(p_per_pos, threshold)
        marginal += g_w
        joint += g_w ** S_threshold
        total_mean_p += float(p_per_pos.mean())
        n_windows += 1

    mean_p = total_mean_p / max(1, n_windows)
    return HazardReport(
        tnp_id=cons.tnp_id, L=L_eff, threshold=threshold,
        S_threshold=S_threshold, consensus_len=L_eff,
        mean_match_prob=mean_p,
        marginal_hazard=marginal, joint_hazard=joint,
        nc_len=len(nc),
    )


# ---------- delta_m_max: junction-region contribution to m_max ----------

@dataclass(frozen=True)
class DeltaMRow:
    """One family row of the 6-row junction-contribution diagnostic.

    Reports TWO Delta values so cross-family comparability and family-specific
    resolution are both preserved (per user spec):
      w_uniform : excl width common to all rows (default 9) — comparable.
      w_native  : family's actual TSD width (2 for IS30, 8 for ISLdl1, ...) —
                  the honest per-family number.
    """
    family: str
    n_tnps: int
    n_nc_positions: int
    # w_uniform column (comparable)
    w_uniform: int
    mean_delta_m_uniform: float
    median_delta_m_uniform: float
    frac_delta_positive_uniform: float
    # w_native column (honest per-family)
    w_native: int
    mean_delta_m_native: float
    median_delta_m_native: float
    frac_delta_positive_native: float


# Native TSD widths per family (from ISfinder README / literature).
FAMILY_TSD_WIDTH: dict[str, int] = {
    "durrant_positive": 0,   # IS110: no characteristic TSD
    "IS10-R": 9,             # NGCTNAGCN consensus
    "IS30":   2,             # short
    "IS903":  9,             # no consensus but 9 bp
    "ISAjo2": 0,             # no TSD, pdif-associated
    "ISLdl1": 8,             # AT-rich
}


def _delta_m_stats(mt: MatchTable, family: str, excl_w: int, L: int,
                   orient: str) -> tuple[int, int, float, float, float]:
    """Aggregate Delta_m = m_max(excl_0) - m_max(excl_w) across all Tnps of
    the family. Returns (n_tnps, n_positions, mean, median, frac_positive).
    """
    all_deltas: list[np.ndarray] = []
    n_tnps = 0
    for tnp_id in mt.tnp_ids:
        if mt.tnps[tnp_id].family != family:
            continue
        for s in mt.tnps[tnp_id].sites:
            m0 = mt.m_max(tnp_id, s.site_idx, orient, L, excl_w=0)
            mw = mt.m_max(tnp_id, s.site_idx, orient, L, excl_w=excl_w)
            n = min(len(m0), len(mw))
            if n > 0:
                all_deltas.append(m0[:n].astype(np.int32) - mw[:n].astype(np.int32))
        n_tnps += 1
    if not all_deltas:
        return n_tnps, 0, 0.0, 0.0, 0.0
    d = np.concatenate(all_deltas)
    return (n_tnps, int(d.size), float(d.mean()),
            float(np.median(d)), float((d > 0).mean()))


def delta_m_max_by_family(mt: MatchTable, family: str,
                           w_uniform: int = 9, L: int = 11,
                           orient: str = "fwd") -> DeltaMRow:
    """Two-column diagnostic: uniform width (default 9, for cross-family
    comparability) and native family TSD width (for the honest per-family
    number). Delta > 0 means m_max at that nc position was achieved using a
    flank window that OVERLAPS the excluded band, i.e. junction-driven.
    """
    n_u, npos_u, mean_u, med_u, frac_u = _delta_m_stats(mt, family, w_uniform, L, orient)
    w_native = FAMILY_TSD_WIDTH.get(family, 0)
    if w_native == 0:
        n_n, npos_n, mean_n, med_n, frac_n = n_u, npos_u, 0.0, 0.0, 0.0
    else:
        n_n, npos_n, mean_n, med_n, frac_n = _delta_m_stats(mt, family, w_native, L, orient)
    return DeltaMRow(
        family=family, n_tnps=n_u, n_nc_positions=npos_u,
        w_uniform=w_uniform, mean_delta_m_uniform=mean_u,
        median_delta_m_uniform=med_u, frac_delta_positive_uniform=frac_u,
        w_native=w_native, mean_delta_m_native=mean_n,
        median_delta_m_native=med_n, frac_delta_positive_native=frac_n,
    )


def delta_m_max_report(mt_pos: MatchTable, mt_negs: dict[str, MatchTable],
                        w_uniform: int = 9, L: int = 11) -> list[DeltaMRow]:
    """Six-row diagnostic: Durrant + 5 negative families.

    Durrant expected ~0 (IS110 has no TSD; target-at-flank-position-0 means
    the "junction band" IS the guide-target region — but m_max should be
    dominated by the real guide-nc alignment regardless of which flank
    offset achieved it, so restricting to f >= 9 should not change the
    per-position maximum much on Durrant). Non-zero Delta on Durrant would
    warn that excl_w has an asymmetric effect between Durrant and negatives.

    Negatives expected elevated per family:
      IS10-R  : 9-bp NGCTNAGCN consensus  -> largest Delta
      IS903   : 9-bp no consensus         -> small (TSDs vary across sites)
      IS30    : 2-bp                       -> minimal
      ISLdl1  : 8-bp AT-rich               -> moderate
      ISAjo2  : no TSD, pdif-associated    -> should be small (no junction match, but conserved site)
    """
    rows: list[DeltaMRow] = []
    fam_pos = next(iter(set(t.family for t in mt_pos.tnps.values())), "durrant_positive")
    rows.append(delta_m_max_by_family(mt_pos, fam_pos, w_uniform=w_uniform, L=L))
    for fam, mt_neg in mt_negs.items():
        rows.append(delta_m_max_by_family(mt_neg, fam, w_uniform=w_uniform, L=L))
    return rows


# ---------- S_outside_TSD helper (used by variant.py) ----------

def tsd_span_for_site(site: SiteRecord) -> tuple[int, int] | None:
    """Return (start, end) offset in the DOWNSTREAM flank where TSD lives,
    or None if no TSD detected.

    Used at scoring time to compute S_outside_TSD: a per-site hit at argmax
    flank offset f is only counted toward the "outside_TSD" score if f is
    OUTSIDE (start, end).
    """
    tsd = within_site_tsd(site)
    if tsd is None:
        return None
    return (tsd.downstream_offset, tsd.downstream_offset + tsd.tsd_length)
