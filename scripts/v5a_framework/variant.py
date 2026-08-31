"""VariantSpec + run_variant — the pure-function interface for window scoring.

Every axis discussed in the diagnostic phase is a `VariantSpec` field:
    admission        : how a per-site hit is admitted (m-threshold / E-threshold / E-top-k)
    L_mode           : fixed one L or min-over a set
    S_threshold      : number of sites required for a peak
    tau              : Gaussian position kernel width; PER-SITE MAX inside cross-site sum
    orient_constraint: True = require all sites to hit on the SAME orientation
    peak_min_dist    : local-max window when emitting peaks

Adding a new variant = adding a `VariantSpec` value, not a new script.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import numpy as np
from scipy.stats import binom

from .match_table import MatchTable, Orient, EXCL_WIDTHS, SiteRecord


AdmissionRule = Literal["m_threshold", "E_threshold", "E_topk"]
LMode = Literal["fixed", "min_over"]
TSDHandling = Literal["off", "partition"]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    admission: AdmissionRule
    L_mode: LMode
    L_value: tuple[int, ...]        # single-element tuple for fixed, multi for min_over
    threshold: float                # m for m_threshold, E for E_threshold, k for E_topk
    S_threshold: int
    tau: float
    orient_constraint: bool = True
    peak_min_dist: int = 5
    tsd_handling: TSDHandling = "off"     # "partition" -> emit S_outside_TSD alongside S_all;
                                          # never a filter, always a label. Junction masking
                                          # would delete guide target when target_flank_start=0
                                          # so "mask" is intentionally NOT in the enum.

    def key(self) -> str:
        L = "L" + "-".join(str(x) for x in self.L_value)
        oc = "oc" if self.orient_constraint else "np"
        return (f"{self.admission}|{L}|thr={self.threshold}|"
                f"S{self.S_threshold}|tau{self.tau}|{oc}|"
                f"md{self.peak_min_dist}|tsd={self.tsd_handling}")


@dataclass(frozen=True)
class Peak:
    """Detected peak with both scores.

    S_all is the standard cross-site aggregate (excl_w=0).
    S_outside_TSD is the same aggregate computed with each site's admitted
    positions restricted to flank windows starting AT OR AFTER that site's
    detected TSD length. Under tsd_handling="off", it equals S_all.

    Labeling per the acid table:
      S_all high AND S_outside_TSD high -> guide-supported
      S_all high AND S_outside_TSD low  -> TSD-explained
      both low                          -> not detected
    """
    tnp_id: str
    position: int
    S_all: float
    S_outside_TSD: float
    orient: str
    L_at_peak: int

    @property
    def score(self) -> float:
        """Backward-compat alias for S_all."""
        return self.S_all


@lru_cache(maxsize=64)
def _E_table(nc_len: int, flank_len: int, L: int, p: float = 0.25) -> np.ndarray:
    """E-value lookup: E_table[m] for m in 0..L. E = N_windows * P(Bin(L,p) >= m)."""
    Nw = max(1, (nc_len - L + 1) * (flank_len - L + 1))
    arr = np.zeros(L + 1)
    for m in range(L + 1):
        arr[m] = Nw * float(1.0 - binom.cdf(m - 1, L, p))
    return arr


def _admitted_positions(m_array: np.ndarray, L: int, nc_len: int, flank_len: int,
                        rule: AdmissionRule, threshold: float) -> set[int]:
    """Return set of nc positions admitted by the rule at fixed (L, orient, site)."""
    if rule == "m_threshold":
        return {int(p) for p in np.where(m_array >= int(threshold))[0]}
    if rule == "E_threshold":
        tbl = _E_table(nc_len, flank_len, L)
        Es = tbl[np.clip(m_array.astype(np.int32), 0, L)]
        return {int(p) for p in np.where(Es < threshold)[0]}
    if rule == "E_topk":
        tbl = _E_table(nc_len, flank_len, L)
        Es = tbl[np.clip(m_array.astype(np.int32), 0, L)]
        k = int(threshold)
        if k >= len(Es):
            return {int(p) for p in np.where(np.isfinite(Es) & (Es > 0))[0]}
        idx = np.argpartition(Es, k)[:k]
        return {int(p) for p in idx if Es[p] < np.inf}
    raise ValueError(f"unknown admission rule: {rule}")


def _min_over_L_hits(mt: MatchTable, tnp_id: str, site_idx: int, orient: Orient,
                     spec: VariantSpec, flank_len: int,
                     excl_w: int = 0) -> set[int]:
    """For L_mode='min_over': admit position p if ANY L in spec.L_value passes.

    excl_w > 0 restricts each L's m_max to flank offsets f >= excl_w (used for
    the S_outside_TSD partition; see run_variant tsd_handling='partition').
    """
    nc_len = len(mt.tnps[tnp_id].nc)
    if spec.admission == "m_threshold":
        hits: set[int] = set()
        for L in spec.L_value:
            arr = mt.m_max(tnp_id, site_idx, orient, L, excl_w=excl_w)
            hits |= _admitted_positions(arr, L, nc_len, flank_len,
                                        "m_threshold", spec.threshold)
        return hits
    best_E = None
    for L in spec.L_value:
        arr = mt.m_max(tnp_id, site_idx, orient, L, excl_w=excl_w)
        tbl = _E_table(nc_len, flank_len, L)
        Es = tbl[np.clip(arr.astype(np.int32), 0, L)]
        if best_E is None:
            best_E = Es.astype(np.float64)
        else:
            n = min(len(best_E), len(Es))
            best_E[:n] = np.minimum(best_E[:n], Es[:n])
    if best_E is None:
        return set()
    if spec.admission == "E_threshold":
        return {int(p) for p in np.where(best_E < spec.threshold)[0]}
    if spec.admission == "E_topk":
        k = int(spec.threshold)
        if k >= len(best_E):
            return {int(p) for p in np.where(np.isfinite(best_E))[0]}
        idx = np.argpartition(best_E, k)[:k]
        return {int(p) for p in idx if best_E[p] < np.inf}
    raise ValueError(f"unknown admission rule: {spec.admission}")


def _fixed_L_hits(mt: MatchTable, tnp_id: str, site_idx: int, orient: Orient,
                  spec: VariantSpec, flank_len: int,
                  excl_w: int = 0) -> set[int]:
    L = spec.L_value[0]
    nc_len = len(mt.tnps[tnp_id].nc)
    arr = mt.m_max(tnp_id, site_idx, orient, L, excl_w=excl_w)
    return _admitted_positions(arr, L, nc_len, flank_len, spec.admission, spec.threshold)


def _tsd_excl_w_for_site(site: SiteRecord,
                          excl_widths: tuple[int, ...] = EXCL_WIDTHS) -> int:
    """Pick the smallest excl_w >= detected TSD length. w=0 if no TSD (which is
    ALSO the Durrant case: upstream_flank is None -> no both-junction context
    -> no TSD detection -> no partition; S_outside_TSD == S_all).

    This is what prevents Durrant's target_flank_start=0 case from being
    silently labeled TSD-explained.
    """
    if site.upstream_flank is None:
        return 0
    from .flank_coherence import within_site_tsd
    tsd = within_site_tsd(site)
    if tsd is None:
        return 0
    for w in sorted(excl_widths):
        if w >= tsd.tsd_length:
            return w
    return max(excl_widths)


def _apply_kernel_max(hits_lists: list[set[int]], nc_len_pos: int,
                      tau: float) -> np.ndarray:
    """Sum over sites of PER-SITE MAX of Gaussian bumps. Semantic fix from X1'.

    Per-site max INSIDE the sum ensures a single hit does not double-count when
    a nearby position also has a hit at the same site.
    """
    S = np.zeros(nc_len_pos, dtype=np.float64)
    if tau <= 0:
        for h in hits_lists:
            per_site = np.zeros(nc_len_pos)
            for pos in h:
                if 0 <= pos < nc_len_pos:
                    per_site[pos] = 1.0
            S += per_site
        return S
    radius = int(np.ceil(3 * tau))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / tau) ** 2)
    for h in hits_lists:
        per_site = np.zeros(nc_len_pos)
        for pos in h:
            lo = max(0, pos - radius)
            hi = min(nc_len_pos, pos + radius + 1)
            k_lo = lo - (pos - radius)
            k_hi = k_lo + (hi - lo)
            np.maximum(per_site[lo:hi], kernel[k_lo:k_hi], out=per_site[lo:hi])
        S += per_site
    return S


def _find_peaks(S: np.ndarray, thresh: float, min_dist: int) -> list[int]:
    peaks: list[int] = []
    L = len(S)
    for i in range(L):
        if S[i] < thresh:
            continue
        lo = max(0, i - min_dist)
        hi = min(L, i + min_dist + 1)
        is_max = True
        for j in range(lo, hi):
            if j != i and S[j] > S[i]:
                is_max = False
                break
        if is_max:
            peaks.append(i)
    return peaks


def _site_hits_for_orient(mt: MatchTable, tnp_id: str, orient: Orient,
                          spec: VariantSpec,
                          per_site_excl_w: list[int] | None = None
                          ) -> list[set[int]]:
    """Per-site admitted-position sets.

    per_site_excl_w: optional list of per-site TSD-exclusion widths (one per
    site of the Tnp). When None, uses excl_w=0 for every site (S_all path).
    """
    tnp = mt.tnps[tnp_id]
    out: list[set[int]] = []
    for i, s in enumerate(tnp.sites):
        flank_len = len(s.flank)
        w = per_site_excl_w[i] if per_site_excl_w is not None else 0
        if spec.L_mode == "fixed":
            h = _fixed_L_hits(mt, tnp_id, s.site_idx, orient, spec, flank_len, excl_w=w)
        else:
            h = _min_over_L_hits(mt, tnp_id, s.site_idx, orient, spec, flank_len, excl_w=w)
        out.append(h)
    return out


def _aggregate_S_over_orients(mt: MatchTable, tnp_id: str, spec: VariantSpec,
                                nc_len_pos: int,
                                per_site_excl_w: list[int] | None = None
                                ) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    """Return (S_pooled, per_orient_S, per_orient_names) for one Tnp."""
    per_orient_S: list[np.ndarray] = []
    per_orient_names: list[str] = []
    for orient in mt.orients:
        hits = _site_hits_for_orient(mt, tnp_id, orient, spec, per_site_excl_w)
        per_orient_S.append(_apply_kernel_max(hits, nc_len_pos, spec.tau))
        per_orient_names.append(orient)
    S_pooled = np.max(np.stack(per_orient_S), axis=0)
    return S_pooled, per_orient_S, per_orient_names


def run_variant(mt: MatchTable, spec: VariantSpec) -> dict[str, list[Peak]]:
    """Score every Tnp in the table under one VariantSpec. Pure function.

    Emits Peak with both S_all (excl_w=0) and S_outside_TSD (per-site excl_w
    picked from within_site_tsd). When spec.tsd_handling='off', S_outside_TSD
    is set equal to S_all (no partition, no re-score, no wasted computation
    because we return early).

    Peak-finding uses S_all only — the partition never drops candidates.
    """
    ref_L = spec.L_value[0]
    peaks_by_tnp: dict[str, list[Peak]] = {}
    kernel_thresh = float(spec.S_threshold) - 0.5 if spec.tau > 0 else float(spec.S_threshold)

    for tnp_id in mt.tnp_ids:
        tnp = mt.tnps[tnp_id]
        nc_len_pos = len(tnp.nc) - ref_L + 1
        if nc_len_pos <= 0:
            peaks_by_tnp[tnp_id] = []
            continue

        # --- S_all pass (excl_w=0 for all sites) ---
        if spec.orient_constraint:
            per_orient_S_all: list[np.ndarray] = []
            per_orient_names: list[str] = []
            for orient in mt.orients:
                hits = _site_hits_for_orient(mt, tnp_id, orient, spec, None)
                per_orient_S_all.append(_apply_kernel_max(hits, nc_len_pos, spec.tau))
                per_orient_names.append(orient)
        else:
            _, per_orient_S_all, per_orient_names = _aggregate_S_over_orients(
                mt, tnp_id, spec, nc_len_pos, None)

        # --- Optional S_outside_TSD pass ---
        per_orient_S_out: list[np.ndarray] | None = None
        if spec.tsd_handling == "partition":
            per_site_excl_w = [_tsd_excl_w_for_site(s) for s in tnp.sites]
            if any(w > 0 for w in per_site_excl_w):
                per_orient_S_out = []
                for orient in mt.orients:
                    hits = _site_hits_for_orient(mt, tnp_id, orient, spec, per_site_excl_w)
                    per_orient_S_out.append(_apply_kernel_max(hits, nc_len_pos, spec.tau))
            # else: no TSD detected on any site -> S_outside_TSD == S_all (leave None,
            # copy at Peak-emit time). This is the Durrant path.

        # --- Peak emission on S_all ---
        peaks: list[Peak] = []
        if spec.orient_constraint:
            for oi, orient in enumerate(mt.orients):
                S_all_o = per_orient_S_all[oi]
                for p in _find_peaks(S_all_o, kernel_thresh, spec.peak_min_dist):
                    S_out = float(per_orient_S_out[oi][p]) if per_orient_S_out else float(S_all_o[p])
                    peaks.append(Peak(tnp_id=tnp_id, position=int(p),
                                      S_all=float(S_all_o[p]),
                                      S_outside_TSD=S_out,
                                      orient=orient, L_at_peak=ref_L))
        else:
            S_all_pooled = np.max(np.stack(per_orient_S_all), axis=0)
            argmax_orient = np.argmax(np.stack(per_orient_S_all), axis=0)
            if per_orient_S_out is not None:
                S_out_pooled = np.max(np.stack(per_orient_S_out), axis=0)
            else:
                S_out_pooled = S_all_pooled
            for p in _find_peaks(S_all_pooled, kernel_thresh, spec.peak_min_dist):
                peaks.append(Peak(tnp_id=tnp_id, position=int(p),
                                  S_all=float(S_all_pooled[p]),
                                  S_outside_TSD=float(S_out_pooled[p]),
                                  orient=per_orient_names[int(argmax_orient[p])],
                                  L_at_peak=ref_L))

        peaks_by_tnp[tnp_id] = peaks
    return peaks_by_tnp


# ---------- canonical spec catalog ----------

def spec_m_threshold_L11(m: int = 8, tau: float = 0, S: int = 5,
                          orient_constraint: bool = True) -> VariantSpec:
    """Historical Channel A baseline: fixed L=11, m>=8."""
    return VariantSpec(
        name=f"m_thresh_L11_m{m}_tau{tau}_S{S}",
        admission="m_threshold", L_mode="fixed", L_value=(11,),
        threshold=float(m), S_threshold=S, tau=tau,
        orient_constraint=orient_constraint,
    )


def spec_min_E_9_12(E: float = 4.0, tau: float = 0, S: int = 5,
                     orient_constraint: bool = True) -> VariantSpec:
    """Mode 1: min-E over L in {9..12} with E-threshold admission."""
    return VariantSpec(
        name=f"min_E_9_12_E{E}_tau{tau}_S{S}",
        admission="E_threshold", L_mode="min_over", L_value=(9, 10, 11, 12),
        threshold=float(E), S_threshold=S, tau=tau,
        orient_constraint=orient_constraint,
    )


def spec_min_E_9_12_topk(k: int = 5, tau: float = 0, S: int = 5,
                          orient_constraint: bool = True) -> VariantSpec:
    """New: min-E over L in {9..12} with top-k admission per site.

    Tests whether Mode 1's tau-invariance is narrow-funnel (E<4 filter leaves
    <4 candidates) vs. precise-position. Widening admission to top-k = 5
    should decouple funnel width from position precision.
    """
    return VariantSpec(
        name=f"min_E_9_12_top{k}_tau{tau}_S{S}",
        admission="E_topk", L_mode="min_over", L_value=(9, 10, 11, 12),
        threshold=float(k), S_threshold=S, tau=tau,
        orient_constraint=orient_constraint,
    )
