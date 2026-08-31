"""features_structure_v2 — BPP-based structural features for ncRNAs.

Replaces the RNAplfold-based ``preprocess/structure.py`` pipeline. The
old pipeline runs `RNAplfold -W 120 -L 60 -u 16` and reports 17 channels
per position (P(unpaired for stretch u=1..16), plus a valid mask). Two
diagnostics falsified that setup on real bridge RNAs (2026-08-31):

  D3 : median max_pair_span / nc_len = 0.83 across T-WT + 5 seekRNA NCRs.
       -L 60 is architecturally blind on 6/6 real ncRNAs.
  D4': median F1 = 0.636 with base-pair-constrained IUPAC sampling — fold
       prediction is a genuine MEDIUM prior, moderate channel weight
       justified (D4's 0.16 was IUPAC-realization artefact, not MFE
       inherent error).
  48CS-A audit: P(delta>0)=0.802 survives matched-L stratification, so
       structural design signal is real; rebuild is worth 4 h.

Channels emitted per nc position (compact 4-channel replacement of the
old 17-channel accessibility profile):

  ch0  dG_open_u1[i] = -RT * ln P(unpaired[i])
       Per-position free energy to open one base at i.

  ch1  dG_open_uL_pn[i] = (-RT * sum_{k in win_L(i)} ln P(unpaired[k])) / L
       Per-nt normalized dG to open the L-length window starting at i.
       *Per-nt* because dG_open_uL is otherwise not comparable across
       different guide lengths (same failure mode as log_tail — see
       [[finding-v5a2-null-ceiling]] and [[finding-diagnostics-d-a9-a13]]).
       When L changes, always normalize to per-nt units.

  ch2  E_span_win[i] = sum_{i',j} P(i',j) * |i'-j| /
                        sum_{i',j} P(i',j)  over i' in win_L(i), j != i'
       Weighted-mean pair-partner distance across the whole L-window.
       Weighted-by-pairing-mass means unpaired positions contribute
       negligibly, so a window that is entirely unpaired returns NaN
       (an informative signal) rather than being conflated with
       "paired at distance 0". User-directed decision, 2026-08-31.

  ch3  H_pair_win[i] = mean over i' in win_L(i) of pair-partner entropy
                        H(i') = -sum_j q(j|i') ln q(j|i')
                        where q(j|i') = P(i',j) / sum_j P(i',j)
       Higher H = more diffuse pairing partners = less-defined structure.
       Positions with sum_j P(i',j) < eps contribute 0 and their weight
       is 0 (same handling as ch2).

L-comparability caveat (repeated for emphasis): channels ch1 and channels
computed at different guide lengths are NOT equivalent. For the current
generator baseline the guide length is fixed at L=11. When difficulty
axes add L in {12, 13, 14}, either add a per-L channel bank or scale
by L (per-nt normalization already applied on ch1 for that reason).

Everything runs at global RNAfold parameters — no window, no L
constraint. On 150-281 nt ncRNAs the cost is ~90-285 ms per sequence
(pf + bpp; A1 + pf() bench, 2026-08-31), so per-bag folding across 50K
bags = ~4 h one-time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import RNA

# ViennaRNA kT at 37 C ≈ 0.6156 kcal/mol.
# fc.pf() returns ensemble energy in the same units.
KT_37C_KCAL_PER_MOL = 0.6156

DEFAULT_GUIDE_LENGTH = 11
EPS_PAIR_MASS = 1e-8


@dataclass(frozen=True)
class StructureFeaturesV2:
    """Per-position BPP-derived structural feature bundle for one ncRNA.

    Shapes:
        dG_open_u1        (nc_len,)
        dG_open_uL_pn     (nc_len - L + 1,)  — window-start indexed
        E_span_win        (nc_len - L + 1,)  — NaN where window all unpaired
        H_pair_win        (nc_len - L + 1,)  — NaN where window all unpaired
        bpp               (nc_len, nc_len)   — symmetric, for downstream use
        p_ss              (nc_len,)          — P(unpaired) per position
        ensemble_energy   scalar             — from fc.pf()
    """
    nc_length: int
    guide_length: int
    dG_open_u1: np.ndarray
    dG_open_uL_pn: np.ndarray
    E_span_win: np.ndarray
    H_pair_win: np.ndarray
    bpp: np.ndarray
    p_ss: np.ndarray
    ensemble_energy: float


def _fold_compound_pf(seq: str) -> tuple:
    """Return (folded ``RNA.fold_compound``, ensemble_energy_kcal_per_mol).
    Accepts DNA (T) or RNA (U). pf() has been run so bpp() is valid."""
    rna = seq.upper().replace("T", "U")
    for ch in rna:
        if ch not in "ACGUN":
            raise ValueError(f"Non-nucleotide character {ch!r} in sequence")
    fc = RNA.fold_compound(rna)
    pf_ret = fc.pf()
    # pf() returns (centroid_structure_str, ensemble_energy_float) on modern
    # ViennaRNA. Fall back to 0.0 if the return shape is unexpected.
    if isinstance(pf_ret, tuple) and len(pf_ret) >= 2:
        ee = float(pf_ret[1])
    else:
        ee = 0.0
    return fc, ee


def _bpp_matrix(fc, n: int) -> np.ndarray:
    """Symmetric BPP as np.float64 (n, n). ViennaRNA fc.bpp() returns
    an ``(n+1) x (n+1)`` list-of-lists, 1-indexed, upper-triangular."""
    raw = fc.bpp()
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n + 1):
        row = raw[i]
        for j in range(i + 1, n + 1):
            p = row[j]
            if p:
                mat[i - 1, j - 1] = p
                mat[j - 1, i - 1] = p
    return mat


def _p_ss(bpp: np.ndarray) -> np.ndarray:
    """P(unpaired[i]) = 1 - sum_j BPP[i,j], clipped to (0, 1]."""
    return np.clip(1.0 - bpp.sum(axis=1), 1e-10, 1.0)


def _dg_open_u1(p_ss: np.ndarray) -> np.ndarray:
    """Per-position dG_open at u=1 in kcal/mol."""
    return -KT_37C_KCAL_PER_MOL * np.log(p_ss)


def _dg_open_uL_per_nt(dg_u1: np.ndarray, L: int) -> np.ndarray:
    """Per-nt-normalized dG_open over each L-window.

    Under the independence approximation P(all unpaired in [i, i+L]) ~=
    prod P(unpaired[k]), so dG_open_uL = sum dG_open_u1 over the window.
    Divide by L for per-nt units so the channel is comparable across
    different L values (per docstring caveat).
    """
    if len(dg_u1) < L:
        return np.zeros(0, dtype=np.float64)
    csum = np.concatenate(([0.0], np.cumsum(dg_u1, dtype=np.float64)))
    win_sum = csum[L:] - csum[:-L]
    return win_sum / L


def _per_position_span(bpp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (numer, denom) per position where
        numer[i] = sum_j BPP[i,j] * |i-j|
        denom[i] = sum_j BPP[i,j]
    Both length n. Weighted-mean E_span for a window is
    sum(numer[win]) / sum(denom[win]); NaN if denom_sum < eps.
    """
    n = bpp.shape[0]
    idx = np.arange(n, dtype=np.float64)
    dist_mat = np.abs(idx[:, None] - idx[None, :])
    numer = (bpp * dist_mat).sum(axis=1)
    denom = bpp.sum(axis=1)
    return numer, denom


def _per_position_pair_entropy(bpp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """H_i = -sum_j q(j|i) ln q(j|i) with q normalized over j != i.
    Returns (entropy_per_position, pairing_mass_per_position).
    Positions with pairing_mass < eps have entropy = 0 and weight = 0
    (a fully-unpaired position has no defined partner distribution).
    """
    n = bpp.shape[0]
    denom = bpp.sum(axis=1)
    entropy = np.zeros(n, dtype=np.float64)
    valid = denom > EPS_PAIR_MASS
    for i in np.nonzero(valid)[0]:
        q = bpp[i] / denom[i]
        pos_q = q[q > 0]
        entropy[i] = -(pos_q * np.log(pos_q)).sum()
    return entropy, denom


def _window_weighted_mean(vals: np.ndarray, weights: np.ndarray, L: int,
                            eps: float = EPS_PAIR_MASS) -> np.ndarray:
    """For each L-window starting at i, compute
        sum(vals[win] * weights[win]) / sum(weights[win])
    Returns NaN if the window's total weight is < eps."""
    if len(vals) < L:
        return np.zeros(0, dtype=np.float64)
    wv = vals * weights
    csum_w = np.concatenate(([0.0], np.cumsum(weights, dtype=np.float64)))
    csum_wv = np.concatenate(([0.0], np.cumsum(wv, dtype=np.float64)))
    win_w = csum_w[L:] - csum_w[:-L]
    win_wv = csum_wv[L:] - csum_wv[:-L]
    out = np.full(len(win_w), np.nan, dtype=np.float64)
    mask = win_w > eps
    out[mask] = win_wv[mask] / win_w[mask]
    return out


def compute_features_v2(seq: str, guide_length: int = DEFAULT_GUIDE_LENGTH
                         ) -> StructureFeaturesV2:
    """Global RNAfold + BPP -> 4-channel structural feature bundle.

    Args:
        seq: ncRNA sequence (DNA or RNA alphabet, ACGT/ACGU + N).
        guide_length: L used for window-level channels (ch1, ch2, ch3).

    Returns:
        StructureFeaturesV2 with per-position and per-window channels.

    Timing (per pf() bench 2026-08-31):
        L=177  ~91 ms   L=225 ~156 ms   L=281 ~285 ms  (pf + bpp)
    """
    n = len(seq)
    if n < guide_length:
        raise ValueError(f"nc_length {n} < guide_length {guide_length}")

    fc, ensemble_energy = _fold_compound_pf(seq)
    bpp = _bpp_matrix(fc, n)
    p_ss = _p_ss(bpp)
    dg_u1 = _dg_open_u1(p_ss)
    dg_uL_pn = _dg_open_uL_per_nt(dg_u1, guide_length)

    # E_span: weighted mean partner-distance over the L-window.
    span_numer_per_pos, pair_mass_per_pos = _per_position_span(bpp)
    E_span_win = _window_weighted_mean(
        span_numer_per_pos / np.maximum(pair_mass_per_pos, EPS_PAIR_MASS),
        pair_mass_per_pos, guide_length)

    # H_pair: weighted mean of per-position partner entropy over the window.
    H_per_pos, pair_mass_h = _per_position_pair_entropy(bpp)
    H_pair_win = _window_weighted_mean(H_per_pos, pair_mass_h, guide_length)

    return StructureFeaturesV2(
        nc_length=n,
        guide_length=guide_length,
        dG_open_u1=dg_u1,
        dG_open_uL_pn=dg_uL_pn,
        E_span_win=E_span_win,
        H_pair_win=H_pair_win,
        bpp=bpp,
        p_ss=p_ss,
        ensemble_energy=ensemble_energy,
    )
