"""difficulty.py — per-bag difficulty draw for the V5 generator.

Three axes are sampled independently:

  L (guide length) uniform on {11, 12, 13, 14}
    Reason: the three natural anchors have L in {11, 13, 14} (T-WT,
    ISEc11, ISEc21), no single L dominates. Family-specific priors
    (e.g., IS110-family L=11) would defeat the generator's purpose,
    same principle as the uniform guide_position axis (user directive
    2026-08-31 — "orientation is family-specific" reasoning applies).

  nc_len uniform on [70, 300] (with fallback max(nc_len, L + 1))
    70 nt covers the mature small seekRNA lower bound, 300 nt exceeds
    the largest natural ncRNA in the anchor set (281 nt ISEc21).
    guide-length/nc-length ratio range: 3.7% (L=11, nc=300) to 20%
    (L=14, nc=70), covering the real anchor range 6.2% (T-WT 11/177)
    to 22% (seekRNA2 13/58 for ISEc21).

  planted_m from a T-WT-like tail at m in {target_m, target_m-1, target_m-2}
    with mass 86% / 10% / 4%. Zero mass at target_m+1 and above (matches
    T-WT: P(planted_m >= 10) == 0 for L=11). target_m per L is reverse-
    solved by EMPIRICAL LOOKUP on the real 2,763-flank pool so that
    E[competitor_count / n_positions] approx 0.21 (T-WT operating point).

Two independent knobs → the acceptance-test invariant `competitor_count`
should hold across L via the reverse-solve, up to composition drift the
analytic formula doesn't capture.

Analytic sanity check (documented in `_analytic_rate_at`, not used for
sampling): rate ~= n_starts × 2 × P(Bin(L, p_hat) >= m), where the
factor 2 is fwd + rc (assumed independent — cheap upper bound; strict
form is 1 - (1-p)^2). Empirical/analytic mismatch tolerance ~15%
(A+ measured 0.204-0.238 vs analytic 0.262).
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np


DEFAULT_L_CHOICES: tuple[int, ...] = (11, 12, 13, 14)
DEFAULT_NC_LEN_LO = 70
DEFAULT_NC_LEN_HI = 300
DEFAULT_TARGET_RATE = 0.21
DEFAULT_PLANTED_M_TAIL: tuple[float, float, float] = (0.86, 0.10, 0.04)
DEFAULT_FLANK_LEN = 120

REAL_FLANK_POOL_FAMILIES = ("IS10-R", "IS30", "IS903", "ISAjo2", "ISLdl1")
REAL_FLANK_POOL_BASEDIR = "/global/scratch/users/kh36969/DL_novel_guide_editor/real_data/formatted"


@dataclass(frozen=True)
class Difficulty:
    """One realized (L, nc_len, planted_m) tuple."""
    L: int
    nc_len: int
    planted_m: int
    target_m: int
    target_rate: float


@dataclass
class RateTable:
    """Empirical (L, m) -> rate = median(competitor_count / n_positions).

    Built by sampling n_probe (nc, flank) pairs where nc is a random ACGT
    sequence at nc_len ~ U[70, 300] and flank is drawn from the real 2763-
    flank pool. For each pair we count positions p in [0, nc_len - L + 1)
    where m_max(p; flank) >= m in either orientation, then rate = count /
    n_positions. Median across pairs is the table entry.
    """
    L_range: tuple[int, ...]
    m_range: tuple[int, ...]
    rate: dict[tuple[int, int], float]           # (L, m) -> median rate
    n_probe: int
    flank_pool_size: int
    p_hat: float                                  # observed non-N frequency in flank pool

    def target_m_for_L(self, L: int, target_rate: float
                        ) -> int:
        """m that minimizes |rate(L, m) - target_rate|. Ties broken to
        the smaller m (harder tail).
        Rationale: T-WT at (L=11, m=8) has empirical rate ~0.22 on random
        nc — slightly above 0.21. A strict "smallest m such that rate <=
        target" rule would overshoot to m=9 (rate ~0.02), collapsing the
        competitor_count distribution. Closest-match preserves the T-WT
        operating point at the natural anchor.
        """
        candidates = [m for m in sorted(self.m_range) if (L, m) in self.rate]
        if not candidates:
            raise KeyError(f"L={L} not covered by rate table")
        return min(candidates, key=lambda m: abs(self.rate[(L, m)] - target_rate))

    def to_json(self) -> dict:
        return {
            "L_range":  list(self.L_range),
            "m_range":  list(self.m_range),
            "rate":     {f"{L},{m}": v for (L, m), v in self.rate.items()},
            "n_probe":  self.n_probe,
            "flank_pool_size": self.flank_pool_size,
            "p_hat":    self.p_hat,
        }

    @classmethod
    def from_json(cls, d: dict) -> "RateTable":
        rate = {}
        for k, v in d["rate"].items():
            L, m = k.split(",")
            rate[(int(L), int(m))] = float(v)
        return cls(
            L_range=tuple(d["L_range"]), m_range=tuple(d["m_range"]),
            rate=rate, n_probe=int(d["n_probe"]),
            flank_pool_size=int(d["flank_pool_size"]),
            p_hat=float(d["p_hat"]),
        )


# ---------------- flank pool + rate table ----------------

def _load_real_flank_pool() -> list[str]:
    """Load the 2,763 real bacterial downstream 120-nt flanks."""
    pool = []
    for fam in REAL_FLANK_POOL_FAMILIES:
        p = f"{REAL_FLANK_POOL_BASEDIR}/real_{fam}_sites.jsonl"
        try:
            with open(p) as f:
                for line in f:
                    d = json.loads(line)
                    m = d.get("generator_metadata", {})
                    if m.get("flank_side") != "downstream":
                        continue
                    fl = d.get("inputs", {}).get("flank")
                    if fl and len(fl) == DEFAULT_FLANK_LEN:
                        pool.append(fl.upper())
        except FileNotFoundError:
            continue
    return pool


def _observed_p_hat(flank_pool: list[str]) -> float:
    """Non-N base frequency in the pool: mass at any of A/C/G/T times 0.25
    (the per-position match probability for a uniform-target random nc)."""
    counts = {b: 0 for b in "ACGT"}
    total = 0
    for fl in flank_pool:
        for b in fl:
            if b in counts:
                counts[b] += 1
            total += 1
    p_hat = sum(counts[b] / total * (1.0 if b in "ACGT" else 0.0)
                 for b in counts) * 0.25
    # Equivalent (simpler): frequency of ACGT bases * 0.25
    ratio_acgt = sum(counts.values()) / max(total, 1)
    return ratio_acgt * 0.25


def _windowed_matches(fwd_dot: np.ndarray, L: int) -> np.ndarray:
    """Convolve running match count on fwd/rc dot plot to windowed sums.
    Simple wrapper for consistency with preprocess.alignment.windowed_matches.
    """
    from preprocess.alignment import windowed_matches
    return windowed_matches(fwd_dot, L)


def _pos_m_max(nc: str, flank: str, L: int) -> np.ndarray:
    """Per-nc-position max over flank offsets of window match count L,
    pooled over both orientations. Same convention as MatchTable's m_max
    at excl_w=0."""
    from preprocess.alignment import dot_plot, windowed_matches
    fwd, rc = dot_plot(nc, flank)
    w_f = windowed_matches(fwd, L)
    w_r = windowed_matches(rc, L)
    if w_f.size == 0 or w_r.size == 0:
        return np.zeros(0, dtype=np.int32)
    n = min(w_f.shape[0], w_r.shape[0])
    return np.maximum(w_f.max(axis=1)[:n], w_r.max(axis=1)[:n])


def build_rate_table(
    L_range: tuple[int, ...] = DEFAULT_L_CHOICES,
    m_range: tuple[int, ...] = (5, 6, 7, 8, 9, 10, 11, 12),
    n_probe: int = 500,
    nc_len_range: tuple[int, int] = (DEFAULT_NC_LEN_LO, DEFAULT_NC_LEN_HI),
    seed: int = 0,
    flank_pool: list[str] | None = None,
) -> RateTable:
    """Empirically measure rate(L, m) = median over n_probe pairs of
    (competitor_count / n_positions) at m_threshold m.

    Sampling: nc uniform ACGT at length ~ U[nc_len_range]; flank drawn
    from the real 2,763-flank pool (both must match generator's own
    sampling to make the table transferable)."""
    if flank_pool is None:
        flank_pool = _load_real_flank_pool()
    p_hat = _observed_p_hat(flank_pool)

    rng = random.Random(seed)
    rates: dict[tuple[int, int], list[float]] = {(L, m): []
                                                    for L in L_range
                                                    for m in m_range}
    n_pool = len(flank_pool)
    for _ in range(n_probe):
        nc_len = rng.randint(*nc_len_range)
        nc = "".join(rng.choices("ACGT", k=nc_len))
        fl = flank_pool[rng.randrange(n_pool)]
        # Precompute per-L per-position m_max once per (nc, flank) pair
        for L in L_range:
            m_arr = _pos_m_max(nc, fl, L)
            if m_arr.size == 0:
                continue
            n_pos = len(m_arr)
            for m in m_range:
                rate = float((m_arr >= m).sum()) / n_pos
                rates[(L, m)].append(rate)
    median_rates: dict[tuple[int, int], float] = {}
    for k, vs in rates.items():
        if vs:
            median_rates[k] = float(np.median(vs))
    return RateTable(
        L_range=tuple(L_range),
        m_range=tuple(m_range),
        rate=median_rates,
        n_probe=n_probe,
        flank_pool_size=n_pool,
        p_hat=p_hat,
    )


# ---------------- analytic sanity ----------------

def _analytic_rate_at(L: int, m: int, n_starts: int, p_hat: float) -> float:
    """Sanity: rate ~= n_starts * 2 * P(Bin(L, p_hat) >= m).
    Cheap upper bound assuming fwd+rc independence. Strict form
    1 - (1 - p)^2 differs by O(p^2), negligible for p << 1.
    """
    p = 0.0
    for k in range(m, L + 1):
        p += math.comb(L, k) * (p_hat ** k) * ((1 - p_hat) ** (L - k))
    return n_starts * 2.0 * p


def analytic_target_m_for_L(L: int, target_rate: float,
                               n_starts: int = DEFAULT_FLANK_LEN - 11 + 1,
                               p_hat: float = 0.25) -> int:
    """Smallest m such that analytic rate <= target_rate. Sanity only."""
    for m in range(1, L + 1):
        if _analytic_rate_at(L, m, n_starts, p_hat) <= target_rate:
            return m
    return L


# ---------------- sampler ----------------

def sample_L(rng: random.Random,
             L_choices: tuple[int, ...] = DEFAULT_L_CHOICES) -> int:
    return rng.choice(L_choices)


def sample_nc_len(rng: random.Random, L: int,
                    lo: int = DEFAULT_NC_LEN_LO,
                    hi: int = DEFAULT_NC_LEN_HI) -> int:
    """Uniform on [lo, hi] with a floor of L+1 (nc must fit the guide)."""
    lo_eff = max(lo, L + 1)
    return rng.randint(lo_eff, hi)


def sample_planted_m(rng: random.Random, target_m: int,
                     tail: tuple[float, float, float] = DEFAULT_PLANTED_M_TAIL
                     ) -> int:
    """Draw planted_m from {target_m, target_m-1, target_m-2} at the
    given probabilities. Zero mass above target_m."""
    u = rng.random()
    p0 = tail[0]
    p1 = tail[0] + tail[1]
    if u < p0:
        return target_m
    if u < p1:
        return target_m - 1
    return target_m - 2


def sample_difficulty(rng: random.Random, rate_table: RateTable,
                        target_rate: float = DEFAULT_TARGET_RATE) -> Difficulty:
    """Draw (L, nc_len, planted_m) for one bag. Uses the rate table
    (empirical, from real flank pool) to reverse-solve target_m per L."""
    L = sample_L(rng)
    nc_len = sample_nc_len(rng, L)
    target_m = rate_table.target_m_for_L(L, target_rate)
    planted_m = sample_planted_m(rng, target_m)
    return Difficulty(
        L=L, nc_len=nc_len, planted_m=planted_m,
        target_m=target_m, target_rate=target_rate,
    )


# ---------------- cache helpers ----------------

_DEFAULT_RATE_TABLE_PATH = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/rate_table.json"


def load_or_build_rate_table(path: str = _DEFAULT_RATE_TABLE_PATH,
                                rebuild: bool = False) -> RateTable:
    """Cache the rate table on disk so downstream imports don't rebuild it."""
    p = Path(path)
    if not rebuild and p.exists():
        with open(p) as f:
            return RateTable.from_json(json.load(f))
    tbl = build_rate_table()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(tbl.to_json(), f, indent=2)
    return tbl
