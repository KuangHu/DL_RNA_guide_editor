"""Phase B of model-free scorer v3 — bilateral per-site motif gates.

Applied to every retained hit (Phase A).  A hit is kept only if BOTH
sides pass:
  - guide  = nc_per_site[site][nc_pos : nc_pos + L]
  - target = flank_per_site[site][flank_pos : flank_pos + L]
    (fwd-frame substring; the gate is strand-invariant for the structures
    it catches — poly-A / poly-T both fail homopolymer, ATAT / TATA both
    fail period-2, etc.)

Gate configuration is explicit, with each threshold a separate knob:

  - homopolymer_frac    : reject if longest_same_base_run / L >= this
  - period_frac         : reject if periodicity_fraction(p) >= this
                           for any p in periods_to_check (default {1..4})
  - at_frac_max         : reject if (A+T) / L > this   -- SEPARATE knob from
                           complexity; AT-rich non-repetitive sequences
                           need their own gate
  - h_base_min          : reject if base Shannon entropy < this (bits)

periodicity_fraction(seq, p) =
    (# positions i in [p, L) with seq[i] == seq[i - p]) / (L - p)
This catches period-4 (ACGTACGT…) which distinct-dinuc / distinct-trinuc
counts miss.

Also keeps a RejectionLog of per-reason counts for diagnostic reporting.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import math
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.v5a_framework.multi_hit_retention import (
    BagRetention, SiteRetention, Hit, SPACER_N,
)


@dataclass(frozen=True)
class GateConfig:
    homopolymer_frac: float = 0.5
    period_frac: float = 0.80
    periods_to_check: tuple[int, ...] = (1, 2, 3, 4)
    at_frac_max: float = 0.75
    h_base_min: float = 1.3


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    longest_run: int = 0
    period_fracs: dict[int, float] = field(default_factory=dict)
    at_frac: float = 0.0
    h_base: float = 0.0


def longest_run(seq: str) -> int:
    if not seq: return 0
    best = cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
            if cur > best: best = cur
        else:
            cur = 1
    return best


def periodicity_fraction(seq: str, p: int) -> float:
    """Fraction of positions i in [p, len(seq)) where seq[i] == seq[i-p]."""
    if p < 1 or len(seq) <= p: return 0.0
    denom = len(seq) - p
    if denom <= 0: return 0.0
    return sum(1 for i in range(p, len(seq)) if seq[i] == seq[i - p]) / denom


def at_fraction(seq: str) -> float:
    valid = [b for b in seq if b in "ACGT"]
    if not valid: return 0.0
    return sum(1 for b in valid if b in "AT") / len(valid)


def shannon_base(seq: str) -> float:
    valid = [b for b in seq if b in "ACGT"]
    if not valid: return 0.0
    n = len(valid)
    H = 0.0
    for b in "ACGT":
        p = valid.count(b) / n
        if p > 0: H -= p * math.log2(p)
    return H


def evaluate_motif(seq: str, cfg: GateConfig) -> GateResult:
    """Return a GateResult: passed=False means at least one gate flagged seq."""
    if not seq:
        return GateResult(passed=False, reasons=["empty_seq"])
    L = len(seq)
    reasons: list[str] = []

    r = longest_run(seq)
    if r / L >= cfg.homopolymer_frac:
        reasons.append(f"homopolymer_run({r}/{L}={r/L:.2f}>={cfg.homopolymer_frac})")

    period_fracs = {p: periodicity_fraction(seq, p) for p in cfg.periods_to_check}
    for p, f in period_fracs.items():
        if f >= cfg.period_frac:
            reasons.append(f"period{p}({f:.2f}>={cfg.period_frac})")

    af = at_fraction(seq)
    if af > cfg.at_frac_max:
        reasons.append(f"at_frac({af:.2f}>{cfg.at_frac_max})")

    h = shannon_base(seq)
    if h < cfg.h_base_min:
        reasons.append(f"h_base({h:.2f}<{cfg.h_base_min})")

    return GateResult(passed=not reasons, reasons=reasons,
                       longest_run=r, period_fracs=period_fracs,
                       at_frac=af, h_base=h)


@dataclass
class RejectionLog:
    """Per-reason counts, keyed by (side, reason)."""
    counts: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    n_hits_in: int = 0
    n_hits_kept: int = 0
    n_hits_rejected: int = 0

    def summary(self) -> list[str]:
        rows = []
        rows.append(f"hits in={self.n_hits_in}  kept={self.n_hits_kept}  "
                    f"rejected={self.n_hits_rejected}")
        for (side, reason), n in sorted(self.counts.items(), key=lambda x: -x[1]):
            rows.append(f"  {side:<6s} {reason:<40s} {n}")
        return rows


def _reconstruct_nc_per_site(sites: list[dict], valid_idx: list[int]) -> dict[int, str]:
    """Rebuild nc_concat (with 13-N spacers) matching what Phase A used."""
    out: dict[int, str] = {}
    for i in valid_idx:
        nc_list = sites[i]["inputs"]["noncoding_regions"]
        nc = nc_list[0]
        for extra in nc_list[1:]:
            nc = nc + "N" * SPACER_N + extra
        out[i] = nc
    return out


def apply_phase_b(bag: BagRetention,
                   sites: list[dict],
                   cfg: GateConfig = GateConfig(),
                   ) -> tuple[BagRetention, RejectionLog]:
    """Return a new BagRetention with Phase-B-failing hits removed per site,
    plus a RejectionLog."""
    nc_per_site = _reconstruct_nc_per_site(sites, bag.valid_site_idx)
    flank_per_site = {i: sites[i]["inputs"]["flank"] for i in bag.valid_site_idx}

    log = RejectionLog()
    new_retained: list[SiteRetention] = []
    for sr in bag.retained:
        new_hits_by_k: dict[tuple[int, int], list[Hit]] = {}
        nc = nc_per_site[sr.site_idx]
        fl = flank_per_site[sr.site_idx]
        for (nc_pos, L), hits in sr.hits_by_nc_pos_L.items():
            kept: list[Hit] = []
            for h in hits:
                log.n_hits_in += 1
                guide = nc[nc_pos:nc_pos + L] if nc_pos + L <= len(nc) else ""
                tgt   = fl[h.flank_pos:h.flank_pos + L] if h.flank_pos + L <= len(fl) else ""
                g = evaluate_motif(guide, cfg)
                t = evaluate_motif(tgt, cfg)
                if g.passed and t.passed:
                    kept.append(h)
                    log.n_hits_kept += 1
                else:
                    log.n_hits_rejected += 1
                    for r in g.reasons:
                        log.counts[("guide", r.split("(")[0])] += 1
                    for r in t.reasons:
                        log.counts[("target", r.split("(")[0])] += 1
            if kept:
                # Re-sort kept (preserve the (-m, flank_pos) ordering from retention)
                kept.sort(key=lambda h: (-h.m, h.flank_pos))
                new_hits_by_k[(nc_pos, L)] = kept
        new_retained.append(SiteRetention(
            site_idx=sr.site_idx,
            hits_by_nc_pos_L=new_hits_by_k,
        ))
    return (BagRetention(K_valid=bag.K_valid,
                          valid_site_idx=bag.valid_site_idx,
                          retained=new_retained,
                          dropped=bag.dropped),
            log)
