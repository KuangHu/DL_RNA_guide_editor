"""Phase A of model-free scorer v3 — multi-hit retention per (site, nc_pos, L).

Replaces the m_max + argmax reduction with a true multi-hit list that lets
downstream Phase C enumerate across-site position clusters even when the
cross-site consistent hit is NOT the single-site best match.

Key correctness points (vs v1/v2):
  - All hits stored in FWD-flank coordinate (RC hits converted via
    rc_flank_pos_to_flank_pos at retention time, never mixed later).
  - Loose retention floor (`m_floor_retain_default`) — recall knob only;
    final credibility threshold is set in Phase G against a null, not here.
  - Windows that straddle the 13-N spacer in nc_concat are dropped
    (no hit kept) — a guide call cannot span a spacer.
  - K_valid is frozen here against site input adequacy, NOT against
    whether any hits survive filtering.

Public API:
  retain_hits(nc_concat, flank, Ls, m_floor_fn=default) -> list[Hit]
  retain_bag(sites, Ls, ...)                             -> BagRetention
  compute_K_valid(sites, min_nc_len, min_flank_len)      -> list[int]
  m_floor_retain_default(L)                              -> int (loose: ceil(0.6*L))
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import math
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocess.alignment import (
    dot_plot, windowed_matches, revcomp, rc_flank_pos_to_flank_pos,
)

SPACER_N = 13  # nc_concat joins noncoding regions with 13 N's


@dataclass(frozen=True)
class Hit:
    """A single retained match of nc[nc_pos:nc_pos+L] against the flank.

    `flank_pos` is always in the FWD flank coordinate (0-indexed into the
    original 5'->3' flank), regardless of which orientation detected it.
    `orient='rc'` means nc[nc_pos:nc_pos+L] matched revcomp(flank[flank_pos:flank_pos+L]),
    i.e. this hit corresponds to the guide pairing with the opposite strand.
    """
    nc_pos: int
    L: int
    flank_pos: int   # fwd-flank coord
    m: int           # matching bases out of L
    orient: str      # 'fwd' or 'rc'


@dataclass
class SiteRetention:
    site_idx: int
    hits_by_nc_pos_L: dict[tuple[int, int], list[Hit]]  # (nc_pos, L) -> hits
    reason_dropped: str | None = None    # None if site contributed


@dataclass
class BagRetention:
    K_valid: int                              # frozen denominator
    valid_site_idx: list[int]                 # the K_valid indices (frozen)
    retained: list[SiteRetention]             # one entry per valid site
    dropped: dict[int, str]                   # site_idx -> drop reason


# ---------- defaults ----------
def m_floor_retain_default(L: int) -> int:
    """Loose retention floor: ceil(0.6 * L).  High-recall knob only.
    Final credibility threshold is calibrated post-hoc in Phase G."""
    return int(math.ceil(0.6 * L))


# ---------- helpers ----------
def _spacer_mask(nc: str, L: int) -> np.ndarray:
    """Return a boolean mask over nc positions [0 .. len(nc)-L+1).  True means
    the L-window starting at that position straddles the 13-N spacer and must
    be dropped.

    A straddle is any window that contains an N base; the spacer is the only
    N stretch in nc_concat so this catches it.
    """
    n_pos = len(nc) - L + 1
    if n_pos <= 0:
        return np.zeros(0, dtype=bool)
    is_n = np.array([b == "N" for b in nc], dtype=bool)
    # window i spans [i, i+L).  "contains N" = any True in that slice.
    # cumulative-count trick:
    cum = np.concatenate(([0], np.cumsum(is_n)))
    n_count = cum[L:L + n_pos] - cum[:n_pos]
    return n_count > 0


# ---------- core ----------
def retain_hits(nc: str, flank: str, Ls: tuple[int, ...],
                m_floor_fn=m_floor_retain_default,
                ) -> list[Hit]:
    """Return every (nc_pos, L, flank_pos, m, orient) with m >= m_floor_fn(L).

    FWD-flank coord throughout. Spacer-straddling windows are omitted.
    Hits are NOT ranked here — ranking is a downstream concern.
    """
    if len(nc) == 0 or len(flank) == 0:
        return []
    fwd_dot, rc_dot = dot_plot(nc, flank)
    flen = len(flank)
    hits: list[Hit] = []
    for L in Ls:
        if len(nc) < L or len(flank) < L:
            continue
        floor = m_floor_fn(L)
        spacer_mask = _spacer_mask(nc, L)  # True => drop

        fwd_win = windowed_matches(fwd_dot, L)   # (n_nc, n_fl)
        rc_win  = windowed_matches(rc_dot,  L)
        if fwd_win.size:
            ii, jj = np.where(fwd_win >= floor)
            keep = ~spacer_mask[ii]
            ii, jj = ii[keep], jj[keep]
            for i, j in zip(ii.tolist(), jj.tolist()):
                hits.append(Hit(nc_pos=int(i), L=L,
                                flank_pos=int(j), m=int(fwd_win[i, j]),
                                orient="fwd"))
        if rc_win.size:
            ii, jj = np.where(rc_win >= floor)
            keep = ~spacer_mask[ii]
            ii, jj = ii[keep], jj[keep]
            for i, j in zip(ii.tolist(), jj.tolist()):
                flank_pos_fwd = rc_flank_pos_to_flank_pos(int(j), L, flen)
                hits.append(Hit(nc_pos=int(i), L=L,
                                flank_pos=flank_pos_fwd,
                                m=int(rc_win[i, j]),
                                orient="rc"))
    return hits


def _group_by_nc_pos_L(hits: list[Hit]) -> dict[tuple[int, int], list[Hit]]:
    out: dict[tuple[int, int], list[Hit]] = {}
    for h in hits:
        out.setdefault((h.nc_pos, h.L), []).append(h)
    for k in out:
        out[k].sort(key=lambda h: (-h.m, h.flank_pos))  # best-first for rank
    return out


# ---------- bag-level ----------
def compute_K_valid(sites: list[dict], min_nc_len: int, min_flank_len: int,
                     ) -> tuple[int, list[int], dict[int, str]]:
    """Freeze the denominator against site input adequacy BEFORE any filtering.

    A site is 'valid' iff it has:
      - a nonempty nc_concat of length >= min_nc_len
      - a flank of length >= min_flank_len
      - no exceptions building nc_concat
    Returns (K_valid, list_of_valid_idx, dict_of_dropped_idx_to_reason).
    """
    valid: list[int] = []
    dropped: dict[int, str] = {}
    for i, s in enumerate(sites):
        try:
            nc_list = s["inputs"]["noncoding_regions"]
            if not nc_list:
                dropped[i] = "no_noncoding_regions"; continue
            nc = nc_list[0]
            for extra in nc_list[1:]:
                nc = nc + "N" * SPACER_N + extra
            flank = s["inputs"]["flank"]
            if len(nc) < min_nc_len:
                dropped[i] = f"nc_too_short({len(nc)}<{min_nc_len})"; continue
            if len(flank) < min_flank_len:
                dropped[i] = f"flank_too_short({len(flank)}<{min_flank_len})"; continue
        except (KeyError, TypeError) as e:
            dropped[i] = f"input_missing:{type(e).__name__}"; continue
        valid.append(i)
    return len(valid), valid, dropped


def retain_bag(sites: list[dict], Ls: tuple[int, ...],
                m_floor_fn=m_floor_retain_default,
                min_nc_len: int | None = None,
                min_flank_len: int | None = None,
                ) -> BagRetention:
    """Freeze K_valid, then run retain_hits per valid site."""
    if min_nc_len is None:
        min_nc_len = max(Ls)
    if min_flank_len is None:
        min_flank_len = max(Ls)
    K_valid, valid_idx, dropped = compute_K_valid(
        sites, min_nc_len=min_nc_len, min_flank_len=min_flank_len,
    )
    retained: list[SiteRetention] = []
    for i in valid_idx:
        s = sites[i]
        nc_list = s["inputs"]["noncoding_regions"]
        nc = nc_list[0]
        for extra in nc_list[1:]:
            nc = nc + "N" * SPACER_N + extra
        flank = s["inputs"]["flank"]
        hits = retain_hits(nc, flank, Ls, m_floor_fn=m_floor_fn)
        retained.append(SiteRetention(
            site_idx=i,
            hits_by_nc_pos_L=_group_by_nc_pos_L(hits),
        ))
    return BagRetention(K_valid=K_valid, valid_site_idx=valid_idx,
                         retained=retained, dropped=dropped)
