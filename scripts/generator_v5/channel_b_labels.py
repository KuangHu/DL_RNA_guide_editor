"""Per-position ordinal label derivation for Channel B training.

`n_planted_at_position[p]` = |{ site : is_planted(site) ∧
                                 guide_span_start(site) == p }|

Derived from the existing V5 schema fields — no regen required. The
labels for all four generator modes fall out of the same computation:

  positive / none:  count = n_sites at bag_nc_start, else 0
  scattered:        count = 1 at each site's own placement, else 0
                    (up to n_sites distinct positions)
  partial:          count = n_planted at bag_nc_start, else 0

The label is intentionally NOT thresholded to a boolean — that's the
inference-time decision. See `docs/channel_b_spec.md` §3.
"""
from __future__ import annotations

from typing import Iterable


def n_planted_at_position(bag_records: Iterable[dict], nc_len: int
                           ) -> list[int]:
    """Return a length-`nc_len` list where entry `p` counts how many of
    the bag's sites planted their guide at nc_start = p.

    `bag_records` is the list of per-site JSONL records that share a
    single transposase_id. Requires each record to have:
      `labels.is_planted` (bool)
      `labels.guide_span_in_active_noncoding` ([nc_start, nc_end))

    Behavior on old batches without `is_planted` (pre-2026-09-01
    positives): every site treated as planted. That reproduces
    n_planted=5 at bag_nc_start for the frozen positive corpus.
    """
    out = [0] * nc_len
    for r in bag_records:
        lab = r["labels"]
        if not lab.get("is_planted", True):
            continue
        span = lab.get("guide_span_in_active_noncoding")
        if not span:
            continue
        p = int(span[0])
        if 0 <= p < nc_len:
            out[p] += 1
    return out


def channel_b_positive_positions(bag_records: Iterable[dict], nc_len: int
                                   ) -> list[tuple[int, int]]:
    """Return `[(position, count), ...]` for positions with count > 0.
    Compact form of `n_planted_at_position` for storage or logging."""
    arr = n_planted_at_position(bag_records, nc_len)
    return [(p, c) for p, c in enumerate(arr) if c > 0]
