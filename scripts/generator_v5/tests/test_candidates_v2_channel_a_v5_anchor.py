"""A8 — v2 position arrays → Channel A end-to-end on V5 50K.

Companion to A7 (Durrant). Same measurement, different corpus. A7 covers
the cross-implementation equivalence on real Tnps (T-WT family, L=11
only, no mm_geometry, no is_split). A8 covers the V5 conditions the v2
layer will actually see in production:

  - all L in {11, 12, 13, 14}
  - mm_geometry axis (clustered/dispersed × 5p/3p)
  - is_split=True corpus fraction (~18%)
  - forced-mm-position artifacts at guide {0, 1} or {L-2, L-1}

If v2 arrays diverge from the current MatchTable ONLY on V5 (not on
Durrant), the divergence is caused by V5-specific handling — likely
`arch.orient='rev'` vs pool 'rc' or a windowed_matches boundary case
that A1/A7 do not exercise.

Anchor (from FROZEN.md V5 corpus anchors, post-mm-geometry regen):
    L=11 Mode 2 (m=8, tau=0, S=5):
      coverage  = 0.5228
      ppv_peak  = 0.9410
      ppv_tnp   = 0.9680
      exact_rate = 0.3975

SCOPE — what A8 anchors and what it does NOT:
  A8 replaces `m_max_by_excl[0]` in every `MatchArrays` with v2's
  `enumerate_position_arrays` output. Every other exclusion width
  (`excl_w ∈ {2, 8, 9, 12}` — used by `tsd_handling="partition"`
  variants) is populated by the old path (`_windowed_max_by_excl`
  applied to `windowed_matches`). Mode 2 (fixed L=11, m>=8, tau=0,
  S=5) reads only excl_w=0, so A8 covers it.
  A future variant that queries excl_w > 0 is NOT anchored by A8.
  Adding such a variant means adding a companion anchor (A8b, say)
  that verifies v2 populates the higher-excl slots identically.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import json
from collections import defaultdict

from scripts.v5a_framework.match_table import (
    load as load_mt, MatchArrays, EXCL_WIDTHS, _windowed_max_by_excl,
)
from scripts.v5a_framework.variant import spec_m_threshold_L11, run_variant
from scripts.generator_v5.channel_a_v5 import compute_channel_a
from preprocess.candidates_v2 import enumerate_position_arrays
from preprocess.alignment import dot_plot, windowed_matches


V5_JSONL = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/large_batch/positives_v5_50k.jsonl"
V5_SHARD = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/channel_a/mt_50k"

# From FROZEN.md V5 corpus anchors, post-mm-geometry regen (2026-09-01).
V5_L11_ANCHOR = {
    "coverage":   0.5228,
    "ppv_peak":   0.9410,
    "ppv_tnp":    0.9680,
    "exact_rate": 0.3975,
}
TOL = 5e-4


def _populate_one_tnp(args):
    tnp_id, tnp_pickle, orients, Ls, EXCL_WIDTHS_ = args
    import pickle
    tnp = pickle.loads(tnp_pickle)
    entry: dict = {}
    for site in tnp.sites:
        v2_arrays = enumerate_position_arrays(
            tnp.nc, site.flank, L_min=min(Ls), L_max=max(Ls),
            orientations=orients,
        )
        fwd_dot, rc_dot = dot_plot(tnp.nc, site.flank)
        for orient in orients:
            dot = fwd_dot if orient == "fwd" else rc_dot
            for L in Ls:
                win = windowed_matches(dot, L)
                all_w = _windowed_max_by_excl(win, EXCL_WIDTHS_)
                v2_arr = v2_arrays.get((orient, L))
                if v2_arr is not None and v2_arr.size > 0:
                    all_w[0] = v2_arr.astype(np.int8)
                entry[(site.site_idx, orient, L)] = MatchArrays(m_max_by_excl=all_w)
    return tnp_id, entry


def _populate_cache_from_v2(mt, workers: int = 1) -> None:
    """Replace mt._cache with per-(site, orient, L) MatchArrays whose
    m_max_by_excl[0] comes from `enumerate_position_arrays`.
    Parallelized over Tnps when workers > 1.
    """
    import multiprocessing as mp
    import pickle
    import time

    if workers <= 1:
        for tnp_id in mt.tnp_ids:
            tnp = mt.tnps[tnp_id]
            _, entry = _populate_one_tnp(
                (tnp_id, pickle.dumps(tnp), mt.orients, mt.Ls, EXCL_WIDTHS))
            mt._cache[tnp_id] = entry
        return

    tasks = [(tnp_id, pickle.dumps(mt.tnps[tnp_id]), mt.orients, mt.Ls, EXCL_WIDTHS)
             for tnp_id in mt.tnp_ids]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for i, (tnp_id, entry) in enumerate(
                pool.imap_unordered(_populate_one_tnp, tasks, chunksize=32), 1):
            mt._cache[tnp_id] = entry
            if i % 5000 == 0:
                dt = time.time() - t0
                print(f"  [v2-populate] {i}/{len(tasks)}  {dt:.1f}s  ({dt/i*1000:.1f} ms/Tnp)", flush=True)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=60)
    args = ap.parse_args()

    print(f"[A8] loading V5 50K MatchTable (cached shards) from {V5_SHARD}")
    mt = load_mt(V5_SHARD)
    print(f"[A8] {len(mt.tnp_ids)} Tnps loaded")

    print(f"[A8] extracting arch metadata from JSONL")
    tnp_arch: dict[str, dict] = {}
    with open(V5_JSONL) as f:
        for line in f:
            r = json.loads(line)
            tnp = r["transposase_id"]
            if tnp in tnp_arch:
                continue
            arch = dict(r["labels"].get("arch", {}))
            nc_len = int(r["labels"].get("ncrna_length", 0))
            if nc_len < 120:   nc_bucket = "070-119"
            elif nc_len < 180: nc_bucket = "120-179"
            elif nc_len < 240: nc_bucket = "180-239"
            else:              nc_bucket = "240-300"
            arch["nc_len_bucket"] = nc_bucket
            tnp_arch[tnp] = arch
    print(f"[A8] arch metadata: {len(tnp_arch)} Tnps")

    print(f"[A8] populating _cache from candidates_v2.enumerate_position_arrays (workers={args.workers})")
    _populate_cache_from_v2(mt, workers=args.workers)

    print(f"[A8] running Mode 2 spec (fixed L=11, m>=8, tau=0, S=5)")
    spec = spec_m_threshold_L11(m=8, tau=0, S=5)
    peaks = run_variant(mt, spec)

    # L=11 restrict via arch metadata
    result = compute_channel_a(mt, peaks, tnp_arch, stratify_by=None,
                                 restrict_to={"L": 11})
    got = result["all"]

    print()
    print(f"  L=11 Mode 2 field           anchor      got   |Δ|      match")
    all_ok = True
    for field, want in V5_L11_ANCHOR.items():
        v = float(got.get(field, float("nan")))
        delta = abs(v - want)
        ok = delta < TOL
        print(f"  {field:<22s} {want:>8.4f} {v:>8.4f} {delta:>6.4f}   {'✓' if ok else '✗'}")
        if not ok:
            all_ok = False

    print()
    print("[A8] " + (
        f"PASS: Channel A on v2 arrays reproduces V5 L=11 anchor (all Δ < {TOL})."
        if all_ok else
        "FAIL: Channel A on v2 arrays diverges from V5 L=11 anchor. "
        "Investigate before any v2-driven V5 measurement is used."
    ))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
