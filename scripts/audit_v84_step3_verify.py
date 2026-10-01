"""V8.4 Step 3 verification — Rfam nc scaffold + contiguous-window
sanity + 7-mode axis parity.

Reports:
  1. Rfam pool composition: total sequences, per-family counts,
     length distribution.
  2. Per-bag Rfam-segment total length distribution
     (left_conserved_len + right_conserved_len) across 1K positive bags.
  3. Contiguous-window proof — for 10 randomly-picked bags print the
     bag's left_conserved + right_conserved, then locate the whole
     window (left + guide_slot + right) in the Rfam pool and print the
     source family + position + surrounding context. If the algorithm
     works, the (left, right) pair is byte-identical to a
     (rfam_seq[i:i+lcl], rfam_seq[i+lcl+L:i+lcl+L+rcl]) pair for some
     (rfam_seq, i).
  4. 7-mode axis parity: bag_guide_L, n_sites, left/right_conserved_len,
     nc_planted_len, nc_noise_len, per_site_target_start,
     per_site_planted_m — all mode-invariant axes must have overlapping
     distributions.
"""
from __future__ import annotations
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generator_v5.bag_v7_real import (
    VALID_V7_REAL_NEGATIVE_MODES, build_bag_v7_real, _load_rfam_pool,
    _RFAM_FAMILIES, _RFAM_CACHE_DIR,
)
from scripts.generator_v5.real_flank_pool import RealFlankPool


N_BAGS = 1000
SEED = 0


def dist(v, fmt="{:6.1f}"):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return "n=0 (no planted sites in this mode)"
    return (f"n={len(v):5d}  min={fmt.format(v.min())}  "
              f"p25={fmt.format(np.percentile(v, 25))}  "
              f"p50={fmt.format(np.median(v))}  "
              f"p75={fmt.format(np.percentile(v, 75))}  "
              f"max={fmt.format(v.max())}  "
              f"mean={fmt.format(v.mean())}")


def find_contiguous_source(pool: list[str], left: str, right: str,
                             guide_slot_len: int) -> tuple[int | None,
                                                              int | None]:
    """Search the pool for a Rfam sequence containing `left` + <any
    guide_slot_len bases> + `right` as a contiguous substring. Returns
    (pool_index, position) or (None, None) if not found."""
    for idx, s in enumerate(pool):
        pos = 0
        while True:
            j = s.find(left, pos)
            if j < 0:
                break
            k = j + len(left) + guide_slot_len
            if s[k:k + len(right)] == right:
                return idx, j
            pos = j + 1
    return None, None


def main():
    print(f"# V8.4 Step 3 verification — Rfam nc scaffold + contiguous-window\n")

    # (1) Rfam pool composition
    pool = _load_rfam_pool()
    print(f"## (1) Rfam pool composition")
    print(f"  cache dir: {_RFAM_CACHE_DIR}")
    print(f"  families:  {_RFAM_FAMILIES}")
    print(f"  total sequences (post length + ACGT filter): {len(pool)}")
    lens = [len(s) for s in pool]
    print(f"  length distribution: {dist(lens, fmt='{:5.0f}')}")

    # Length by family — reload each family file and count.
    import gzip
    for fam in _RFAM_FAMILIES:
        raw = (_RFAM_CACHE_DIR / f"{fam}.fa.gz").read_bytes()
        text = gzip.decompress(raw).decode("utf-8", errors="replace")
        n_raw = text.count(">")
        # count kept ones (after filter identical to loader)
        kept = 0
        header = None
        parts: list[str] = []
        for line in text.splitlines():
            if line.startswith(">"):
                if header is not None:
                    s = ("".join(parts).upper().replace("U", "T")
                            .replace(".", "").replace("-", ""))
                    if 100 <= len(s) <= 250 and all(c in "ACGT" for c in s):
                        kept += 1
                header = line[1:]
                parts = []
            else:
                parts.append(line.strip())
        if header is not None:
            s = ("".join(parts).upper().replace("U", "T")
                    .replace(".", "").replace("-", ""))
            if 100 <= len(s) <= 250 and all(c in "ACGT" for c in s):
                kept += 1
        print(f"    {fam}  raw={n_raw:5d}  kept={kept:5d}")
    print()

    # (2) Per-bag Rfam-segment length distribution
    real_pool = RealFlankPool.load_default()
    rng = random.Random(SEED)
    bag_seg_total_lens = []
    left_lens = []
    right_lens = []
    for i in range(N_BAGS):
        b = build_bag_v7_real(bag_id=f"v84_len_{i:06d}",
                                rng=rng, real_flank_pool=real_pool,
                                negative_mode="none")
        if b is None:
            continue
        left_lens.append(b.left_conserved_len)
        right_lens.append(b.right_conserved_len)
        bag_seg_total_lens.append(b.left_conserved_len
                                        + b.right_conserved_len)
    print(f"## (2) per-bag Rfam-segment lengths across {len(bag_seg_total_lens)} positive bags")
    print(f"  left_conserved_len :  {dist(left_lens)}")
    print(f"  right_conserved_len:  {dist(right_lens)}")
    print(f"  left + right total :  {dist(bag_seg_total_lens)}")
    print()

    # (3) Contiguous-window proof — 10 random positive bags
    print(f"## (3) contiguous-window proof — 10 random positive bags")
    print(f"  for each bag: bag's (left, right) MUST appear as (rfam[i:i+lcl], rfam[i+lcl+L:i+lcl+L+rcl]) somewhere in the Rfam pool")
    rng = random.Random(SEED + 1)
    n_ok = 0
    n_bad = 0
    for i in range(10):
        b = build_bag_v7_real(bag_id=f"v84_proof_{i:06d}",
                                rng=rng, real_flank_pool=real_pool,
                                negative_mode="none")
        if b is None:
            continue
        left = b.left_conserved
        right = b.right_conserved
        L = b.bag_guide_L
        lcl = b.left_conserved_len
        rcl = b.right_conserved_len
        idx, pos = find_contiguous_source(pool, left, right, L)
        if idx is None:
            n_bad += 1
            print(f"  bag {i}: NOT FOUND (contiguous-window property VIOLATED)")
            print(f"    left ({lcl}bp): {left!r}")
            print(f"    right ({rcl}bp): {right!r}")
        else:
            n_ok += 1
            src = pool[idx]
            middle = src[pos + lcl: pos + lcl + L]
            print(f"  bag {i}: OK  L={L} lcl={lcl} rcl={rcl}  "
                  f"src=pool[{idx}] pos={pos}")
            print(f"    left  : {left!r}")
            print(f"    middle (Rfam's own bases at guide slot, discarded): {middle!r}")
            print(f"    right : {right!r}")
    print(f"\n  contiguous-window check: {n_ok}/{n_ok + n_bad} bags PASS")
    print()

    # (4) 7-mode axis parity
    print(f"## (4) 7-mode axis parity, {N_BAGS} bags per mode")
    per_mode = {}
    for mode in VALID_V7_REAL_NEGATIVE_MODES:
        rng = random.Random(SEED)
        bags = []
        for i in range(N_BAGS):
            b = build_bag_v7_real(bag_id=f"v84_axis_{mode}_{i:06d}",
                                    rng=rng, real_flank_pool=real_pool,
                                    negative_mode=mode)
            if b is not None:
                bags.append(b)
        per_mode[mode] = bags

    def row(label, extract, fmt="{:5.1f}"):
        print(f"  --- {label} ---")
        for mode in VALID_V7_REAL_NEGATIVE_MODES:
            vs = extract(per_mode[mode])
            print(f"    {mode:>22s}  {dist(vs, fmt)}")
        print()

    row("bag_guide_L",             lambda bs: [b.bag_guide_L for b in bs])
    row("n_sites",                 lambda bs: [b.n_sites for b in bs])
    row("left_conserved_len",      lambda bs: [b.left_conserved_len for b in bs])
    row("right_conserved_len",     lambda bs: [b.right_conserved_len for b in bs])
    row("len(nc_planted)",         lambda bs: [len(b.nc_planted) for b in bs],
          fmt="{:6.1f}")
    row("len(nc_noise)",           lambda bs: [len(b.nc_noise) for b in bs],
          fmt="{:6.1f}")
    row("per_site_target_start",
          lambda bs: [ts for b in bs for ts in b.per_site_target_start],
          fmt="{:6.1f}")
    row("per_site_planted_m (planted sites only)",
          lambda bs: [b.per_site_planted_m[i] for b in bs
                          for i in range(b.n_sites) if b.per_site_is_planted[i]])

    print("# V8.4 Step 3 verification complete")


if __name__ == "__main__":
    main()
