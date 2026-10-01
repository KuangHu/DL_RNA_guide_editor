"""V8.4 Phase 3 acceptance A/B/C on the 7 × 50K corpora.

A. CORPUS: per-corpus bag+site counts, schema check, 7-mode axis
   consistency across all corpora, flank uniqueness, nc content hash.
B. ALIGNMENT: per positive bag, gold plant position vs actual m_max
   argmax on nc, peak width, per-site m std at gold position.
C. CONSTRUCTION: Rfam family distribution on positive corpus
   (family-balanced sampling verification at scale), contiguous-window
   verification on a random subsample, truncation fractions.
"""
from __future__ import annotations
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.generator_v5.bag_v7_real as _bag_mod
from scripts.generator_v5.bag_v7_real import (
    _load_rfam_pool, _RFAM_FAMILIES,
)


CORPUS_DIR = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                    "v84_generation/50k")
MODES = ("none", "partial", "scattered", "flank_scattered",
           "no_alignment", "repeat_flank", "tsd_negative")


def dist(v, fmt="{:6.1f}"):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return "n=0"
    return (f"n={len(v):6d}  min={fmt.format(v.min())}  "
              f"p25={fmt.format(np.percentile(v, 25))}  "
              f"p50={fmt.format(np.median(v))}  "
              f"p75={fmt.format(np.percentile(v, 75))}  "
              f"max={fmt.format(v.max())}  mean={fmt.format(v.mean())}")


def load_records(mode: str) -> list[dict]:
    path = CORPUS_DIR / f"v84_{mode}.jsonl"
    return [json.loads(l) for l in open(path)]


def bag_sites_by_id(records: list[dict]) -> dict[str, list[dict]]:
    bags = defaultdict(list)
    for r in records:
        bags[r["transposase_id"]].append(r)
    return dict(bags)


def m_max_at(nc: str, guide: str, pos: int) -> int:
    L = len(guide)
    if pos < 0 or pos + L > len(nc):
        return -1
    return sum(1 for i in range(L) if nc[pos + i] == guide[i])


def m_max_scan(nc: str, guide: str) -> np.ndarray:
    L = len(guide)
    n_pos = len(nc) - L + 1
    if n_pos <= 0:
        return np.array([], dtype=np.int32)
    ncb = np.frombuffer(nc.encode("ascii"), dtype=np.uint8)
    gb = np.frombuffer(guide.encode("ascii"), dtype=np.uint8)
    m = np.zeros(n_pos, dtype=np.int32)
    for i in range(n_pos):
        m[i] = int((ncb[i:i + L] == gb).sum())
    return m


def group_A(all_recs: dict[str, list[dict]]) -> None:
    print("## A. CORPUS")
    print(f"  {'mode':>18s}  {'n_records':>10s}  {'n_bags':>7s}  {'mean_sites':>10s}  {'schema_ok':>10s}")
    required_keys = {"site_id", "transposase_id", "ncrna_id",
                       "inputs", "labels", "generator_metadata"}
    for mode in MODES:
        recs = all_recs[mode]
        bag_counts = Counter(r["transposase_id"] for r in recs)
        schema_ok = all(required_keys <= set(r.keys()) for r in recs[:1000])
        print(f"  {mode:>18s}  {len(recs):10d}  {len(bag_counts):7d}  "
              f"{len(recs) / len(bag_counts):10.3f}  {str(schema_ok):>10s}")
    print()

    # Cross-mode axis consistency at bag_id = "v8_none_bag_000000" etc.
    # Since bag_ids are prefixed by mode name in generator, cross-mode
    # comparison uses the numeric bag index. Group by index.
    print("  cross-mode axis consistency (bag index 0 in each mode)")
    per_mode_bag0 = {}
    for mode in MODES:
        for r in all_recs[mode]:
            if r["transposase_id"].endswith("_bag_000000"):
                per_mode_bag0.setdefault(mode, []).append(r)
        s = per_mode_bag0[mode][0]
        n_sites = s["labels"]["arch"]["n_sites"]
        L = s["labels"]["guide_length"]
        guide = s["labels"]["guide_dna"]
        print(f"    {mode:>18s}  n_sites={n_sites}  L={L}  guide[:11]={guide[:11]!r}")
    # If rng aligned, n_sites and guide should be identical across modes
    # for the same bag_index (except scattered — different per-site
    # guide_dna within a bag).
    print()

    # Flank uniqueness within positive corpus (should be near-unique
    # since we sample from a 50-genome pool).
    print("  flank uniqueness within positive corpus")
    flanks_pos = [r["inputs"]["flank"] for r in all_recs["none"]]
    unique = len(set(flanks_pos))
    print(f"    n_flanks={len(flanks_pos)}  n_unique={unique}  "
          f"unique_frac={unique / len(flanks_pos):.4f}")
    print()

    # nc content hash cross-mode agreement — positive vs partial (both
    # have same rng-drawn nc_planted_base). For each bag_index in the
    # first 200, compare hash of nc_planted after plant. Should DIFFER
    # (partial only plants a subset of sites).
    # Actually — the nc CONTENT is identical for none vs partial when
    # bag has n_planted = n_sites (or when partial happens to plant
    # all sites, which is not possible per the branch code). Otherwise
    # differs. This check is more sanity than test.
    print("  nc_planted hash sample (first 200 bags, positive vs partial)")
    same = 0
    total = 0
    for idx in range(200):
        bid = f"v8_none_bag_{idx:06d}"
        pos_recs = [r for r in all_recs["none"]
                        if r["transposase_id"] == bid]
        pid = bid.replace("none", "partial")
        par_recs = [r for r in all_recs["partial"]
                        if r["transposase_id"] == pid]
        if not pos_recs or not par_recs:
            continue
        # active nc (all sites in a bag share the same nc)
        ai_pos = pos_recs[0]["labels"]["active_noncoding_index"]
        ai_par = par_recs[0]["labels"]["active_noncoding_index"]
        h_pos = hashlib.md5(pos_recs[0]["inputs"]["noncoding_regions"][ai_pos].encode()).hexdigest()[:12]
        h_par = hashlib.md5(par_recs[0]["inputs"]["noncoding_regions"][ai_par].encode()).hexdigest()[:12]
        total += 1
        if h_pos == h_par:
            same += 1
    print(f"    active-nc hash equal (pos vs partial): {same}/{total}")
    print()


def group_B(all_recs: dict[str, list[dict]]) -> None:
    print("## B. ALIGNMENT (positive corpus)")
    pos = all_recs["none"]
    # For each POSITIVE record, verify m_max at the gold guide position
    # equals ~L, and scan for the peak position.
    peak_offsets = []
    peak_heights = []
    m_at_gold = []
    n_ok = 0
    n_missing_span = 0
    for r in pos:
        labels = r["labels"]
        span = labels.get("guide_span_in_active_noncoding")
        if span is None:
            n_missing_span += 1
            continue
        gold_pos = int(span[0])
        active_idx = int(labels["active_noncoding_index"])
        nc = r["inputs"]["noncoding_regions"][active_idx]
        # Compute concat_nc from all regions with N spacers, since
        # gold_pos is in concat coord.
        MAX_L = int(r["generator_metadata"]["max_l_at_generation"])
        spacer_len = int(r["generator_metadata"]["concat_spacer_len"])
        spacer = "N" * spacer_len
        parts = []
        for k, reg in enumerate(r["inputs"]["noncoding_regions"]):
            if k > 0:
                parts.append(spacer)
            parts.append(reg)
        concat = "".join(parts)
        guide = labels["guide_dna"]
        L = len(guide)
        m_gold = m_max_at(concat, guide, gold_pos)
        m_at_gold.append(m_gold)
        # Only run the full scan on a subsample to keep runtime bounded.
        if n_ok < 5000:
            m_scan = m_max_scan(concat, guide)
            if len(m_scan) > 0:
                pk = int(np.argmax(m_scan))
                peak_offsets.append(pk - gold_pos)
                peak_heights.append(int(m_scan.max()))
        n_ok += 1
    print(f"  n_positive_records={len(pos)}  n_missing_span={n_missing_span}  "
          f"n_scored={n_ok}")
    print(f"  m at gold position (all pos): {dist(np.asarray(m_at_gold))}")
    print(f"  (scanned subsample n={len(peak_heights)})")
    print(f"  peak position offset from gold: {dist(np.asarray(peak_offsets))}")
    print(f"  peak height (max m):            {dist(np.asarray(peak_heights))}")
    # per-site m std at gold — for each BAG, compute std of m_at_gold
    # across its K sites (should be small since all sites plant same
    # guide at same planted_m distribution).
    bags = bag_sites_by_id(pos)
    per_site_m_stds = []
    for bid, recs in list(bags.items())[:5000]:
        m_vals = []
        for r in recs:
            span = r["labels"].get("guide_span_in_active_noncoding")
            if span is None:
                continue
            gold_pos = int(span[0])
            active_idx = int(r["labels"]["active_noncoding_index"])
            spacer_len = int(r["generator_metadata"]["concat_spacer_len"])
            spacer = "N" * spacer_len
            parts = []
            for k, reg in enumerate(r["inputs"]["noncoding_regions"]):
                if k > 0:
                    parts.append(spacer)
                parts.append(reg)
            concat = "".join(parts)
            m_vals.append(m_max_at(concat, r["labels"]["guide_dna"], gold_pos))
        if m_vals:
            per_site_m_stds.append(float(np.std(m_vals)))
    print(f"  per-site m std at gold (per bag, n={len(per_site_m_stds)}): "
          f"{dist(np.asarray(per_site_m_stds), fmt='{:6.3f}')}")
    print()


def group_C(all_recs: dict[str, list[dict]]) -> None:
    print("## C. CONSTRUCTION")

    # C1: sample-based Rfam family attribution on positive corpus
    _load_rfam_pool()   # ensures _RFAM_POOL_BY_FAMILY populated
    print(f"  (C1) Rfam family attribution (200 random positive bags, "
          f"expect ~25 per family)")
    pos_records = all_recs["none"]
    pos_bags = bag_sites_by_id(pos_records)
    bag_ids = list(pos_bags.keys())
    random.Random(0).shuffle(bag_ids)
    fam_counter = Counter()
    for bid in bag_ids[:200]:
        recs = pos_bags[bid]
        r0 = recs[0]
        labels = r0["labels"]
        span = labels["guide_span_in_active_noncoding"]
        if span is None:
            continue
        active_idx = int(labels["active_noncoding_index"])
        nc = r0["inputs"]["noncoding_regions"][active_idx]
        guide = labels["guide_dna"]
        L = len(guide)
        gpos_concat = int(span[0])
        # Recover local position on active region: subtract offset.
        spacer_len = int(r0["generator_metadata"]["concat_spacer_len"])
        if active_idx == 0:
            local = gpos_concat
        else:
            local = gpos_concat
            for k in range(active_idx):
                local -= len(r0["inputs"]["noncoding_regions"][k]) + spacer_len
        # Peek left+right context beyond the guide (35 bp each side is
        # bracket max). Then find which Rfam family contains that
        # (left, right) as a contiguous pair.
        left_ctx_len = min(35, local)
        right_ctx_len = min(35, len(nc) - (local + L))
        left_ctx = nc[local - left_ctx_len:local]
        right_ctx = nc[local + L:local + L + right_ctx_len]
        # Search for a matching contiguous window. We do a linear scan
        # per family looking for a long-enough match — since we don't
        # know exact bracket lengths, take an inner window of e.g. 15
        # bp on each side (guaranteed to be ≤ actual bracket len).
        L_search = 15
        R_search = 15
        left_probe = left_ctx[-L_search:] if len(left_ctx) >= L_search else left_ctx
        right_probe = right_ctx[:R_search] if len(right_ctx) >= R_search else right_ctx
        found = None
        for fam in _RFAM_FAMILIES:
            for s in _bag_mod._RFAM_POOL_BY_FAMILY[fam]:
                p = s.find(left_probe)
                while p >= 0:
                    end_left = p + len(left_probe)
                    right_start = end_left + L
                    if s[right_start:right_start + len(right_probe)] == right_probe:
                        found = fam
                        break
                    p = s.find(left_probe, p + 1)
                if found:
                    break
            if found:
                break
        if found is None:
            fam_counter["_NOT_FOUND"] += 1
        else:
            fam_counter[found] += 1
    for fam in _RFAM_FAMILIES:
        print(f"    {fam:>10s}: {fam_counter.get(fam, 0):3d}")
    if fam_counter.get("_NOT_FOUND", 0):
        print(f"    _NOT_FOUND: {fam_counter['_NOT_FOUND']:3d}  (should be 0)")
    total_placed = sum(fam_counter[fam] for fam in _RFAM_FAMILIES)
    chi_sq = sum((fam_counter.get(fam, 0) - total_placed / 8) ** 2
                     / (total_placed / 8) for fam in _RFAM_FAMILIES)
    print(f"    chi-sq (df=7) vs uniform: {chi_sq:.2f}")
    print()

    # C2: truncation / boundary check — ts positions should be in
    # [ts_lo_bound, ts_hi_bound]. Compute per-record.
    print(f"  (C2) target_start boundary check on positive corpus")
    from scripts.generator_v5.bag_v7_real import (
        RNA_CONSERVED_LEN_MIN, RNA_CONSERVED_LEN_MAX,
        FLANK_LEN, TARGET_L_MAX,
    )
    per_bag_ts = defaultdict(list)
    for r in pos_records:
        span = r["labels"]["target_position_in_flank"]
        per_bag_ts[r["transposase_id"]].append(int(span[0]))
    ts_all = [ts for tss in per_bag_ts.values() for ts in tss]
    ts_hi_bound = FLANK_LEN - TARGET_L_MAX - RNA_CONSERVED_LEN_MAX  # 120 - 14 - 35 = 71
    ts_lo_bound = RNA_CONSERVED_LEN_MIN                              # 15
    print(f"    ts distribution: {dist(np.asarray(ts_all))}")
    print(f"    ts_lo_bound={ts_lo_bound}  ts_hi_bound={ts_hi_bound} "
          f"(from RNA_CONSERVED_LEN_MIN=15, RNA_CONSERVED_LEN_MAX=35, TARGET_L_MAX=14)")
    n_below = sum(1 for ts in ts_all if ts < ts_lo_bound)
    n_above = sum(1 for ts in ts_all if ts > ts_hi_bound)
    print(f"    n_below_lo={n_below}  n_above_hi={n_above}  "
          f"(truncation should be minimal by construction)")
    print()

    # C3: per-bag ts span (invariant 2 held by construction; verify at scale)
    span_max = max(max(tss) - min(tss) for tss in per_bag_ts.values())
    span_p95 = float(np.percentile(
        [max(tss) - min(tss) for tss in per_bag_ts.values()], 95))
    print(f"  (C3) per-bag ts span (invariant 2, non-scattered modes ≤ 4)")
    print(f"    span p95={span_p95:.1f}  span max={span_max}  "
          f"(expected max 4 for positive)")


def main():
    print(f"# V8.4 Phase 3 acceptance A/B/C\n")
    print(f"# loading 7 corpora from {CORPUS_DIR}")
    all_recs = {mode: load_records(mode) for mode in MODES}
    print()

    group_A(all_recs)
    group_B(all_recs)
    group_C(all_recs)

    print("\n# V8.4 Phase 3 acceptance complete")


if __name__ == "__main__":
    main()
