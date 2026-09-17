"""Channel A per-site position localization accuracy on Durrant real data.

Uses existing match_table _compute_site_arrays on the raw Durrant
cognate jsonl (no shard build needed). For each site s in each Tnp:
  - compute m_max array at L=11 over all nc positions (both fwd + rc)
  - argmax_position(m_max)
  - compare to gold guide_start_in_nc from IS110_gold/annotation/durrant_gold_v1

Metrics per site:
  - top-1 exact: argmax matches gold within tol nt
  - top-5 within: gold in the top-5 highest-m positions
  - m_at_gold: match count at gold position (context)

Reports per file + overall + stratified by gold match_status
(not_found / partial / exact) + per orient.

n_sites = 325 gives CI ≈ ±5 pp — much stronger than n=65 bag AUROC.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import math
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.v5a_framework.match_table import _compute_site_arrays

BASE = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold")
GOLD_TABLE = BASE / "annotation" / "durrant_gold_v1.jsonl"

L_TARGET = 11
RNG_SEED = 0
LOCAL_PEAK_HALFWIDTH = 10  # neighborhood radius for A_LOCAL_PEAK correction


def _base_entropy(seq: str) -> float:
    """Shannon entropy (bits) of the base composition of a k-mer."""
    if not seq: return 0.0
    counts = {b: 0 for b in "ACGT"}
    for c in seq.upper():
        if c in counts: counts[c] += 1
    total = sum(counts.values())
    if total == 0: return 0.0
    h = 0.0
    for k in counts.values():
        if k == 0: continue
        p = k / total
        h -= p * math.log2(p)
    return h


def _max_base_frac(seq: str) -> float:
    """Fraction of the most-common base in a k-mer (proxy for low-complexity)."""
    if not seq: return 0.0
    counts = {b: 0 for b in "ACGT"}
    for c in seq.upper():
        if c in counts: counts[c] += 1
    total = sum(counts.values())
    if total == 0: return 0.0
    return max(counts.values()) / total


_COMP = str.maketrans("ACGTacgt", "TGCAtgca")

def _rc(s: str) -> str:
    return s.translate(_COMP)[::-1]


def _parse_guide_start(span) -> int | None:
    """Extract start index from `guide_span_in_active_noncoding`. Robust to list / dict / tuple."""
    if span is None: return None
    if isinstance(span, (list, tuple)) and len(span) >= 1:
        try: return int(span[0])
        except (TypeError, ValueError): return None
    if isinstance(span, dict):
        for k in ("start", "begin", "s"):
            if k in span:
                try: return int(span[k])
                except (TypeError, ValueError): return None
    if isinstance(span, int):
        return span
    return None


def _local_peak_score(S: np.ndarray, halfwidth: int) -> np.ndarray:
    """Score(p) = S(p) - mean(S over ±halfwidth excluding p). Positions on broad
    plateaus get down-weighted; sharp peaks that stand out get preferred."""
    n = S.size
    if n == 0: return S.astype(np.float32)
    # Cumulative sum for O(n) windowed mean
    cs = np.cumsum(S.astype(np.float64))
    cs = np.concatenate(([0.0], cs))
    out = np.empty(n, dtype=np.float32)
    for p in range(n):
        lo = max(0, p - halfwidth)
        hi = min(n, p + halfwidth + 1)
        window_sum = cs[hi] - cs[lo]
        window_count = hi - lo
        # Exclude center
        neighbor_sum = window_sum - S[p]
        neighbor_count = window_count - 1
        neighbor_mean = neighbor_sum / neighbor_count if neighbor_count > 0 else 0.0
        out[p] = float(S[p]) - neighbor_mean
    return out


def _paired_diff_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 5000,
                            rng_seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap CI on mean(a - b). Returns (mean_diff, lo, hi) at 95%."""
    assert a.shape == b.shape
    rng = np.random.default_rng(rng_seed)
    n = a.shape[0]
    diffs = a - b
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = diffs[idx].mean()
    return float(diffs.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _load_gold_table() -> dict:
    """site_id → gold info. Also strips 'realbg_' prefix for cross-file matching."""
    d = {}
    for line in open(GOLD_TABLE):
        r = json.loads(line)
        d[r["site_id"]] = r
    return d


def _lookup_gold(gold: dict, site_id: str) -> dict | None:
    """Try the site_id as-is; if not found, try stripping 'realbg_' prefix."""
    if site_id in gold:
        return gold[site_id]
    # realbg record ids look like: durrant_realbg_bridge_RNA_..._site_0000
    # gold table has:              durrant_bridge_RNA_..._site_0000
    stripped = site_id.replace("durrant_realbg_", "durrant_")
    return gold.get(stripped)


def _score_site(nc: str, flank: str) -> tuple[np.ndarray, str]:
    """Return (m_max_array_at_L=L_TARGET, best_orient). Array is
    max_over_orientation(m_max) at each nc position. LEGACY orient-max form
    kept for backward compat; new callers should use `_score_site_per_orient`."""
    site_arrays = _compute_site_arrays(nc, flank, orients=("fwd", "rc"),
                                              Ls=(L_TARGET,), excl_widths=(0,))
    fwd_arr = site_arrays[("fwd", L_TARGET)].m_max_by_excl[0]
    rc_arr = site_arrays[("rc", L_TARGET)].m_max_by_excl[0]
    # Pointwise max over orientations
    n = min(fwd_arr.shape[0], rc_arr.shape[0])
    m_max = np.maximum(fwd_arr[:n], rc_arr[:n])
    best_orient = "fwd" if fwd_arr[:n].sum() >= rc_arr[:n].sum() else "rc"
    return m_max, best_orient


def _score_site_per_orient(nc: str, flank: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (m_max_fwd, m_max_rc) both aligned to the same nc positions.
    Consumers can construct orient-consistent S per orient separately."""
    site_arrays = _compute_site_arrays(nc, flank, orients=("fwd", "rc"),
                                              Ls=(L_TARGET,), excl_widths=(0,))
    fwd_arr = site_arrays[("fwd", L_TARGET)].m_max_by_excl[0]
    rc_arr = site_arrays[("rc", L_TARGET)].m_max_by_excl[0]
    n = min(fwd_arr.shape[0], rc_arr.shape[0])
    return fwd_arr[:n], rc_arr[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+",
                     default=["durrant_cognate.jsonl",
                              "durrant_cognate_realbg.jsonl",
                              "durrant_cognate_realbg_ncpad240.jsonl"])
    ap.add_argument("--tol", type=int, default=0,
                     help="Tolerance in nt for top-1 exact match")
    args = ap.parse_args()

    gold = _load_gold_table()
    print(f"[loc] gold table: {len(gold)} site records")

    all_file_results: dict[str, dict[str, dict]] = {}   # fname -> tnp_id -> row

    for fname in args.files:
        path = BASE / "inference" / fname
        if not path.exists():
            print(f"  [{fname}] file not found; skipping")
            continue
        print(f"\n=== {fname} ===")

        # Group records by tnp_id. Also verify coord system per site by locating
        # bridge_rna_sequence (from gold) as a substring in nc — the offset is
        # then the correction that maps unpadded gold_start_in_nc → padded coords.
        by_tnp: dict[str, list[dict]] = defaultdict(list)
        offsets_seen: dict[int, int] = defaultdict(int)
        n_verified = 0
        n_total = 0
        for line in open(path):
            r = json.loads(line)
            g = _lookup_gold(gold, r["site_id"])
            if g is None or g.get("guide_start_in_nc") is None:
                continue
            n_total += 1
            nc = r["inputs"]["noncoding_regions"][r["labels"].get("active_noncoding_index", 0)]
            bridge_dna = g.get("bridge_rna_sequence", "").replace("U", "T").replace("u", "t").upper()
            gs = int(g["guide_start_in_nc"])
            ge = int(g["guide_end_in_nc"])
            # Locate bridge_rna in nc → per-site padding offset
            nc_offset = nc.upper().find(bridge_dna) if bridge_dna else -1
            if nc_offset < 0:
                r["_nc_offset"] = None
                continue
            # Sanity: guide region in nc matches guide region in bridge_rna
            if nc[nc_offset + gs:nc_offset + ge].upper() != bridge_dna[gs:ge]:
                r["_nc_offset"] = None
                continue
            r["_nc_offset"] = nc_offset
            r["_gold"] = g
            r["_effective_gold_start"] = gs + nc_offset
            offsets_seen[nc_offset] += 1
            n_verified += 1
            by_tnp[r["transposase_id"]].append(r)

        print(f"  ── coord verification (bridge_rna substring in nc) ──")
        print(f"    verified: {n_verified}/{n_total}   offsets seen: {dict(sorted(offsets_seen.items()))}")
        if n_verified == 0:
            print(f"  no verified sites, skipping file")
            continue

        # Per-Tnp: score each site (m_max at L=11), aggregate S = |{sites: m≥8 at p}|,
        # take argmax_p S → Channel A localization prediction. Compare to gold.
        tnp_rows = []
        per_site_rows = []   # also keep per-site for by-status stratification
        n_gold_off_tail = 0
        n_tnp_no_sites = 0
        for tnp, sites in by_tnp.items():
            if not sites: continue
            # Effective gold_start = raw gold_start_in_nc + per-site padding offset.
            # All sites in a Tnp share the same nc and thus the same offset (per verify).
            gold_start = int(sites[0]["_effective_gold_start"])
            match_statuses = [s["_gold"].get("match_status", "?") for s in sites]
            # Reference nc from first site (all sites in a Tnp share the same nc per 100% co-occurrence anchor)
            nc_ref = sites[0]["inputs"]["noncoding_regions"][
                sites[0]["labels"].get("active_noncoding_index", 0)]
            # Score each site's m_max arrays (per orient AND orient-max for legacy)
            site_arrays = []           # orient-max per position (legacy)
            site_arrays_fwd = []       # fwd only
            site_arrays_rc  = []       # rc only
            n_positions = None
            for r in sites:
                nc = r["inputs"]["noncoding_regions"][r["labels"].get("active_noncoding_index", 0)]
                flank = r["inputs"]["flank"]
                m_fwd, m_rc = _score_site_per_orient(nc, flank)
                if m_fwd.size == 0 or m_rc.size == 0: continue
                m_max = np.maximum(m_fwd, m_rc)
                site_arrays.append(m_max)
                site_arrays_fwd.append(m_fwd)
                site_arrays_rc.append(m_rc)
                n_here = m_max.size
                n_positions = n_here if n_positions is None else min(n_positions, n_here)
            if not site_arrays or n_positions is None:
                n_tnp_no_sites += 1
                continue
            # Truncate to common length
            site_arrays     = [a[:n_positions] for a in site_arrays]
            site_arrays_fwd = [a[:n_positions] for a in site_arrays_fwd]
            site_arrays_rc  = [a[:n_positions] for a in site_arrays_rc]
            n_sites = len(site_arrays)
            # LEGACY S_p = |{sites: max_orient m_p >= 8}|  (orient-max aggregation)
            S = np.zeros(n_positions, dtype=np.int32)
            for a in site_arrays:
                S += (a >= 8).astype(np.int32)
            argmax_S = int(S.argmax())
            argmax_S_val = int(S.max())
            # ORIENT-CONSISTENT S_p = max_orient |{sites: m_orient_p >= 8}|
            S_fwd = np.zeros(n_positions, dtype=np.int32)
            for a in site_arrays_fwd:
                S_fwd += (a >= 8).astype(np.int32)
            S_rc = np.zeros(n_positions, dtype=np.int32)
            for a in site_arrays_rc:
                S_rc += (a >= 8).astype(np.int32)
            S_oc = np.maximum(S_fwd, S_rc)
            argmax_S_oc = int(S_oc.argmax())
            S_oc_at_argmax = int(S_oc.max())
            S_oc_at_gold = int(S_oc[gold_start]) if 0 <= gold_start < n_positions else -1
            # Rank of gold in S (higher S = higher rank; ties broken adversarially)
            if gold_start >= n_positions:
                n_gold_off_tail += 1
                continue
            S_at_gold = int(S[gold_start])
            # Rank of gold: how many positions strictly outrank it (0 means gold is unique top).
            n_strict_above = int((S > S_at_gold).sum())
            n_at_or_above = int((S >= S_at_gold).sum())
            # top-k membership: some position within tol of gold sits in the top-k of S
            def _in_topk_of(scores: np.ndarray, k: int) -> bool:
                k = min(k, scores.size)
                if k == 0: return False
                top_idx = np.argpartition(scores, -k)[-k:]
                return any(abs(int(p) - gold_start) <= args.tol for p in top_idx)
            top1_hit = _in_topk_of(S, 1)
            top5_hit = _in_topk_of(S, 5)
            top10_hit = _in_topk_of(S, 10)
            # A_LOCAL_PEAK: position-dependent correction
            lp_scores = _local_peak_score(S, LOCAL_PEAK_HALFWIDTH)
            lp_argmax = int(lp_scores.argmax())
            lp_top1 = _in_topk_of(lp_scores, 1)
            lp_top5 = _in_topk_of(lp_scores, 5)
            lp_top10 = _in_topk_of(lp_scores, 10)
            # A_ORIENT_CONSISTENT: enforce single-orient consistency across sites
            oc_top1 = _in_topk_of(S_oc, 1)
            oc_top5 = _in_topk_of(S_oc, 5)
            oc_top10 = _in_topk_of(S_oc, 10)
            # 11-mers at gold and at Channel A winner (nc_ref shared across sites in this Tnp)
            gold_seq = nc_ref[gold_start:gold_start + L_TARGET] if gold_start + L_TARGET <= len(nc_ref) else ""
            winner_seq = nc_ref[argmax_S:argmax_S + L_TARGET] if argmax_S + L_TARGET <= len(nc_ref) else ""
            tnp_rows.append({
                "tnp_id": tnp,
                "gold_start": gold_start,
                "argmax_S": argmax_S, "S_at_argmax": argmax_S_val,
                "S_at_gold": S_at_gold,
                "n_strict_above": n_strict_above,
                "n_at_or_above": n_at_or_above,
                "top1": top1_hit, "top5": top5_hit, "top10": top10_hit,
                "lp_argmax": lp_argmax,
                "lp_top1": lp_top1, "lp_top5": lp_top5, "lp_top10": lp_top10,
                "oc_argmax": argmax_S_oc,
                "oc_top1": oc_top1, "oc_top5": oc_top5, "oc_top10": oc_top10,
                "S_oc_at_argmax": S_oc_at_argmax, "S_oc_at_gold": S_oc_at_gold,
                "n_sites": n_sites, "nc_len": n_positions,
                "match_statuses": match_statuses,
                "gold_seq": gold_seq, "winner_seq": winner_seq,
                "gold_entropy": _base_entropy(gold_seq),
                "winner_entropy": _base_entropy(winner_seq),
                "gold_max_frac": _max_base_frac(gold_seq),
                "winner_max_frac": _max_base_frac(winner_seq),
            })
            # per-site for stratification: top-1 / top-5 / top-10 per site vs shared gold
            for r, m_arr, ms in zip(sites[:len(site_arrays)], site_arrays, match_statuses):
                if not m_arr.size:
                    continue
                # top-k membership for this individual site
                def _site_in_topk(k: int) -> bool:
                    k = min(k, m_arr.size)
                    top_idx = np.argpartition(m_arr, -k)[-k:]
                    return any(abs(int(p) - gold_start) <= args.tol for p in top_idx)
                per_site_rows.append({
                    "tnp_id": tnp, "match_status": ms,
                    "m_at_gold": int(m_arr[gold_start]) if gold_start < m_arr.size else 0,
                    "m_at_argmax": int(m_arr.max()),
                    "site_top1":  _site_in_topk(1),
                    "site_top5":  _site_in_topk(5),
                    "site_top10": _site_in_topk(10),
                })

        n = len(tnp_rows)
        if n_tnp_no_sites or n_gold_off_tail:
            print(f"  dropped tnps: no_valid_sites={n_tnp_no_sites}  gold_off_tail={n_gold_off_tail}")
        if n == 0:
            print(f"  no valid rows")
            continue
        top1 = np.mean([r["top1"] for r in tnp_rows])
        top5 = np.mean([r["top5"] for r in tnp_rows])
        top10 = np.mean([r["top10"] for r in tnp_rows])
        se = np.sqrt(top1 * (1 - top1) / max(n, 1))
        n_positions_ref = tnp_rows[0]["nc_len"]
        # Analytic null (uniform-random pick): 1/N for top-1, 5/N for top-5, 10/N for top-10
        null_top1 = 1.0 / n_positions_ref
        null_top5 = 5.0 / n_positions_ref
        null_top10 = 10.0 / n_positions_ref
        print(f"  n_tnps={n}  nc_len={n_positions_ref}  n_sites_mean={np.mean([r['n_sites'] for r in tnp_rows]):.2f}")
        print(f"  RANDOM NULL (uniform pick over {n_positions_ref} positions): "
              f"top-1={null_top1:.4f}  top-5={null_top5:.4f}  top-10={null_top10:.4f}")
        print(f"  ── three-level accuracy table ──")
        print(f"    {'level':<32s}  {'top-1':>8s}  {'top-5':>8s}  {'top-10':>8s}  {'×random top-1':>14s}")
        # Per-site level (best-orient argmax on individual site)
        if per_site_rows:
            ps_t1 = np.mean([r["site_top1"] for r in per_site_rows])
            ps_t5 = np.mean([r["site_top5"] for r in per_site_rows])
            ps_t10 = np.mean([r["site_top10"] for r in per_site_rows])
            print(f"    {'per-site argmax (best-orient)':<32s}  {ps_t1:8.4f}  {ps_t5:8.4f}  {ps_t10:8.4f}  {ps_t1/null_top1:14.1f}×")
        print(f"    {'cross-site argmax_p S':<32s}  {top1:8.4f}  {top5:8.4f}  {top10:8.4f}  {top1/null_top1:14.1f}×")
        lp_t1 = np.mean([r["lp_top1"] for r in tnp_rows])
        lp_t5 = np.mean([r["lp_top5"] for r in tnp_rows])
        lp_t10 = np.mean([r["lp_top10"] for r in tnp_rows])
        oc_t1 = np.mean([r["oc_top1"] for r in tnp_rows])
        oc_t5 = np.mean([r["oc_top5"] for r in tnp_rows])
        oc_t10 = np.mean([r["oc_top10"] for r in tnp_rows])
        print(f"    {'A_LOCAL_PEAK (±%d)' % LOCAL_PEAK_HALFWIDTH:<32s}  {lp_t1:8.4f}  {lp_t5:8.4f}  {lp_t10:8.4f}  {lp_t1/null_top1:14.1f}×")
        print(f"    {'A_ORIENT_CONSISTENT':<32s}  {oc_t1:8.4f}  {oc_t5:8.4f}  {oc_t10:8.4f}  {oc_t1/null_top1:14.1f}×")
        print(f"    {'random null':<32s}  {null_top1:8.4f}  {null_top5:8.4f}  {null_top10:8.4f}  {1.0:14.1f}×")
        # Paired within-Tnp: does A_ORIENT_CONSISTENT beat cross-site S argmax?
        s_top1_arr = np.array([r["top1"] for r in tnp_rows], dtype=float)
        lp_top1_arr = np.array([r["lp_top1"] for r in tnp_rows], dtype=float)
        oc_top1_arr = np.array([r["oc_top1"] for r in tnp_rows], dtype=float)
        m_lp, lo_lp, hi_lp = _paired_diff_bootstrap(lp_top1_arr, s_top1_arr)
        m_oc, lo_oc, hi_oc = _paired_diff_bootstrap(oc_top1_arr, s_top1_arr)
        print(f"    paired Δ (A_LOCAL_PEAK − cross-site S) top-1: {m_lp:+.4f}  CI [{lo_lp:+.4f}, {hi_lp:+.4f}]  "
              f"{'SIG' if (lo_lp > 0 or hi_lp < 0) else 'ns'}")
        print(f"    paired Δ (A_ORIENT_CONSISTENT − cross-site S) top-1: {m_oc:+.4f}  CI [{lo_oc:+.4f}, {hi_oc:+.4f}]  "
              f"{'SIG' if (lo_oc > 0 or hi_oc < 0) else 'ns'}")

        # ── Stratify top-1 by S_at_gold ── direct Cause-1 confirmation
        print(f"\n  ── top-1 stratified by S_at_gold ──")
        print(f"    {'S_at_gold':>10s}  {'n_tnps':>7s}  {'S top-1':>9s}  {'LP top-1':>10s}  {'OC top-1':>10s}")
        for s_val in sorted({r["S_at_gold"] for r in tnp_rows}):
            sub = [r for r in tnp_rows if r["S_at_gold"] == s_val]
            n_sub = len(sub)
            t1 = np.mean([r["top1"] for r in sub])
            lp = np.mean([r["lp_top1"] for r in sub])
            oc = np.mean([r["oc_top1"] for r in sub])
            print(f"    {s_val:>10d}  {n_sub:>7d}  {t1:>9.4f}  {lp:>10.4f}  {oc:>10.4f}")
        # Also stratify by S_at_argmax (deployable — known at inference; S_at_gold is not)
        print(f"\n  ── top-1 stratified by S_at_argmax (deployable confidence bins) ──")
        print(f"    {'S_at_argmax':>12s}  {'n_tnps':>7s}  {'S top-1':>9s}  {'LP top-1':>10s}  {'OC top-1':>10s}")
        for s_val in sorted({r["S_at_argmax"] for r in tnp_rows}):
            sub = [r for r in tnp_rows if r["S_at_argmax"] == s_val]
            n_sub = len(sub)
            t1 = np.mean([r["top1"] for r in sub])
            lp = np.mean([r["lp_top1"] for r in sub])
            oc = np.mean([r["oc_top1"] for r in sub])
            print(f"    {s_val:>12d}  {n_sub:>7d}  {t1:>9.4f}  {lp:>10.4f}  {oc:>10.4f}")
        # Same for S_oc_at_argmax — the orient-consistent deployable bin
        print(f"\n  ── OC-based deployable bin (S_oc_at_argmax) ──")
        print(f"    {'S_oc_argmax':>12s}  {'n_tnps':>7s}  {'OC top-1':>10s}")
        for s_val in sorted({r["S_oc_at_argmax"] for r in tnp_rows}):
            sub = [r for r in tnp_rows if r["S_oc_at_argmax"] == s_val]
            n_sub = len(sub)
            oc = np.mean([r["oc_top1"] for r in sub])
            print(f"    {s_val:>12d}  {n_sub:>7d}  {oc:>10.4f}")
        print(f"  cross-site details:")
        print(f"    top-1 exact (tol={args.tol}) 95%CI ±{1.96*se:.4f}")
        print(f"    mean S_at_gold: {np.mean([r['S_at_gold'] for r in tnp_rows]):.2f}   "
              f"mean S_at_argmax: {np.mean([r['S_at_argmax'] for r in tnp_rows]):.2f}")

        # Distribution of S_at_gold + rank
        s_at_gold_dist = defaultdict(int)
        rank_dist = defaultdict(int)
        for r in tnp_rows:
            s_at_gold_dist[r["S_at_gold"]] += 1
            rank_dist[r["n_strict_above"]] += 1
        print(f"    S_at_gold distribution:            {dict(sorted(s_at_gold_dist.items()))}")
        print(f"    n_strict_above (rank-1) dist:      {dict(sorted(rank_dist.items()))}")

        # Per-site m_max analytics — stratified by match_status
        if per_site_rows:
            print(f"\n  per-site m_max by match_status (n_site_rows={len(per_site_rows)}):")
            for status in ("exact", "partial", "not_found"):
                sub = [r for r in per_site_rows if r["match_status"] == status]
                if not sub: continue
                m_gold = np.mean([r["m_at_gold"] for r in sub])
                m_argmax = np.mean([r["m_at_argmax"] for r in sub])
                t1_s = np.mean([r["site_top1"] for r in sub])
                print(f"    {status:<11s} n={len(sub):>4d}  "
                      f"per-site top-1={t1_s:.4f}  "
                      f"mean m@gold={m_gold:.2f}  m@site_argmax={m_argmax:.2f}")

        # Winner-position vs gold-position 11-mer complexity
        # Focus on Tnps where argmax_S != gold (miss cases) — where is Channel A drawn?
        miss_rows = [r for r in tnp_rows if r["argmax_S"] != r["gold_start"]]
        hit_rows = [r for r in tnp_rows if r["argmax_S"] == r["gold_start"]]
        if miss_rows:
            miss_win_ent = np.mean([r["winner_entropy"] for r in miss_rows])
            miss_gold_ent = np.mean([r["gold_entropy"] for r in miss_rows])
            miss_win_maxf = np.mean([r["winner_max_frac"] for r in miss_rows])
            miss_gold_maxf = np.mean([r["gold_max_frac"] for r in miss_rows])
            # Reference: random-position entropy sampled from the same nc_ref
            rng = np.random.default_rng(RNG_SEED)
            rand_ents = []
            rand_maxf = []
            for r in miss_rows:
                # sample 20 random positions from the same tnp's nc_ref for a within-Tnp null
                nc_here = None
                # Retrieve winner_seq's parent nc: reconstruct from stored fields is hard;
                # instead just approximate by using entropy of positions drawn from a Bernoulli
                # (0.25 each base) at length 11: E[H]=log2(4)≈2.0.
                # Use empirical positions from within nc via winner_seq context? We already
                # have winner_seq/gold_seq. Cheaper: log per-Tnp entropy of ALL 11-mers uniformly.
                pass  # skip within-Tnp random sample here to keep it simple
            print(f"\n  winner-position complexity ({len(miss_rows)} miss Tnps, {len(hit_rows)} hit Tnps):")
            print(f"    {'metric':<24s}  {'miss winner':>12s}  {'miss gold':>12s}  {'hit winner=gold':>16s}")
            hit_ent = np.mean([r["winner_entropy"] for r in hit_rows]) if hit_rows else float("nan")
            hit_maxf = np.mean([r["winner_max_frac"] for r in hit_rows]) if hit_rows else float("nan")
            print(f"    {'mean base entropy':<24s}  {miss_win_ent:12.3f}  {miss_gold_ent:12.3f}  {hit_ent:16.3f}")
            print(f"    {'mean max-base-frac':<24s}  {miss_win_maxf:12.3f}  {miss_gold_maxf:12.3f}  {hit_maxf:16.3f}")
            # Show low-complexity winners (max_frac >= 0.55 → 6+/11 bases same)
            low_comp_winners = [r for r in miss_rows if r["winner_max_frac"] >= 6/11]
            print(f"    low-complexity winners (max-base-frac ≥ 6/11): "
                  f"{len(low_comp_winners)}/{len(miss_rows)} = {len(low_comp_winners)/len(miss_rows):.2f}")
            if low_comp_winners[:5]:
                for r in sorted(low_comp_winners, key=lambda x: -x["winner_max_frac"])[:5]:
                    print(f"      tnp={r['tnp_id'][:40]:40s}  winner_seq={r['winner_seq']}  "
                          f"max_frac={r['winner_max_frac']:.2f}  S_at_argmax={r['S_at_argmax']}")

        # Save for cross-file paired analysis — use normalized tnp key (strip realbg_ prefix)
        def _norm_tnp(t: str) -> str:
            return t.replace("durrant_realbg_", "durrant_")
        all_file_results[fname] = {_norm_tnp(r["tnp_id"]): r for r in tnp_rows}

    # ── Cross-file paired comparison ──
    print(f"\n\n=== paired cross-file analysis (normalized tnp_id) ===")
    fnames = [f for f in args.files if f in all_file_results]
    for i, fa in enumerate(fnames):
        for fb in fnames[i+1:]:
            A = all_file_results[fa]
            B = all_file_results[fb]
            common = sorted(set(A.keys()) & set(B.keys()))
            if len(common) < 10:
                print(f"  [{fa}] vs [{fb}]  n_matched={len(common)} — insufficient for paired CI")
                continue
            t1_a = np.array([A[t]["top1"] for t in common], dtype=float)
            t1_b = np.array([B[t]["top1"] for t in common], dtype=float)
            t5_a = np.array([A[t]["top5"] for t in common], dtype=float)
            t5_b = np.array([B[t]["top5"] for t in common], dtype=float)
            lp_a = np.array([A[t]["lp_top1"] for t in common], dtype=float)
            lp_b = np.array([B[t]["lp_top1"] for t in common], dtype=float)
            oc_a = np.array([A[t]["oc_top1"] for t in common], dtype=float)
            oc_b = np.array([B[t]["oc_top1"] for t in common], dtype=float)
            m1, lo1, hi1 = _paired_diff_bootstrap(t1_a, t1_b)
            m5, lo5, hi5 = _paired_diff_bootstrap(t5_a, t5_b)
            mlp, lolp, hilp = _paired_diff_bootstrap(lp_a, lp_b)
            moc, looc, hioc = _paired_diff_bootstrap(oc_a, oc_b)
            print(f"  [{fa[:38]}] vs [{fb[:38]}]  n_paired={len(common)}")
            print(f"    S top-1:  {t1_a.mean():.4f} vs {t1_b.mean():.4f}  Δ={m1:+.4f}  "
                  f"CI [{lo1:+.4f}, {hi1:+.4f}]  {'SIG' if (lo1 > 0 or hi1 < 0) else 'ns'}")
            print(f"    S top-5:  {t5_a.mean():.4f} vs {t5_b.mean():.4f}  Δ={m5:+.4f}  "
                  f"CI [{lo5:+.4f}, {hi5:+.4f}]  {'SIG' if (lo5 > 0 or hi5 < 0) else 'ns'}")
            print(f"    LP top-1: {lp_a.mean():.4f} vs {lp_b.mean():.4f}  Δ={mlp:+.4f}  "
                  f"CI [{lolp:+.4f}, {hilp:+.4f}]  {'SIG' if (lolp > 0 or hilp < 0) else 'ns'}")
            print(f"    OC top-1: {oc_a.mean():.4f} vs {oc_b.mean():.4f}  Δ={moc:+.4f}  "
                  f"CI [{looc:+.4f}, {hioc:+.4f}]  {'SIG' if (looc > 0 or hioc < 0) else 'ns'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
