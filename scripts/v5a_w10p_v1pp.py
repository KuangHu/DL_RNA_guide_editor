"""W10' + V1'' — final diagnostics.

W10': Does V4.2 have per-Tnp guide consistency across its sites, even with
different ncRNA sequences? Three possible outcomes; if any consistency exists,
Channel B trains on V4.2 without regeneration.

V1'': Proper dual-null cross-family estimate. Replaces the 1.62^5 = 11×
one-sided extrapolation with an empirical dual-null number: dinuc-shuffle
BOTH the real flanks AND the shuffled baseline, recompute real/shuffled ratio
under matched external-flank conditions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, "/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer")

from preprocess.alignment import dot_plot, windowed_matches, encode_dna


def _site_hit_positions(nc: str, flank: str, L: int, m_thresh: int) -> set[int]:
    fwd_dot, _ = dot_plot(nc, flank)
    win = windowed_matches(fwd_dot, L)
    if win.size == 0: return set()
    per_nc_max = win.max(axis=1)
    return set(int(i) for i in np.where(per_nc_max >= m_thresh)[0])


def _dinuc_shuffle(seq: str, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    if len(seq) < 2: return seq
    adj = defaultdict(list)
    for i in range(len(seq) - 1):
        adj[seq[i]].append(seq[i + 1])
    for lst in adj.values(): rng.shuffle(lst)
    out = [seq[0]]
    while len(out) < len(seq):
        last = out[-1]
        if not adj[last]:
            remaining = [c for c in adj if adj[c]]
            if not remaining: break
            out.append(remaining[0])
        else:
            out.append(adj[last].pop())
    return "".join(out) if len(out) == len(seq) else seq


# ---------------- W10' ---------------------------------------------------

def w10p_v42_guide_consistency(pos_v42_path: str, max_tnps: int = 5000):
    """For each V4.2 Tnp with >=5 sites, check whether planted guide_dna and
    guide_span (normalized nc_start) are consistent across sites."""
    print(f"\n=== W10' :: V4.2 per-Tnp guide consistency across sites ===")
    tnp_records = defaultdict(list)
    n_seen = 0
    with open(pos_v42_path) as f:
        for line in f:
            r = json.loads(line)
            L = r["labels"]
            tnp = r["transposase_id"]
            active_nc = L.get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if active_nc >= len(ncs): active_nc = 0
            nc = ncs[active_nc]
            gspan = L.get("guide_span_in_active_noncoding")
            if gspan is None: continue
            tnp_records[tnp].append({
                "guide_dna":  L.get("guide_dna"),
                "guide_L":    L.get("guide_length"),
                "nc_start":   gspan[0],
                "nc_len":     len(nc),
                "nc_start_norm": gspan[0] / max(1, len(nc)),
                "nc":         nc,
            })
            n_seen += 1
            if len(tnp_records) >= max_tnps and tnp not in tnp_records: break

    multi = {t: recs for t, recs in tnp_records.items() if len(recs) >= 5}
    print(f"  V4.2 Tnps with >=5 sites: {len(multi)} (out of {len(tnp_records)} sampled)")

    if not multi:
        print(f"  no multi-site Tnps in sample — cannot test consistency")
        return {"n_multi": 0}

    # (a) guide_dna consistency
    n_guide_identical = 0; n_guide_near_identical = 0
    guide_seq_spread = []
    # (b) normalized nc_start spread
    nc_start_spread = []; nc_start_norm_spread = []
    for tnp, recs in multi.items():
        guides = [r["guide_dna"] for r in recs[:5]]
        # exact identity
        if len(set(guides)) == 1: n_guide_identical += 1
        # near-identical: same length + hamming <= 2
        if len(set(len(g) for g in guides)) == 1:
            g0 = guides[0]
            L = len(g0)
            max_ham = max(sum(1 for i in range(L) if g0[i] != g[i]) for g in guides)
            if max_ham <= 2: n_guide_near_identical += 1
        starts = [r["nc_start"] for r in recs[:5]]
        starts_norm = [r["nc_start_norm"] for r in recs[:5]]
        nc_start_spread.append(max(starts) - min(starts))
        nc_start_norm_spread.append(max(starts_norm) - min(starts_norm))

    print(f"  planted guide sequence IDENTICAL across all 5 sites: {n_guide_identical}/{len(multi)} ({n_guide_identical/len(multi):.2%})")
    print(f"  planted guide NEAR-identical (≤2 hamming): {n_guide_near_identical}/{len(multi)} ({n_guide_near_identical/len(multi):.2%})")
    print(f"  absolute nc_start spread: median {int(np.median(nc_start_spread))}   mean {np.mean(nc_start_spread):.1f}   max {max(nc_start_spread)}")
    print(f"  normalized nc_start spread: median {np.median(nc_start_norm_spread):.3f}   mean {np.mean(nc_start_norm_spread):.3f}")

    print(f"\n  VERDICT:")
    if n_guide_identical / len(multi) > 0.8:
        print(f"    V4.2 has planted-guide-sequence identity across sites — coherence exists via sequence alignment.")
        print(f"    Channel B can train on V4.2 WITHOUT regeneration, using sequence-alignment-based coordinates.")
        print(f"    ncRNA-differences-per-site are just structural background, not a blocker.")
    elif np.median(nc_start_norm_spread) < 0.05:
        print(f"    V4.2 has consistent normalized nc position — partial coherence via normalization.")
    else:
        print(f"    V4.2 has NO consistent planted guide across sites of a Tnp.")
        print(f"    Coherence truly absent from training data. Regeneration required.")
    return {
        "n_multi": len(multi),
        "frac_guide_identical": n_guide_identical / len(multi),
        "frac_guide_near_identical": n_guide_near_identical / len(multi),
        "median_nc_start_spread": int(np.median(nc_start_spread)),
        "median_nc_start_norm_spread": float(np.median(nc_start_norm_spread)),
    }


# ---------------- V1'' ---------------------------------------------------

def v1pp_dual_null(cog_path, gold_path, L: int, m: int, n_perm: int = 50, seed: int = 0):
    """Dual-null test. Real: 5 unshuffled flanks. Shuffled-external: 5
    dinuc-shuffled flanks from any source. Real-external: 5 dinuc-shuffled of
    the same Tnp's own flanks. Report all three."""
    print(f"\n=== V1'' :: dual-null cross-family estimate ===")
    gold = {json.loads(l)["site_id"]: json.loads(l) for l in open(gold_path)}
    tnp_sites = defaultdict(list); tnp_nc = {}
    with open(cog_path) as f:
        for line in f:
            r = json.loads(line)
            if gold.get(r["site_id"]) is None: continue
            a = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if a >= len(ncs): a = 0
            nc = ncs[a]; tnp = r["transposase_id"]
            if tnp not in tnp_nc: tnp_nc[tnp] = nc
            elif tnp_nc[tnp] != nc: continue
            tnp_sites[tnp].append({"flank": r["inputs"]["flank"]})
    rng = np.random.default_rng(seed)
    all_flanks = [(t, s["flank"]) for t, ss in tnp_sites.items() for s in ss]

    def _coh_S5(hits):
        c = Counter()
        for h in hits: c.update(h)
        return {p for p, cc in c.items() if cc >= 5}

    # Real (original real flanks, same-Tnp)
    real_S5 = []
    real_ext_S5 = []       # 5 real flanks BUT dinuc-shuffled → "real Tnp but external quality"
    shuf_int_S5 = []       # 5 random real flanks (original V1.d shuffled)
    shuf_ext_S5 = []       # 5 dinuc-shuffled random flanks → matched-external baseline

    for tnp in list(tnp_sites.keys()):
        sites = tnp_sites[tnp]
        if len(sites) < 5: continue
        nc = tnp_nc[tnp]
        real_flanks = [s["flank"] for s in sites[:5]]
        # (a) real
        real_hits = [_site_hit_positions(nc, fl, L, m) for fl in real_flanks]
        real_S5.append(len(_coh_S5(real_hits)))
        # (b) real Tnp's flanks but dinuc-shuffled — "would real Tnp still detect if flanks were external-quality?"
        ext_flanks = [_dinuc_shuffle(fl, seed=seed + hash(fl) % (2**31)) for fl in real_flanks]
        ext_hits = [_site_hit_positions(nc, fl, L, m) for fl in ext_flanks]
        real_ext_S5.append(len(_coh_S5(ext_hits)))
        # (c) shuffled internal — original V1.d shuffled
        for _ in range(n_perm):
            idx = rng.choice(len(all_flanks), size=5, replace=False)
            shuf_flanks = [all_flanks[int(i)][1] for i in idx]
            shuf_hits = [_site_hit_positions(nc, fl, L, m) for fl in shuf_flanks]
            shuf_int_S5.append(len(_coh_S5(shuf_hits)))
        # (d) shuffled + dinuc-shuffled — matched external-flank null
        for _ in range(n_perm):
            idx = rng.choice(len(all_flanks), size=5, replace=False)
            shuf_flanks = [_dinuc_shuffle(all_flanks[int(i)][1], seed=seed + hash(all_flanks[int(i)][1]) % (2**31))
                              for i in idx]
            shuf_hits = [_site_hit_positions(nc, fl, L, m) for fl in shuf_flanks]
            shuf_ext_S5.append(len(_coh_S5(shuf_hits)))

    r = np.mean(real_S5); rx = np.mean(real_ext_S5)
    si = np.mean(shuf_int_S5); sx = np.mean(shuf_ext_S5)
    print(f"  n_tnps evaluated = {len(real_S5)}")
    print(f"  (a) REAL real-flanks:                     S=5 mean per Tnp = {r:.4f}")
    print(f"  (b) REAL dinuc-shuffled own flanks:       S=5 mean per Tnp = {rx:.4f}")
    print(f"  (c) SHUFFLED internal Durrant flanks:     S=5 mean per Tnp = {si:.4f}")
    print(f"  (d) SHUFFLED dinuc-shuffled flanks:       S=5 mean per Tnp = {sx:.4f}")

    print(f"\n  Ratios (real / null), same-rule pairs:")
    print(f"    original (V1.d): a / c = {r/max(1e-9,si):.2f}×")
    print(f"    matched external: b / d = {rx/max(1e-9,sx):.2f}×")
    print(f"    real real vs external null: a / d = {r/max(1e-9,sx):.2f}×  (over-optimistic; two-sided)")

    print(f"\n  Cross-family estimate:")
    print(f"    If a novel family behaves like external flanks in both signal AND null:")
    print(f"    b / d = {rx/max(1e-9,sx):.2f}× is the proper dual-null number.")
    print(f"    Compare to Durrant intra-family a / c = {r/max(1e-9,si):.2f}×.")
    print(f"    Degradation ratio: {(r/si) / max(1e-9, rx/sx):.2f}×  (this is the honest one-sided-corrected number)")

    return {"real": float(r), "real_dinuc_own": float(rx),
              "shuf_internal": float(si), "shuf_dinuc": float(sx),
              "ratio_original": float(r / max(1e-9, si)),
              "ratio_dual_null": float(rx / max(1e-9, sx)),
              "degradation_factor": float((r / si) / max(1e-9, rx / sx))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durrant-cog", required=True)
    ap.add_argument("--durrant-gold", required=True)
    ap.add_argument("--v42-pos", required=True)
    ap.add_argument("--L", type=int, default=11)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r_w10p = w10p_v42_guide_consistency(args.v42_pos)
    r_v1pp = v1pp_dual_null(args.durrant_cog, args.durrant_gold, args.L, args.m)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"W10_prime": r_w10p, "V1_prime_prime": r_v1pp}, f, indent=2)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
