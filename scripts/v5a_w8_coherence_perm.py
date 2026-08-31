"""W8 — coherence permutation test.

Question: is cross-site NC-position coherence (measured by W7) a real biological
signal, or an artifact of Durrant's 5 sites sharing the same ncRNA?

Test design (avoids all analytic-null pitfalls):
  For each nc window (nc_start, L) and threshold m, precompute per site whether
  that site's flank produces a matching window at ≥ m matches at that (nc_start, L).
  Then for each Tnp T (real or fake) with S sites, count "coherent positions" =
  positions where at least S_thresh of the S sites have a hit.

  Real:      genuine Tnp groupings (5 sites per Tnp for Durrant).
  Shuffled:  keep each ncRNA fixed; permute the assignment of flanks to Tnps.
             For fake Tnp T', its 5 sites use the same ncRNA T but flanks from
             5 randomly chosen Durrant records.

  Compare coherence counts. If shuffled ≈ real, W7 was circular (per-position
  significance recomputed for a shared sequence). If real ≫ shuffled, cross-site
  coherence is a real signal and its gain = (real / shuffled) at each S is the
  expected improvement of a nc-window-first joint-significance proposer over a
  per-site proposer.

Also reports:
  - Analytic prediction under Bernoulli independence: p_hit^S, expected coherent
    positions per Tnp.
  - Gold-position rank under coherence score in real vs shuffled: does the
    annotated TBL nc position rise to the top under joint significance?
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
from scipy.stats import binom

sys.path.insert(0, "/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer")
sys.path.insert(0, "/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/scripts")

from preprocess.alignment import dot_plot, windowed_matches


# For W8 we only look at forward-orient windows (as the user's L=11 analytic).
def _site_hits(nc: str, flank: str, L: int, m_thresh: int) -> set[int]:
    """Return set of nc_start positions where the fwd L-window has >= m_thresh matches.

    Uses windowed_matches on the fwd dot plot; picks nc_start such that
    max over flank positions is >= m_thresh.
    """
    fwd_dot, _ = dot_plot(nc, flank)
    win = windowed_matches(fwd_dot, L)   # (nc_len - L + 1, flank_len - L + 1)
    per_nc_max = win.max(axis=1) if win.size > 0 else np.array([], dtype=np.int32)
    return set(int(i) for i in np.where(per_nc_max >= m_thresh)[0])


def w8_coherence(cog_path, gold_path, L: int = 11, m_thresh: int = 8,
                    n_perm: int = 50, seed: int = 0):
    print(f"\n=== W8 :: coherence permutation (L={L}, m≥{m_thresh}) ===")
    gold = {json.loads(l)["site_id"]: json.loads(l) for l in open(gold_path)}
    # Group by tnp: list of (site_id, flank, gold_nc_start)
    tnp_sites: dict[str, list[dict]] = defaultdict(list)
    tnp_nc: dict[str, str] = {}
    with open(cog_path) as f:
        for line in f:
            r = json.loads(line)
            g = gold.get(r["site_id"])
            if g is None: continue
            a = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if a >= len(ncs): a = 0
            nc = ncs[a]
            tnp = r["transposase_id"]
            if tnp in tnp_nc:
                if tnp_nc[tnp] != nc:
                    # sanity: same tnp should share nc; if not, skip this record
                    continue
            else:
                tnp_nc[tnp] = nc
            tnp_sites[tnp].append({
                "site_id":  r["site_id"],
                "flank":    r["inputs"]["flank"],
                "gold_nc":  g["guide_start_in_nc"],
            })
    print(f"  n_tnps = {len(tnp_sites)}   (per-Tnp site counts: {Counter(len(v) for v in tnp_sites.values()).most_common()})")

    # Precompute per-flank hits for each nc window. Because ncRNA is shared per
    # Tnp, we precompute hits per (tnp, flank_index).
    print(f"  precomputing per-site hit sets at (L={L}, m≥{m_thresh}) ...", flush=True)
    per_site_hits: dict[tuple[str, int], set[int]] = {}
    for tnp, sites in tnp_sites.items():
        nc = tnp_nc[tnp]
        for i, s in enumerate(sites):
            per_site_hits[(tnp, i)] = _site_hits(nc, s["flank"], L, m_thresh)

    # ---- Real grouping ----
    def _coherent_positions(hits_lists: list[set[int]], S_thresh: int) -> set[int]:
        if not hits_lists: return set()
        # Count how many sites contain each nc position; keep positions with count >= S_thresh.
        counts = Counter()
        for h in hits_lists: counts.update(h)
        return {p for p, c in counts.items() if c >= S_thresh}

    real_counts = {S: [] for S in (1, 2, 3, 4, 5)}
    # Also: does the gold nc position appear as a coherent position?
    gold_coherent_at_S = {S: [] for S in (2, 3, 4, 5)}
    for tnp, sites in tnp_sites.items():
        if len(sites) < 2: continue
        hits = [per_site_hits[(tnp, i)] for i in range(len(sites))]
        for S in (1, 2, 3, 4, 5):
            if len(sites) < S: continue
            n_coh = len(_coherent_positions(hits, S))
            real_counts[S].append(n_coh)
        gold_ncs = set(s["gold_nc"] for s in sites)
        for S in (2, 3, 4, 5):
            if len(sites) < S: continue
            coh = _coherent_positions(hits, S)
            # gold is coherent iff at least one gold nc_start ∈ coh
            gold_coherent_at_S[S].append(int(len(gold_ncs & coh) > 0))

    # ---- Shuffled grouping ----
    # For each real Tnp T with |sites|=k, form n_perm fake Tnps: keep nc=nc_T,
    # but draw flank hit-sets from RANDOM other Tnps' sites.
    rng = np.random.default_rng(seed)
    all_site_keys = list(per_site_hits.keys())
    tnp_of_key = {k: k[0] for k in all_site_keys}
    # For shuffling, we need to pick flanks that were computed on the SAME ncRNA
    # (so hits are comparable). But flanks are hit-sets already computed on
    # each Tnp's ncRNA — we can't move a flank to another Tnp's ncRNA without
    # recomputation. So for the shuffle we do:
    #   Fake Tnp with nc=nc_T: pick k random sites from ALL sites, use their
    #   flank *sequences* (from `tnp_sites[.]`), compute NEW hit-sets on nc_T.
    # This is expensive; do a smaller n_perm and cache within same fake-tnp.
    print(f"  running {n_perm} shuffled permutations ...", flush=True)
    shuffled_counts = {S: [] for S in (1, 2, 3, 4, 5)}
    for perm in range(n_perm):
        for tnp, sites in tnp_sites.items():
            if len(sites) < 2: continue
            nc = tnp_nc[tnp]
            k = len(sites)
            # Draw k random site indices from all sites (any Tnp)
            idx = rng.choice(len(all_site_keys), size=k, replace=False)
            fake_flanks = [tnp_sites[all_site_keys[int(i)][0]][all_site_keys[int(i)][1]]["flank"]
                             for i in idx]
            fake_hits = [_site_hits(nc, fl, L, m_thresh) for fl in fake_flanks]
            for S in (1, 2, 3, 4, 5):
                if k < S: continue
                shuffled_counts[S].append(len(_coherent_positions(fake_hits, S)))

    print(f"\n  Coherent positions per Tnp:")
    print(f"  {'S_thresh':>8} {'real_median':>12} {'real_mean':>12} {'shuffled_median':>16} {'shuffled_mean':>15} {'ratio(mean)':>13}")
    for S in (1, 2, 3, 4, 5):
        rm = np.mean(real_counts[S]) if real_counts[S] else 0
        rmed = np.median(real_counts[S]) if real_counts[S] else 0
        sm = np.mean(shuffled_counts[S]) if shuffled_counts[S] else 1e-9
        smed = np.median(shuffled_counts[S]) if shuffled_counts[S] else 0
        ratio = rm / max(1e-9, sm)
        print(f"  S={S:>4}      {rmed:>12.1f} {rm:>12.2f} {smed:>16.1f} {sm:>15.2f} {ratio:>13.2f}")

    # Gold coherence
    print(f"\n  Fraction of Tnps where the annotated gold nc position is a coherent position:")
    for S in (2, 3, 4, 5):
        vals = gold_coherent_at_S[S]
        if not vals: continue
        print(f"    S≥{S}:  {np.mean(vals):.3%}   n={len(vals)}")

    # Analytic prediction
    print(f"\n  Analytic prediction under Bernoulli independence (p=0.25, L={L}, m≥{m_thresh}):")
    p_hit = float(1.0 - binom.cdf(m_thresh - 1, L, 0.25))
    # crude: single-nc probability of a hit at some flank = 1 - (1-p_hit)^flank_len (~110)
    # instead use windowed_matches-based expectation approximated as p_hit * flank_windows.
    for S in (1, 2, 3, 4, 5):
        # E[coherent positions | S sites] ≈ 167 * (p_effective)^S
        # p_effective is the per-site probability of getting a hit at a given nc_start
        # For per_nc_max >= m_thresh across ~110 flank windows: p_eff ≈ 1 - (1 - p_hit)^110
        p_eff = float(1.0 - (1.0 - p_hit) ** 110)
        E = 167 * (p_eff ** S)
        print(f"    S={S}: p_eff={p_eff:.4f}  E[coherent nc positions] = {E:.4f}")

    print(f"\n  VERDICT:")
    print(f"    If real / shuffled ratio at S=5 is 100×+ → cross-site is a real signal, build v3 (reversed loop).")
    print(f"    If ratio < 10× or shuffled ≈ real → W7 was largely circular; per-site is what remains.")

    return {"real_counts_summary": {f"S{S}": {"median": float(np.median(real_counts[S])) if real_counts[S] else 0,
                                                    "mean":   float(np.mean(real_counts[S])) if real_counts[S] else 0}
                                          for S in (1, 2, 3, 4, 5)},
              "shuffled_counts_summary": {f"S{S}": {"median": float(np.median(shuffled_counts[S])) if shuffled_counts[S] else 0,
                                                       "mean":   float(np.mean(shuffled_counts[S])) if shuffled_counts[S] else 0}
                                              for S in (1, 2, 3, 4, 5)},
              "gold_coherent_frac": {f"S{S}": (float(np.mean(v)) if v else 0)
                                          for S, v in gold_coherent_at_S.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durrant-cog", required=True)
    ap.add_argument("--durrant-gold", required=True)
    ap.add_argument("--L", type=int, default=11)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r = w8_coherence(args.durrant_cog, args.durrant_gold, L=args.L, m_thresh=args.m, n_perm=args.n_perm)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"W8": r, "params": {"L": args.L, "m_thresh": args.m, "n_perm": args.n_perm}}, f, indent=2)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
