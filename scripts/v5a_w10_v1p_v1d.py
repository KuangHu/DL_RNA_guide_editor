"""W10 + V1' + V1.d.

W10: Compute Durrant-style coherence permutation on V4.2 mining data.
     For each V4.2 Tnp with >=5 site records, compute S distribution + shuffled
     baseline + real/shuffled ratio. Report vs. Durrant's numbers.

V1': Rerun V1.b using truly external flanks — dinucleotide-preserving shuffle
     of Durrant flanks (approximates non-IS110 sequence structure while
     preserving local composition). Compare mean_m at the 22 S=5 positions.

V1.d: Partition W9-style shuffle draws by whether they contained same-Tnp
      contamination. Report S=5 hit rate CONDITIONAL on contamination present
      vs absent.
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
sys.path.insert(0, "/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/scripts")

from preprocess.alignment import dot_plot, windowed_matches, encode_dna


def _site_hit_positions(nc: str, flank: str, L: int, m_thresh: int) -> set[int]:
    fwd_dot, _ = dot_plot(nc, flank)
    win = windowed_matches(fwd_dot, L)
    if win.size == 0: return set()
    per_nc_max = win.max(axis=1)
    return set(int(i) for i in np.where(per_nc_max >= m_thresh)[0])


def _max_matches_at_nc(nc: str, flank: str, nc_start: int, L: int) -> int:
    fwd_dot, _ = dot_plot(nc, flank)
    win = windowed_matches(fwd_dot, L)
    if win.size == 0 or nc_start >= win.shape[0]: return 0
    return int(win[nc_start].max())


def _dinuc_shuffle(seq: str, seed: int = 0) -> str:
    """Dinucleotide-preserving shuffle: preserves each dinucleotide's frequency."""
    rng = np.random.default_rng(seed)
    if len(seq) < 2: return seq
    # Build dinuc adjacency list
    adj = defaultdict(list)
    for i in range(len(seq) - 1):
        adj[seq[i]].append(seq[i + 1])
    # Random walk from start
    for lst in adj.values(): rng.shuffle(lst)
    out = [seq[0]]
    while len(out) < len(seq):
        last = out[-1]
        if not adj[last]:
            # Restart from any remaining char
            remaining = [c for c in adj if adj[c]]
            if not remaining: break
            out.append(remaining[0])
        else:
            out.append(adj[last].pop())
    return "".join(out) if len(out) == len(seq) else seq


# ------------ W10 --------------------------------------------------------

def w10_v42_distribution(pos_v42_path: str, L: int, m: int, max_tnps: int = 200,
                            n_perm: int = 20, seed: int = 0):
    """V4.2 coherence permutation. Load positive records grouped by tnp_id."""
    print(f"\n=== W10 :: V4.2 S distribution vs Durrant ===")
    tnp_sites = defaultdict(list)
    with open(pos_v42_path) as f:
        for line in f:
            r = json.loads(line)
            a = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if a >= len(ncs): a = 0
            tnp = r["transposase_id"]
            if len(tnp_sites[tnp]) >= 10: continue    # cap
            tnp_sites[tnp].append({"nc": ncs[a], "flank": r["inputs"]["flank"]})
            if len(tnp_sites) >= max_tnps and tnp not in tnp_sites: break

    # V4.2: does ncRNA vary across sites of a Tnp?
    n_multi = 0; nc_varies = 0
    for tnp, sites in tnp_sites.items():
        if len(sites) >= 5:
            n_multi += 1
            if len(set(s["nc"] for s in sites)) > 1: nc_varies += 1
    print(f"  V4.2 Tnps with >=5 sites (in {len(tnp_sites)} sampled): {n_multi}")
    print(f"  ...of which ncRNA VARIES across sites: {nc_varies} ({nc_varies/max(1,n_multi):.2%})")
    if nc_varies > 0:
        print(f"  → V4.2 does NOT share ncRNA across sites of a Tnp — the shared-ncRNA assumption")
        print(f"    that makes Channel A work on Durrant does NOT hold in V4.2 training data.")
        print(f"    Consequence: cannot train Channel B on V4.2 without regenerating with shared ncRNA per Tnp.")

    # Compute S distribution on V4.2 (real vs shuffled)
    rng = np.random.default_rng(seed)
    real_counts = {S: [] for S in (1, 2, 3, 4, 5)}
    shuf_counts = {S: [] for S in (1, 2, 3, 4, 5)}
    all_flanks = [(t, s["flank"]) for t, ss in tnp_sites.items() for s in ss]

    def _coh(hits, S_thresh):
        c = Counter()
        for h in hits: c.update(h)
        return {p for p, cc in c.items() if cc >= S_thresh}

    # Sample subset of Tnps to keep this fast
    tnps_eval = [t for t, s in tnp_sites.items() if len(s) >= 5][:100]
    for tnp in tnps_eval:
        sites = tnp_sites[tnp]
        # For V4.2 with varying nc, use the first site's nc as reference for shuffled comparison
        # For real, each site uses its own nc → compute per-site hits and take union
        # But if nc varies, per-site hit sets are on different coordinate spaces — can't union directly
        if len(set(s["nc"] for s in sites)) > 1:
            # Can't compute coherence when ncRNAs differ across sites.
            # Skip; report separately.
            continue
        nc = sites[0]["nc"]
        hits = [_site_hit_positions(nc, s["flank"], L, m) for s in sites[:5]]
        for S in (1, 2, 3, 4, 5):
            real_counts[S].append(len(_coh(hits, S)))
        # Shuffled: draw 5 random flanks
        for _ in range(n_perm):
            idx = rng.choice(len(all_flanks), size=5, replace=False)
            fake_flanks = [all_flanks[int(i)][1] for i in idx]
            fake_hits = [_site_hit_positions(nc, fl, L, m) for fl in fake_flanks]
            for S in (1, 2, 3, 4, 5):
                shuf_counts[S].append(len(_coh(fake_hits, S)))

    print(f"\n  V4.2 coherence (only Tnps with shared ncRNA; n_evaluable = {len(real_counts[1])}):")
    print(f"  {'S':>4} {'real_median':>12} {'real_mean':>10} {'shuf_mean':>10} {'ratio':>8}")
    for S in (1, 2, 3, 4, 5):
        if not real_counts[S]: continue
        rm = np.mean(real_counts[S]); sm = np.mean(shuf_counts[S]) or 1e-9
        rmed = int(np.median(real_counts[S]))
        print(f"  S={S:>2}      {rmed:>12} {rm:>10.2f} {sm:>10.2f} {rm/sm:>8.2f}")
    print(f"\n  Durrant reference (from W8): S=5 real 0.35, shuffled 0.02, ratio 16.9×")
    print(f"  → If V4.2 real >> Durrant real, planted coherence is inflated; Channel B model will overtrain.")
    print(f"  → If V4.2 nc VARIES across sites (line above), Channel A's mechanism is not exposed in training.")

    return {"n_v42_tnps_multi":     n_multi,
              "n_v42_tnps_nc_varies": nc_varies,
              "n_evaluable_shared_nc": len(real_counts[1])}


# ------------ V1' --------------------------------------------------------

def v1p_external_flanks(cog_path, gold_path, L: int, m: int, n_external: int = 50, seed: int = 0):
    """V1.b rerun with truly external flanks (dinucleotide-shuffled Durrant flanks)."""
    print(f"\n=== V1' :: promiscuity of S=5 positions with EXTERNAL flanks ===")
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
            tnp_sites[tnp].append({"flank": r["inputs"]["flank"],
                                       "gold_nc": gold[r["site_id"]]["guide_start_in_nc"]})

    # Identify S=5 positions
    per_tnp_S5 = {}
    for tnp, sites in tnp_sites.items():
        if len(sites) < 5: continue
        hits = [_site_hit_positions(tnp_nc[tnp], s["flank"], L, m) for s in sites]
        counts = Counter()
        for h in hits: counts.update(h)
        S5 = [p for p, c in counts.items() if c >= 5]
        if S5: per_tnp_S5[tnp] = S5

    # Build EXTERNAL flanks: dinucleotide-preserving shuffles of ALL Durrant flanks
    rng = np.random.default_rng(seed)
    all_flanks = [s["flank"] for sites in tnp_sites.values() for s in sites]
    ext_flanks = [_dinuc_shuffle(all_flanks[i % len(all_flanks)], seed=seed + i)
                   for i in range(n_external)]

    S5_mean_m_ext = []; S5_mean_m_int = []
    for tnp, positions in per_tnp_S5.items():
        nc = tnp_nc[tnp]
        # Internal Durrant flanks (excluding this Tnp) for comparison
        other_int = [s["flank"] for tt, sites in tnp_sites.items() if tt != tnp for s in sites]
        idx_int = rng.choice(len(other_int), size=n_external, replace=False)
        int_flanks = [other_int[int(i)] for i in idx_int]
        for pos in positions:
            m_ext = np.mean([_max_matches_at_nc(nc, fl, pos, L) for fl in ext_flanks])
            m_int = np.mean([_max_matches_at_nc(nc, fl, pos, L) for fl in int_flanks])
            S5_mean_m_ext.append(m_ext); S5_mean_m_int.append(m_int)
    print(f"  S=5 positions ({len(S5_mean_m_ext)}):")
    print(f"    mean_m from INTERNAL Durrant flanks (V1.b baseline):  {np.mean(S5_mean_m_int):.2f}")
    print(f"    mean_m from EXTERNAL (dinuc-shuffled) flanks:          {np.mean(S5_mean_m_ext):.2f}")
    print(f"    control random ncRNA positions (V1.b baseline): 6.66")
    print(f"    Δ from control: internal {np.mean(S5_mean_m_int) - 6.66:+.2f}   external {np.mean(S5_mean_m_ext) - 6.66:+.2f}")
    if abs(np.mean(S5_mean_m_ext) - 6.66) < 0.3:
        print(f"  → EXTERNAL flanks return to control level. The +0.59 is IS110-family shared bias.")
        print(f"    Cross-family Channel A will lose most of the +0.6 boost. V2 background null helps.")
    else:
        print(f"  → EXTERNAL flanks still elevated. Effect is broader than family-shared bias.")
    return {"S5_mean_m_internal": float(np.mean(S5_mean_m_int)),
              "S5_mean_m_external": float(np.mean(S5_mean_m_ext))}


# ------------ V1.d -------------------------------------------------------

def v1d_contamination_stratification(cog_path, gold_path, L: int, m: int, n_perm: int = 500,
                                        seed: int = 0):
    """W9-style shuffle; partition draws by same-Tnp contamination presence."""
    print(f"\n=== V1.d :: contamination stratification of shuffled hits ===")
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
    n_all = len(all_flanks)

    n_contam_S5 = 0; n_clean_S5 = 0
    n_contam_draws = 0; n_clean_draws = 0

    def _coh(hits, S_thresh):
        c = Counter()
        for h in hits: c.update(h)
        return {p for p, cc in c.items() if cc >= S_thresh}

    for tnp in list(tnp_sites.keys())[:65]:
        if len(tnp_sites[tnp]) < 5: continue
        nc = tnp_nc[tnp]
        for _ in range(n_perm):
            idx = rng.choice(n_all, size=5, replace=False)
            has_contam = any(all_flanks[int(i)][0] == tnp for i in idx)
            fake_flanks = [all_flanks[int(i)][1] for i in idx]
            fake_hits = [_site_hit_positions(nc, fl, L, m) for fl in fake_flanks]
            n_S5 = len(_coh(fake_hits, 5))
            if has_contam:
                n_contam_S5 += n_S5; n_contam_draws += 1
            else:
                n_clean_S5 += n_S5; n_clean_draws += 1
    rate_contam = n_contam_S5 / max(1, n_contam_draws)
    rate_clean = n_clean_S5 / max(1, n_clean_draws)
    print(f"  contaminated draws (≥1 same-Tnp flank of 5): {n_contam_draws}   S=5 rate = {rate_contam:.4f}")
    print(f"  clean draws (no same-Tnp flank):              {n_clean_draws}   S=5 rate = {rate_clean:.4f}")
    print(f"  ratio contam/clean = {rate_contam/max(1e-9, rate_clean):.2f}")
    if rate_contam > 5 * rate_clean:
        print(f"  → Shuffled 0.021 baseline is INFLATED by contamination. Real/shuffled 16.9× is UNDERESTIMATE.")
        print(f"    Clean-only shuffled S=5 rate = {rate_clean:.4f}, corrected real/shuffled = {0.35 / max(1e-9, rate_clean):.1f}×")
    else:
        print(f"  → Contamination alone does not explain shuffled hits. 0.021 baseline stands.")
    return {"contam_S5_rate": float(rate_contam),
              "clean_S5_rate": float(rate_clean),
              "ratio":          float(rate_contam / max(1e-9, rate_clean))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durrant-cog", required=True)
    ap.add_argument("--durrant-gold", required=True)
    ap.add_argument("--v42-pos", required=True)
    ap.add_argument("--L", type=int, default=11)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r_w10 = w10_v42_distribution(args.v42_pos, args.L, args.m)
    r_v1p = v1p_external_flanks(args.durrant_cog, args.durrant_gold, args.L, args.m)
    r_v1d = v1d_contamination_stratification(args.durrant_cog, args.durrant_gold, args.L, args.m)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"W10": r_w10, "V1_prime": r_v1p, "V1d": r_v1d}, f, indent=2)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
