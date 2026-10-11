"""Model-free scorer that replicates the v84 fake-positive construction rule
as a direct statistical scorer. No neural net.

Rule recap:
  Flank = [genome_bg][target_motif][genome_bg]  (120bp)
  NC    = [nc_bg][structure][guide_motif][structure][nc_bg]

Positive-bag criteria:
  A. Per-site target lands at CONSISTENT POSITION in flank (across sites)
  B. Per-site structure region (nc around guide) has FOLD CONSISTENCY
  C. Genome backgrounds (flank OUTSIDE target zone) are DIVERSE (not clonal)
  D. WOBBLE BONUS: even if guide/target sequences differ, if positions cluster, +points

Scoring (per bag):
  1. For each (nc_pos p, length L), use match_table to get per-site (m_max, flank_argmax)
  2. Pick (top_p, top_L) that maximizes:  mean(m_max/L) * exp(-std(argmax)/5)
  3. At (top_p, top_L):
     - m_consistency = mean(m_max/L)
     - position_consistency = exp(-std(flank_argmax)/5)
     - wobble_bonus = 1 if sequences differ but positions still cluster
     - flank_diversity = 1 - mean pairwise identity of flanks OUTSIDE target zone
  4. bag_score = m_consistency * position_consistency * flank_diversity * (1 + wobble_bonus)

No ViennaRNA fold yet (optional add — structure consistency component commented out
for speed; can be added as a weighted multiplier).
"""
from __future__ import annotations
import sys, csv
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.channel_b.data import ChannelBDataset
from model.channel_b.constants import Ls

ROOT = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/"
                   "fna_ins_discovery_v5_md_seed2")
OUT = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/"
                 "fna_ins_discovery_v5_md_seed2/modelfree_scores.tsv")


def pairwise_identity(seqs: list[str], exclude_window: tuple[int, int]) -> float:
    """Mean pairwise identity across seqs, excluding exclude_window."""
    if len(seqs) < 2: return 0.0
    lo, hi = exclude_window
    masked = [s[:lo] + s[hi:] for s in seqs]
    n = 0; sim = 0.0
    for i in range(len(masked)):
        for j in range(i+1, len(masked)):
            a, b = masked[i], masked[j]
            L = min(len(a), len(b))
            if L == 0: continue
            m = sum(x == y for x, y in zip(a[:L], b[:L]))
            sim += m / L; n += 1
    return sim / n if n > 0 else 0.0


def score_bag(ds: ChannelBDataset, bag_id: str) -> dict:
    sites = ds._sites_by_bag.get(bag_id, [])
    K = len(sites)
    if K < 2: return {"score": 0.0, "reason": "K<2"}

    # Gather m_max and flank_argmax at every (p, L) across sites
    # shape: (L, nc_pos) → per-site lists
    best = {"score": -1, "top_p": -1, "top_L": -1}
    for L in Ls:
        m_arrs = []; a_arrs = []
        for s in range(K):
            try:
                ma_f = ds._mt.get(bag_id, s, "fwd", L)
                ma_r = ds._mt.get(bag_id, s, "rc",  L)
                m_f = ma_f.m_max_by_excl.get(0)
                m_r = ma_r.m_max_by_excl.get(0)
                a_f = ma_f.flank_argmax_by_excl.get(0)
                a_r = ma_r.flank_argmax_by_excl.get(0)
                if m_f is None or m_r is None:
                    m_arrs.append(None); a_arrs.append(None); continue
                n_pos = min(len(m_f), len(m_r))
                m = np.maximum(m_f[:n_pos], m_r[:n_pos])
                a = np.where(m_f[:n_pos] >= m_r[:n_pos], a_f[:n_pos], a_r[:n_pos])
                m_arrs.append(m); a_arrs.append(a)
            except Exception:
                m_arrs.append(None); a_arrs.append(None)
        valid = [i for i, m in enumerate(m_arrs) if m is not None]
        if len(valid) < 2: continue
        n_pos_min = min(len(m_arrs[i]) for i in valid)
        if n_pos_min == 0: continue
        # Stack: (K_valid, n_pos)
        M = np.vstack([m_arrs[i][:n_pos_min] for i in valid])
        A = np.vstack([a_arrs[i][:n_pos_min] for i in valid])
        # Score per nc position: mean(m/L) * exp(-std(a)/5)
        mean_m = M.mean(axis=0) / L   # normalized 0..1
        std_a = A.std(axis=0)
        pos_consist = np.exp(-std_a / 5.0)
        combined = mean_m * pos_consist
        p_best = int(combined.argmax())
        s_best = float(combined[p_best])
        if s_best > best["score"]:
            best = {"score": s_best, "top_p": p_best, "top_L": L,
                        "m_at_top": [int(m_arrs[i][p_best]) for i in valid],
                        "a_at_top": [int(a_arrs[i][p_best]) for i in valid],
                        "K_valid": len(valid), "valid_idx": valid}

    if best["top_p"] < 0: return {"score": 0.0, "reason": "no_valid_L"}

    # Compute additional components at best (top_p, top_L)
    top_p, top_L = best["top_p"], best["top_L"]
    m_vals = best["m_at_top"]
    a_vals = best["a_at_top"]
    valid_idx = best["valid_idx"]
    m_consistency = np.mean(m_vals) / top_L
    std_a = float(np.std(a_vals))
    position_consistency = float(np.exp(-std_a / 5.0))

    # Target sequences at per-site flank positions
    flanks = [sites[i]["inputs"]["flank"] for i in valid_idx]
    targets = []
    for i, a in zip(range(len(valid_idx)), a_vals):
        f = flanks[i]
        if 0 <= a and a + top_L <= len(f):
            targets.append(f[a:a+top_L])
        else:
            targets.append("")
    unique_targets = len(set(t for t in targets if t))

    # Guide sequences at nc[top_p..top_p+top_L]
    guides = []
    for i in valid_idx:
        s = sites[i]
        nc_list = s["inputs"]["noncoding_regions"]
        nc = nc_list[0]
        if len(nc_list) > 1: nc += "N"*13 + nc_list[1]
        if len(nc) >= top_p + top_L:
            guides.append(nc[top_p:top_p+top_L])
        else:
            guides.append("")
    unique_guides = len(set(g for g in guides if g))

    # Wobble bonus: positions cluster but sequences differ
    # If std_a < 5 AND unique_targets > K_valid/2 → wobble bonus = 1
    K_valid = best["K_valid"]
    wobble_bonus = 1.0 if (std_a < 5 and unique_targets > K_valid / 2) else 0.0

    # Flank diversity OUTSIDE target zone
    # Mask flank[a:a+L] for each site; compute mean pairwise identity on rest
    if len(flanks) >= 2:
        # use common window [min(a), max(a)+L] as the excluded zone
        a_min = max(0, min(a_vals))
        a_max = min(120, max(a_vals) + top_L)
        flank_diversity = 1.0 - pairwise_identity(flanks, (a_min, a_max))
    else:
        flank_diversity = 0.0

    # Final combined score
    bag_score = m_consistency * position_consistency * flank_diversity * (1.0 + wobble_bonus)

    return {
        "bag_id": bag_id,
        "score": bag_score,
        "K_valid": K_valid,
        "top_p": top_p, "top_L": top_L,
        "m_mean": m_consistency,
        "pos_consist": position_consistency,
        "std_argmax": std_a,
        "flank_diversity": flank_diversity,
        "wobble_bonus": wobble_bonus,
        "unique_guides": unique_guides,
        "unique_targets": unique_targets,
    }


def main():
    ds = ChannelBDataset(ROOT/"adapted.jsonl", ROOT/"adapted_shard",
                                preload_mt=True, cache_dir=None, sort_by_nc_len=False)
    print(f"# scoring {len(ds)} bags")
    results = []
    for i in range(len(ds)):
        bid = ds._bag_index[i][0]
        try:
            r = score_bag(ds, bid)
        except Exception as e:
            r = {"bag_id": bid, "score": 0.0, "reason": f"ERR:{type(e).__name__}"}
        if "bag_id" not in r: r["bag_id"] = bid
        results.append(r)
        if (i+1) % 500 == 0:
            print(f"  [{i+1}/{len(ds)}]", flush=True)

    with OUT.open("w") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=[
            "bag_id","score","K_valid","top_p","top_L","m_mean","pos_consist",
            "std_argmax","flank_diversity","wobble_bonus","unique_guides","unique_targets"])
        w.writeheader()
        for r in sorted(results, key=lambda x: -x.get("score",0)):
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print(f"# wrote {OUT}")

    # Show top 30
    vals = [r.get("score",0) for r in results]
    print(f"\n# distribution: min={min(vals):.3f} p50={np.median(vals):.3f} "
              f"p95={np.percentile(vals,95):.3f} max={max(vals):.3f}\n")
    top = sorted(results, key=lambda x: -x.get("score",0))[:30]
    print(f"  {'rank':>4s}  {'bag':<10s}  {'score':>6s}  {'K':>2s}  {'top_p':>5s}  "
              f"{'L':>2s}  {'mmean':>5s}  {'posc':>5s}  {'div':>4s}  {'wb':>2s}  "
              f"{'ug':>2s}  {'ut':>2s}")
    for i, r in enumerate(top, 1):
        print(f"  {i:>3d}   {r['bag_id']:<10s}  {r['score']:>6.3f}  "
                  f"{r.get('K_valid',0):>2d}  {r.get('top_p',0):>5d}  "
                  f"{r.get('top_L',0):>2d}  {r.get('m_mean',0):>5.2f}  "
                  f"{r.get('pos_consist',0):>5.2f}  {r.get('flank_diversity',0):>4.2f}  "
                  f"{int(r.get('wobble_bonus',0)):>2d}  "
                  f"{r.get('unique_guides',0):>2d}  {r.get('unique_targets',0):>2d}")


if __name__ == "__main__":
    main()
