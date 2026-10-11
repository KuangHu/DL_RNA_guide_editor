"""Model-free scorer v2 — fixes three correctness issues in v1.

Fixes (see critique 2026-10-10):

1. RC frame conversion.  v1 did `a = np.where(m_f >= m_r, a_f, a_r)` — mixing
   fwd-flank coord and rc-flank coord on the same axis.  v2 converts
       a_r_in_fwd = flank_len - a_r - L
   using the identity from preprocess/alignment.py:228 before orient-collapse.
   position_consistency and the target-substring extraction are then on a
   single consistent axis.

2. Candidate-level complexity gate.  Low-complexity guides (ATATAT, poly-A)
   won v1 selection by (m_consistency=1 × pos_consist=1 × high flank_diversity).
   v2 multiplies a guide complexity factor into the per-candidate score at
   Step 2 so these peaks lose selection rather than merely being penalised
   post-hoc.
     complexity_g(seq) = (H_base/2) * min(H_dinuc/2.5, 1) * max(0, 1 - run/L)
   caps:
     - ATATATATAT:  0.5 * 0.4 * 1    = 0.20
     - ACGTACGTACGT: 1 * 0.8 * 1     = 0.80
     - AAAAAAAAAA:  0   = 0.00
     - random:      ~0.9

3. Multi-hit penalty.  Low-complexity targets produce many flank offsets
   that tie m_max; v1 took the first argmax silently.  v2, at the chosen
   (top_p, top_L), recounts per-site hit multiplicity (flank offsets whose
   m >= m_max_at_top) and multiplies by 1/sqrt(mean_multiplicity).

Also:
  - WOBBLE×2 bonus is REMOVED.  Kept as a descriptive stat only.
  - Target complexity reported post-hoc (not in selection) — low-target
    cases are usually caught by multiplicity anyway.

Output schema is a superset of v1 so results are directly joinable on bag_id.
"""
from __future__ import annotations
import sys, csv, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.channel_b.data import ChannelBDataset
from model.channel_b.constants import Ls

ROOT = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/"
            "fna_ins_discovery_v5_md_seed2")
OUT = ROOT / "modelfree_scores_v2.tsv"

_COMPL = {"A":"T","T":"A","C":"G","G":"C","N":"N"}
def revcomp(s: str) -> str:
    return "".join(_COMPL.get(b, "N") for b in reversed(s))


# ---------- complexity ----------
def shannon_base(s: str) -> float:
    s = "".join(b for b in s if b in "ACGT")
    if not s: return 0.0
    n = len(s)
    H = 0.0
    for b in "ACGT":
        p = s.count(b) / n
        if p > 0: H -= p * math.log2(p)
    return H

def shannon_dinuc(s: str) -> float:
    dinucs = [s[i:i+2] for i in range(len(s)-1) if s[i] in "ACGT" and s[i+1] in "ACGT"]
    if len(dinucs) < 2: return 0.0
    n = len(dinucs)
    from collections import Counter
    c = Counter(dinucs)
    H = 0.0
    for v in c.values():
        p = v / n
        H -= p * math.log2(p)
    return H

def longest_run(s: str) -> int:
    best = cur = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            cur += 1; best = max(best, cur)
        else:
            cur = 1
    return best if s else 0

def complexity(seq: str) -> dict:
    """Return a dict with H_base, H_dinuc, longest_run, and composite factor in [0,1]."""
    if not seq: return dict(H_base=0, H_dinuc=0, run=0, c=0.0)
    H  = shannon_base(seq)
    Hd = shannon_dinuc(seq)
    r  = longest_run(seq)
    c1 = H / 2.0                       # 0..1 (2 bits = uniform ACGT)
    c2 = min(Hd / 2.5, 1.0)            # 0..1 (saturate at 2.5 bits)
    c3 = max(0.0, 1.0 - r / len(seq))  # 0..1 (run spanning whole seq → 0)
    return dict(H_base=H, H_dinuc=Hd, run=r, c=c1 * c2 * c3)


# ---------- flank utilities ----------
def pairwise_identity(seqs: list[str], exclude_window: tuple[int, int]) -> float:
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


def count_multiplicity(nc: str, flank: str, p: int, L: int,
                        m_target: int, orient: str) -> int:
    """Count how many L-windows in the right flank frame achieve m >= m_target.

    - orient='fwd': scan `flank`
    - orient='rc' : scan `revcomp(flank)` (that's the frame match_table used)

    The nc guide used is nc[p:p+L].  m is the per-base match count.
    """
    guide = nc[p:p+L] if p + L <= len(nc) else ""
    if not guide or len(guide) < L: return 0
    target = revcomp(flank) if orient == "rc" else flank
    if len(target) < L: return 0
    n = 0
    for j in range(len(target) - L + 1):
        m = 0
        for k in range(L):
            if guide[k] == target[j+k]: m += 1
        if m >= m_target: n += 1
    return n


# ---------- scoring ----------
def score_bag(ds: ChannelBDataset, bag_id: str) -> dict:
    sites = ds._sites_by_bag.get(bag_id, [])
    K = len(sites)
    if K < 2: return {"score": 0.0, "reason": "K<2"}

    # Build nc_concat per site (same convention as v1)
    def nc_concat(s):
        nc_list = s["inputs"]["noncoding_regions"]
        nc = nc_list[0]
        for extra in nc_list[1:]:
            nc = nc + "N"*13 + extra
        return nc
    nc_per_site = [nc_concat(s) for s in sites]
    flank_per_site = [s["inputs"]["flank"] for s in sites]
    flank_lens = [len(f) for f in flank_per_site]

    best = {"score": -1, "top_p": -1, "top_L": -1}
    for L in Ls:
        m_arrs = []; a_arrs = []; orient_arrs = []
        valid_idx = []
        for s in range(K):
            try:
                ma_f = ds._mt.get(bag_id, s, "fwd", L)
                ma_r = ds._mt.get(bag_id, s, "rc",  L)
                m_f = ma_f.m_max_by_excl.get(0)
                m_r = ma_r.m_max_by_excl.get(0)
                a_f = ma_f.flank_argmax_by_excl.get(0)
                a_r = ma_r.flank_argmax_by_excl.get(0)
                if m_f is None or m_r is None:
                    m_arrs.append(None); continue
                n_pos = min(len(m_f), len(m_r))
                m_f = m_f[:n_pos]; m_r = m_r[:n_pos]
                a_f = a_f[:n_pos]; a_r = a_r[:n_pos]
                # FIX 1: convert rc-flank coord to fwd-flank coord
                flen = flank_lens[s]
                a_r_in_fwd = flen - a_r - L
                # orient-collapse on a single flank axis
                pick_fwd = m_f >= m_r
                m = np.where(pick_fwd, m_f, m_r)
                a = np.where(pick_fwd, a_f, a_r_in_fwd)
                orient = np.where(pick_fwd, 0, 1)  # 0=fwd, 1=rc
                m_arrs.append(m); a_arrs.append(a); orient_arrs.append(orient)
                valid_idx.append(s)
            except Exception:
                m_arrs.append(None)
        if len(valid_idx) < 2: continue
        # trim to common length
        n_pos_min = min(len(m_arrs[s]) for s in valid_idx)
        if n_pos_min == 0: continue
        M = np.vstack([m_arrs[s][:n_pos_min] for s in valid_idx])
        A = np.vstack([a_arrs[s][:n_pos_min] for s in valid_idx])

        mean_m = M.mean(axis=0) / L
        std_a  = A.std(axis=0)
        pos_consist = np.exp(-std_a / 5.0)

        # FIX 2: candidate-level complexity gate on guide = nc_per_site[valid[0]][p:p+L]
        # The nc may differ across sites in v5 reconstruction, but the guide at nc_pos p
        # is the proposed site-shared signal.  Use nc_per_site[valid[0]] as the reference
        # for selection; final per-site guide complexity is computed again post-hoc.
        nc_ref = nc_per_site[valid_idx[0]]
        comp_g_arr = np.zeros(n_pos_min, dtype=np.float32)
        for p in range(n_pos_min):
            if p + L > len(nc_ref):
                comp_g_arr[p] = 0.0
            else:
                comp_g_arr[p] = complexity(nc_ref[p:p+L])["c"]

        combined = mean_m * pos_consist * comp_g_arr

        p_best = int(combined.argmax())
        s_best = float(combined[p_best])
        if s_best > best["score"]:
            best = dict(score=s_best, top_p=p_best, top_L=L,
                        m_at_top=[int(m_arrs[s][p_best]) for s in valid_idx],
                        a_at_top=[int(a_arrs[s][p_best]) for s in valid_idx],
                        orient_at_top=[int(orient_arrs[si][p_best])
                                       for si, s in enumerate(valid_idx)],
                        K_valid=len(valid_idx),
                        valid_idx=list(valid_idx))

    if best["top_p"] < 0:
        return {"score": 0.0, "reason": "no_valid_L"}

    top_p, top_L = best["top_p"], best["top_L"]
    valid_idx = best["valid_idx"]
    m_vals = best["m_at_top"]
    a_vals = best["a_at_top"]
    orient_vals = best["orient_at_top"]
    K_valid = best["K_valid"]

    # Final components at (top_p, top_L)
    m_consistency = float(np.mean(m_vals) / top_L)
    std_a = float(np.std(a_vals))
    position_consistency = float(np.exp(-std_a / 5.0))

    nc_ref = nc_per_site[valid_idx[0]]
    guide = nc_ref[top_p:top_p + top_L] if top_p + top_L <= len(nc_ref) else ""
    comp_g = complexity(guide)

    # per-site target substrings (now on the FIXED fwd-flank axis)
    targets = []
    for i_s, s in enumerate(valid_idx):
        f = flank_per_site[s]
        a = a_vals[i_s]
        if orient_vals[i_s] == 1:
            # window on original flank runs flank[a:a+L]; the matched
            # content is revcomp(nc_ref[top_p:top_p+top_L]) i.e. the RC guide.
            if 0 <= a and a + top_L <= len(f):
                targets.append(f[a:a+top_L])
            else:
                targets.append("")
        else:
            if 0 <= a and a + top_L <= len(f):
                targets.append(f[a:a+top_L])
            else:
                targets.append("")

    # target complexity (descriptive, not in selection)
    comp_t_vals = [complexity(t)["c"] for t in targets if t]
    comp_t_mean = float(np.mean(comp_t_vals)) if comp_t_vals else 0.0

    unique_targets = len({t for t in targets if t})
    unique_guides = 1  # guide at (top_p, top_L) is a single sequence from nc_ref
    # However the per-site nc may differ (v5 reconstruction) — check it:
    per_site_guides = []
    for s in valid_idx:
        nc = nc_per_site[s]
        if top_p + top_L <= len(nc):
            per_site_guides.append(nc[top_p:top_p+top_L])
    unique_guides = len({g for g in per_site_guides if g})

    # FIX 3: hit multiplicity per site at (top_p, top_L)
    multi = []
    for i_s, s in enumerate(valid_idx):
        nc = nc_per_site[s]
        f  = flank_per_site[s]
        orient = "rc" if orient_vals[i_s] == 1 else "fwd"
        n = count_multiplicity(nc, f, top_p, top_L, m_vals[i_s], orient)
        multi.append(max(n, 1))
    mean_multi = float(np.mean(multi))
    multiplicity_penalty = 1.0 / math.sqrt(mean_multi)  # 1 hit → 1.0, 4 hits → 0.5

    # Flank diversity on fwd-flank axis
    if len(flank_per_site) >= 2 and K_valid >= 2:
        valid_flanks = [flank_per_site[s] for s in valid_idx]
        a_min = max(0, min(a_vals))
        a_max = min(min(flank_lens), max(a_vals) + top_L)
        flank_diversity = 1.0 - pairwise_identity(valid_flanks, (a_min, a_max))
    else:
        flank_diversity = 0.0

    # Wobble: descriptive only, NO score multiplier
    wobble_flag = int(std_a < 5 and unique_targets > K_valid / 2)

    bag_score = (m_consistency *
                 position_consistency *
                 comp_g["c"] *
                 flank_diversity *
                 multiplicity_penalty)

    return dict(
        bag_id=bag_id,
        score=bag_score,
        K_valid=K_valid,
        top_p=top_p, top_L=top_L,
        m_mean=m_consistency,
        pos_consist=position_consistency,
        std_argmax=std_a,
        complexity_guide=comp_g["c"],
        H_base_guide=comp_g["H_base"],
        H_dinuc_guide=comp_g["H_dinuc"],
        longest_run_guide=comp_g["run"],
        complexity_target_mean=comp_t_mean,
        mean_hit_multiplicity=mean_multi,
        multiplicity_penalty=multiplicity_penalty,
        flank_diversity=flank_diversity,
        wobble_flag=wobble_flag,
        unique_guides=unique_guides,
        unique_targets=unique_targets,
        guide_at_top=guide,
    )


def main():
    ds = ChannelBDataset(ROOT/"adapted.jsonl", ROOT/"adapted_shard",
                         preload_mt=True, cache_dir=None, sort_by_nc_len=False)
    print(f"# scoring {len(ds)} bags with v2 (RC fix + complexity + multiplicity)")
    results = []
    for i in range(len(ds)):
        bid = ds._bag_index[i][0]
        try:
            r = score_bag(ds, bid)
        except Exception as e:
            r = {"bag_id": bid, "score": 0.0, "reason": f"ERR:{type(e).__name__}:{e}"[:200]}
        if "bag_id" not in r: r["bag_id"] = bid
        results.append(r)
        if (i+1) % 500 == 0:
            print(f"  [{i+1}/{len(ds)}]", flush=True)

    cols = ["bag_id","score","K_valid","top_p","top_L",
            "m_mean","pos_consist","std_argmax",
            "complexity_guide","H_base_guide","H_dinuc_guide","longest_run_guide",
            "complexity_target_mean",
            "mean_hit_multiplicity","multiplicity_penalty",
            "flank_diversity","wobble_flag","unique_guides","unique_targets",
            "guide_at_top"]
    with OUT.open("w") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=cols)
        w.writeheader()
        for r in sorted(results, key=lambda x: -x.get("score", 0)):
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"# wrote {OUT}")

    vals = [r.get("score",0) for r in results]
    print(f"\n# distribution: min={min(vals):.3f}  p50={np.median(vals):.3f}  "
          f"p95={np.percentile(vals,95):.3f}  max={max(vals):.3f}")

    top = sorted(results, key=lambda x: -x.get("score",0))[:30]
    print(f"\n## top-30\n")
    print(f"  {'rk':>3s}  {'bag':<10s}  {'score':>6s}  K  tp   L  mmean  pcon   cg "
          f" cT  mult  div  ug ut  guide_at_top")
    for i, r in enumerate(top, 1):
        print(f"  {i:>3d}  {r['bag_id']:<10s}  {r['score']:>6.3f}  "
              f"{r.get('K_valid',0):>2d}  {r.get('top_p',0):>4d} "
              f"{r.get('top_L',0):>2d}  "
              f"{r.get('m_mean',0):>5.2f}  "
              f"{r.get('pos_consist',0):>4.2f}  "
              f"{r.get('complexity_guide',0):>4.2f}  "
              f"{r.get('complexity_target_mean',0):>4.2f}  "
              f"{r.get('mean_hit_multiplicity',0):>4.1f}  "
              f"{r.get('flank_diversity',0):>4.2f}  "
              f"{r.get('unique_guides',0):>2d} "
              f"{r.get('unique_targets',0):>2d}  "
              f"{r.get('guide_at_top','')!r}")


if __name__ == "__main__":
    main()
