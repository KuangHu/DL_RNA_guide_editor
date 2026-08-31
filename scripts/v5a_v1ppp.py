"""V1''' — proper cross-family construction.

Preserve TBL-flank targeting; shuffle only surrounding background sequence.
For each real Durrant record:
  - Keep the annotated TBL span [target_flank_start, target_flank_start + L] EXACTLY as in real flank.
  - Dinucleotide-shuffle EVERYTHING OUTSIDE the TBL span.
  - This preserves the biological guide-target match (mechanism 1) while
    removing family-specific background sequence structure that carried +0.55.

Compare S=5 detection under three constructions:
  (a) REAL — real flanks unchanged (V1.d/V1'' 'a' group).
  (a') SAME rule but with SW-preserved TBL and shuffled background.
       Preserves targeting_intact=True, changes null_model to "shuffled_background".
       This IS a proper varying_dim="null_model" test.
  (d)  SHUFFLED Tnp, dinuc-shuffled flanks (V1'' 'd' group), for reference.

Also uses v5a_eval_asserts to demonstrate the assert_same_rule discipline:
before reporting a/a' ratio, verify only null_model differs.
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

from preprocess.alignment import dot_plot, windowed_matches
from v5a_eval_asserts import Metric, MetricCondition, safe_ratio


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


def _preserve_tbl_shuffle_background(flank: str, tbl_start: int, tbl_len: int, seed: int) -> str:
    """Keep flank[tbl_start:tbl_start+tbl_len] intact; dinuc-shuffle the rest,
    then splice back."""
    if tbl_start < 0 or tbl_start + tbl_len > len(flank):
        return flank    # safe no-op
    before = flank[:tbl_start]
    tbl    = flank[tbl_start:tbl_start + tbl_len]
    after  = flank[tbl_start + tbl_len:]
    before_sh = _dinuc_shuffle(before, seed=seed) if len(before) >= 2 else before
    after_sh  = _dinuc_shuffle(after,  seed=seed + 1) if len(after)  >= 2 else after
    return before_sh + tbl + after_sh


def v1ppp(cog_path, gold_path, L: int, m: int, seed: int = 0):
    """Proper cross-family construction.
    For each real Tnp, the 5 real flanks each get: preserve their annotated TBL span,
    dinuc-shuffle the rest. Then compute S=5 coherence with these background-shuffled
    flanks. Compare to the original real S=5 rate.
    """
    print(f"\n=== V1''' :: preserve TBL targeting, shuffle background ===")
    gold = {json.loads(l)["site_id"]: json.loads(l) for l in open(gold_path)}
    tnp_sites = defaultdict(list); tnp_nc = {}
    with open(cog_path) as f:
        for line in f:
            r = json.loads(line)
            g = gold.get(r["site_id"])
            if g is None: continue
            a = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if a >= len(ncs): a = 0
            nc = ncs[a]; tnp = r["transposase_id"]
            if tnp not in tnp_nc: tnp_nc[tnp] = nc
            elif tnp_nc[tnp] != nc: continue
            tnp_sites[tnp].append({"flank": r["inputs"]["flank"],
                                       "tbl_start": g["target_flank_start"],
                                       "tbl_L":     g["target_binding_loop_length"]})

    def _coh_S5(hits):
        c = Counter()
        for h in hits: c.update(h)
        return {p for p, cc in c.items() if cc >= 5}

    real_S5 = []; bgshuf_S5 = []
    for tnp, sites in tnp_sites.items():
        if len(sites) < 5: continue
        nc = tnp_nc[tnp]
        # (a) real
        real_hits = [_site_hit_positions(nc, s["flank"], L, m) for s in sites[:5]]
        real_S5.append(len(_coh_S5(real_hits)))
        # (a') TBL-preserved, background-shuffled
        bg_flanks = [_preserve_tbl_shuffle_background(s["flank"], s["tbl_start"], s["tbl_L"],
                                                            seed=seed + hash(s["flank"]) % (2**31))
                        for s in sites[:5]]
        bg_hits = [_site_hit_positions(nc, fl, L, m) for fl in bg_flanks]
        bgshuf_S5.append(len(_coh_S5(bg_hits)))

    r_mean = float(np.mean(real_S5)); bg_mean = float(np.mean(bgshuf_S5))
    print(f"  n_tnps evaluated = {len(real_S5)}")
    print(f"  (a)  REAL flanks:                                         S=5 mean = {r_mean:.4f}")
    print(f"  (a') REAL flanks with TBL preserved + background shuffled: S=5 mean = {bg_mean:.4f}")

    # Now use assert_same_rule as demonstration
    a_metric = Metric("V1'''(a) REAL", r_mean, MetricCondition(
        match_rule="strict_WC", null_model="unshuffled_family_background",
        coordinate_system="absolute_nc", targeting_intact=True,
        tie_break="average_rank", denominator="tnp"))
    ap_metric = Metric("V1'''(a') TBL_preserved_bg_shuffled", bg_mean, MetricCondition(
        match_rule="strict_WC", null_model="shuffled_family_background",
        coordinate_system="absolute_nc", targeting_intact=True,   # ← preserved
        tie_break="average_rank", denominator="tnp"))
    # This IS a valid varying_dim="null_model" test.
    ratio = safe_ratio(a_metric, ap_metric, varying_dim="null_model")
    print(f"\n  Ratio real / TBL-preserved-bg-shuffled = {ratio:.2f}×")
    print(f"  (targeting_intact=True on both sides — proper varying_dim = null_model)")

    if bg_mean > 0.7 * r_mean:
        print(f"\n  → Family-specific background contributes ≤ 30% to Channel A detection.")
        print(f"    Cross-family degradation on flank background alone is modest.")
    elif bg_mean > 0.3 * r_mean:
        print(f"\n  → Family-specific background contributes 30-70%.")
        print(f"    Cross-family degradation is meaningful but Channel A still detects.")
    else:
        print(f"\n  → Family-specific background is MOST of the signal.")
        print(f"    Cross-family degradation is severe. Channel A may not transfer without V2.")

    return {"real_S5_mean": r_mean, "bgshuffled_S5_mean": bg_mean, "ratio": ratio}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durrant-cog", required=True)
    ap.add_argument("--durrant-gold", required=True)
    ap.add_argument("--L", type=int, default=11)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r = v1ppp(args.durrant_cog, args.durrant_gold, args.L, args.m)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"V1_prime_prime_prime": r}, f, indent=2)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
