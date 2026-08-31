"""V5A Gate B: statistical cross-site hypothesis test.

No training. Measures whether the cross-site coherence signal exists in
train / val / Durrant.

Metrics:
  (1) Records-per-Tnp distribution (multi-site prevalence).
  (2) Same-Tnp gold-gold |Δ nc_start| distribution vs random-pair null.
  (3) Same-Tnp gold-gold k-mer composition similarity (Bhattacharyya-like)
      vs random-pair null.

Go / No-Go rule:
  A. If < 40% of Durrant Tnps have >= 2 sites, cross-site is INERT on the
     acceptance benchmark → NO-GO on current 3b design.
  B. If gold-gold effect size on |Δnc_start| and/or k-mer similarity is
     < 0.5 (Cohen's d), the yield ceiling of 3b is limited → SCOPE-LIMITED.
  C. Otherwise, GO.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np


K = 3   # k-mer size for local composition


def _kmer_hist(s: str, k: int = K) -> np.ndarray:
    """4^k-dim histogram of k-mer counts in s."""
    n = 4 ** k
    h = np.zeros(n, dtype=np.float32)
    lookup = {"A": 0, "C": 1, "G": 2, "T": 3}
    s = s.upper().replace("U", "T")
    if len(s) < k:
        return h
    for i in range(len(s) - k + 1):
        idx = 0; ok = True
        for j in range(k):
            b = lookup.get(s[i + j])
            if b is None: ok = False; break
            idx = idx * 4 + b
        if ok: h[idx] += 1
    tot = h.sum()
    if tot > 0: h /= tot
    return h


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    if n == 0: return 0.0
    return float(np.dot(a, b) / n)


def _effect_size(a: np.ndarray, b: np.ndarray) -> dict:
    if len(a) == 0 or len(b) == 0: return {"n_a": len(a), "n_b": len(b), "d": float("nan")}
    ma, mb = float(np.mean(a)), float(np.mean(b))
    va, vb = float(np.var(a)), float(np.var(b))
    sp = (va + vb) / 2.0
    d = (ma - mb) / max(1e-8, sp ** 0.5)
    return {
        "n_a":       int(len(a)),
        "n_b":       int(len(b)),
        "mean_a":    ma,
        "mean_b":    mb,
        "median_a":  float(np.median(a)),
        "median_b":  float(np.median(b)),
        "d_cohen":   d,
    }


def collect_v42(pos_jsonl: str, keep_tnps: set | None = None, cap_per_tnp: int = 10) -> dict:
    per_tnp = defaultdict(list)     # tnp_id -> list of (nc_start, kmer_hist)
    with open(pos_jsonl) as f:
        for line in f:
            r = json.loads(line)
            tnp = r["transposase_id"]
            if keep_tnps is not None and tnp not in keep_tnps: continue
            if len(per_tnp[tnp]) >= cap_per_tnp: continue
            L = r["labels"]
            gspan = L.get("guide_span_in_active_noncoding")
            if gspan is None: continue
            active_nc = L.get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if active_nc >= len(ncs): active_nc = 0
            nc = ncs[active_nc]
            g_L = int(L.get("guide_length", 12))
            w = 20
            lo = max(0, gspan[0] - w); hi = min(len(nc), gspan[0] + g_L + w)
            window = nc[lo:hi]
            per_tnp[tnp].append({
                "site_id":  r["site_id"],
                "nc_start": int(gspan[0]),
                "kmer":     _kmer_hist(window),   # cache the histogram once
            })
    return per_tnp


def collect_durrant(cog_path: str, gold_path: str) -> dict:
    gold = {}
    with open(gold_path) as f:
        for line in f:
            g = json.loads(line); gold[g["site_id"]] = g
    per_tnp = defaultdict(list)
    with open(cog_path) as f:
        for line in f:
            r = json.loads(line)
            sid = r["site_id"]
            g = gold.get(sid)
            if g is None: continue
            active_nc = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if active_nc >= len(ncs): active_nc = 0
            nc = ncs[active_nc]
            nc_start = int(g["guide_start_in_nc"])
            g_L = int(g["target_binding_loop_length"])
            w = 20
            lo = max(0, nc_start - w); hi = min(len(nc), nc_start + g_L + w)
            window = nc[lo:hi]
            per_tnp[r["transposase_id"]].append({
                "site_id":  sid,
                "nc_start": nc_start,
                "kmer":     _kmer_hist(window),   # cache the histogram once
            })
    return per_tnp


def analyze(per_tnp: dict, name: str, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    records_per_tnp = [len(v) for v in per_tnp.values()]
    n_tnps = len(per_tnp)
    multi = sum(1 for n in records_per_tnp if n >= 2)
    multi_frac = multi / max(1, n_tnps)

    # Gold-gold |Δnc_start| within-Tnp pairs (cached kmer already).
    gg_delta = []
    gg_sim = []
    for tnp, recs in per_tnp.items():
        if len(recs) < 2: continue
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                dnc = abs(recs[i]["nc_start"] - recs[j]["nc_start"])
                gg_delta.append(dnc)
                gg_sim.append(_cosine(recs[i]["kmer"], recs[j]["kmer"]))

    # Cross-Tnp null: pair random records from DIFFERENT tnps.
    all_recs = [(t, r) for t, rs in per_tnp.items() for r in rs]
    null_delta = []
    null_sim = []
    if len(all_recs) >= 2:
        n_samples = min(len(gg_delta) if gg_delta else 5000, 20000)
        for _ in range(n_samples):
            i, j = rng.choice(len(all_recs), size=2, replace=False)
            if all_recs[i][0] == all_recs[j][0]: continue
            r1 = all_recs[i][1]; r2 = all_recs[j][1]
            null_delta.append(abs(r1["nc_start"] - r2["nc_start"]))
            null_sim.append(_cosine(r1["kmer"], r2["kmer"]))

    delta_stats = _effect_size(np.asarray(gg_delta), np.asarray(null_delta))
    sim_stats   = _effect_size(np.asarray(gg_sim),   np.asarray(null_sim))

    print(f"\n=== Gate B :: {name} ===")
    print(f"  n_tnps={n_tnps}  records={sum(records_per_tnp)}  multi-site fraction={multi_frac:.3f} (n_multi={multi})")
    print(f"  records-per-Tnp median={int(np.median(records_per_tnp))}  "
          f"p25={int(np.percentile(records_per_tnp,25))} p75={int(np.percentile(records_per_tnp,75))} "
          f"max={max(records_per_tnp) if records_per_tnp else 0}")
    if gg_delta:
        print(f"  |Δ nc_start| gold-gold vs null:")
        print(f"    gold-gold n={delta_stats['n_a']} median={delta_stats['median_a']:.1f} mean={delta_stats['mean_a']:.1f}")
        print(f"    null      n={delta_stats['n_b']} median={delta_stats['median_b']:.1f} mean={delta_stats['mean_b']:.1f}")
        print(f"    d_cohen={delta_stats['d_cohen']:+.3f}   (negative = gold-gold is CLOSER, which is the hypothesis)")
    if gg_sim:
        print(f"  k=3 window k-mer cosine similarity gold-gold vs null:")
        print(f"    gold-gold n={sim_stats['n_a']} median={sim_stats['median_a']:.3f} mean={sim_stats['mean_a']:.3f}")
        print(f"    null      n={sim_stats['n_b']} median={sim_stats['median_b']:.3f} mean={sim_stats['mean_b']:.3f}")
        print(f"    d_cohen={sim_stats['d_cohen']:+.3f}   (positive = gold-gold is MORE similar, which is the hypothesis)")

    return {
        "n_tnps":         n_tnps,
        "n_records":      sum(records_per_tnp),
        "multi_frac":     multi_frac,
        "n_multi_tnps":   multi,
        "delta_stats":    delta_stats,
        "sim_stats":      sim_stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v42-pos", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--durrant-cog", required=True)
    ap.add_argument("--durrant-gold", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    splits = json.load(open(args.splits))

    # Subsample V4.2 tnps aggressively — distribution test needs O(few hundred) tnps, not 5000.
    train_keep = set(splits["train"][:500])   # 500 train tnps × up to 10 records = 5k
    val_keep   = set(splits["val"][:200])

    print(f"[collect] V4.2 train ({len(train_keep)} tnps, cap 10/tnp)", flush=True)
    train_v42 = collect_v42(args.v42_pos, keep_tnps=train_keep, cap_per_tnp=10)
    print(f"[collect] V4.2 val   ({len(val_keep)} tnps, cap 10/tnp)", flush=True)
    val_v42   = collect_v42(args.v42_pos, keep_tnps=val_keep,   cap_per_tnp=10)

    print("[collect] Durrant cognate + gold")
    dur = collect_durrant(args.durrant_cog, args.durrant_gold)

    r_train = analyze(train_v42, "V4.2 train")
    r_val   = analyze(val_v42,   "V4.2 val")
    r_dur   = analyze(dur,       "Durrant")

    # Go/No-Go rule
    print("\n=== Gate B verdict ===")
    multi = r_dur["multi_frac"]
    d_delta = r_dur["delta_stats"].get("d_cohen", float("nan"))
    d_sim   = r_dur["sim_stats"].get("d_cohen", float("nan"))
    print(f"  Durrant multi-site fraction: {multi:.3f}")
    print(f"  Durrant gold-gold |Δnc| effect size (Cohen d): {d_delta:+.3f}")
    print(f"  Durrant gold-gold k-mer similarity effect size (Cohen d): {d_sim:+.3f}")
    if multi < 0.40:
        verdict = "NO-GO  (Durrant multi-site fraction < 0.40; 3b cross-site is inert on the acceptance benchmark)"
    elif abs(d_delta) < 0.5 and abs(d_sim) < 0.5:
        verdict = "SCOPE-LIMITED  (Durrant gold-gold effect sizes below 0.5; 3b yield ceiling is limited)"
    else:
        verdict = "GO"
    print(f"  → {verdict}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"train_v42": r_train, "val_v42": r_val, "durrant": r_dur,
                     "verdict": verdict}, f, indent=2)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
