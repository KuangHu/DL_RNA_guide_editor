"""V5A amendments A1 + A2 + A3.

A1: length_pen (α=0.5, L0=12) on frozen Durrant — the actual Gate A bar for real data.
A2: reweighted val TAXONOMY profile audit — check the reweight matches Durrant on
    per-decoy-bucket P(c*>d), not just aggregate MRR.
A3: weight distribution report (formalized). Already have from Gate 0 but restated here.

Tnp-clustered paired-bootstrap CIs on Δ MRR between raw_m, length_pen (and later,
selectors) — the correct unit for statistical inference on Durrant given 5 sites/Tnp.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, "/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer")

from preprocess.candidates import build_candidate_arrays, DEFAULT_L_MIN, DEFAULT_L_MAX


def _overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def _find_gold(feats, mask, cands, orient, L, nc, fl, of=0.5):
    valid = np.where(mask)[0]
    if len(valid) == 0: return -1, 0.0
    matches = feats[:, 3]
    best = -1; best_m = -1.0
    for i in valid:
        c = cands[int(i)]
        if c.orient != orient: continue
        mn = min(c.L, L)
        nc_ov = _overlap(c.nc_start, c.nc_start + c.L, nc, nc + L)
        f_ov = _overlap(c.flank_start, c.flank_start + c.L, fl, fl + L)
        if nc_ov < of*mn or f_ov < of*mn: continue
        if matches[i] > best_m: best_m = float(matches[i]); best = int(i)
    return best, best_m


def _classify(c, orient, L, nc_start, flank_start, of=0.5):
    if c.orient != orient: return "wrong_orientation"
    mn = min(c.L, L)
    nc_ov = _overlap(c.nc_start, c.nc_start + c.L, nc_start, nc_start + L)
    f_ov = _overlap(c.flank_start, c.flank_start + c.L, flank_start, flank_start + L)
    th = of * mn
    if nc_ov < th: return "different_region"
    dL = c.L - L
    if dL > 0: return "same_region_longer_L"
    if dL < 0: return "same_region_shorter_L"
    if f_ov < th: return "same_region_same_L_wrong_flank"
    return "near_gold"


def _rank_stats(qs, cs_local, k_list=(1,4,8)):
    q_cs = qs[cs_local]
    other = np.delete(qs, cs_local)
    n_gt = int((other > q_cs).sum()); n_eq = int((other == q_cs).sum())
    tie = n_eq + 1
    rank_avg = n_gt + 1 + n_eq / 2.0
    R = {k: (0.0 if n_gt >= k else min(1.0, (k - n_gt) / tie)) for k in k_list}
    E_recip = float(np.mean(1.0 / (n_gt + 1 + np.arange(tie, dtype=np.float64))))
    return rank_avg, R, E_recip


def score(slot, method, coefs=None):
    m = float(slot["matches"]); L = int(slot["L"])
    if method == "raw_m":    return m
    if method == "identity": return m / max(1, L)
    if method == "length_pen":
        a, L0 = coefs
        return m - a * max(0.0, L - L0)
    raise ValueError(method)


def _bootstrap_delta_clustered(cluster_ids, a, b, w=None, n_boot=5000, seed=0):
    """Paired bootstrap on Δ mean(a) - mean(b), clustered by cluster_ids.
    Resample clusters with replacement; within each cluster include all rows;
    optional per-row weights w."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if w is None: w = np.ones_like(a)
    else: w = np.asarray(w, dtype=np.float64)
    cluster_ids = np.asarray(cluster_ids)
    uniq = np.unique(cluster_ids)
    idx_by = {c: np.where(cluster_ids == c)[0] for c in uniq}
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        picks = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by[c] for c in picks])
        ww = w[rows]
        num = ((a[rows] - b[rows]) * ww).sum()
        den = ww.sum() or 1.0
        deltas[i] = num / den
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def run_durrant(cog_path: str, gold_path: str):
    """A1: raw_m and length_pen on Durrant with Tnp-clustered bootstrap CI."""
    gold = {json.loads(l)["site_id"]: json.loads(l) for l in open(gold_path)}
    pooled = []
    cluster = []
    p_beats = defaultdict(list)
    p_beats_len = defaultdict(list)
    with open(cog_path) as f:
        for line in f:
            r = json.loads(line)
            g = gold.get(r["site_id"]);
            if g is None: continue
            a = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if a >= len(ncs): a = 0
            nc = ncs[a]; flank = r["inputs"]["flank"]
            prof = np.zeros((len(nc), 16), dtype=np.float32); val = np.zeros((len(nc), 16), dtype=bool)
            _, feats, mask, cands = build_candidate_arrays(
                nc, flank, prof, val, L_min=DEFAULT_L_MIN, L_max=DEFAULT_L_MAX)
            slot, gm = _find_gold(feats, mask, cands,
                                     g["target_flank_orientation"],
                                     g["target_binding_loop_length"],
                                     g["guide_start_in_nc"],
                                     g["target_flank_start"])
            if slot < 0: continue
            valid = np.where(mask)[0]
            local = list(valid)
            cs_local = local.index(slot)
            # raw_m and length_pen(0.5, 12)
            m_arr = feats[valid, 3]
            Ls = np.asarray([cands[int(i)].L for i in valid], dtype=np.float32)
            q_raw = m_arr
            q_lp = m_arr - 0.5 * np.maximum(0.0, Ls - 12.0)
            _, _, MRR_raw = _rank_stats(q_raw, cs_local)
            _, _, MRR_lp  = _rank_stats(q_lp,  cs_local)
            pooled.append((MRR_raw, MRR_lp))
            cluster.append(r["transposase_id"])
            # taxonomy P(gold>d) — raw_m and length_pen
            for j, slot_id in enumerate(local):
                if j == cs_local: continue
                c = cands[int(slot_id)]
                bucket = _classify(c, g["target_flank_orientation"],
                                     g["target_binding_loop_length"],
                                     g["guide_start_in_nc"], g["target_flank_start"])
                p_beats[bucket].append(int(q_raw[cs_local] > q_raw[j]))
                p_beats_len[bucket].append(int(q_lp[cs_local] > q_lp[j]))

    n = len(pooled)
    raws = np.asarray([p[0] for p in pooled])
    lens = np.asarray([p[1] for p in pooled])
    print(f"\n=== A1 :: Durrant  n_in_pool={n} ===")
    print(f"  raw_m       MRR = {raws.mean():.4f}")
    print(f"  length_pen  MRR = {lens.mean():.4f}")
    lo, hi = _bootstrap_delta_clustered(cluster, lens, raws)
    print(f"  Δ MRR (length_pen − raw_m), Tnp-clustered 95% CI = [{lo:+.4f}, {hi:+.4f}]")

    print("  taxonomy P(gold>d):")
    print(f"    {'bucket':<32} {'raw_m':>7} {'length_pen':>11} {'n':>7}")
    tax_report = {}
    for k in ("wrong_orientation","different_region","same_region_longer_L",
              "same_region_shorter_L","same_region_same_L_wrong_flank","near_gold"):
        v_raw = p_beats.get(k, []); v_lp = p_beats_len.get(k, [])
        if not v_raw: print(f"    {k:<32} n=0"); continue
        pr = float(np.mean(v_raw)); pl = float(np.mean(v_lp))
        print(f"    {k:<32} {pr:>7.3f} {pl:>11.3f} {len(v_raw):>7}")
        tax_report[k] = {"raw_m": pr, "length_pen": pl, "n": len(v_raw)}

    return {"n": n,
             "raw_m_MRR": float(raws.mean()),
             "length_pen_MRR": float(lens.mean()),
             "delta_CI": [lo, hi],
             "taxonomy": tax_report}


def run_reweight_taxonomy(val_pool_path: str, gate0_json: str, splits_json: str):
    """A2: taxonomy P(c*>d) on val — unweighted vs reweighted — for raw_m and length_pen.
    Also A3: reproduce weight distribution."""
    from collections import Counter
    print("\n=== A2 :: Reweight taxonomy audit + A3 weight dist ===")

    # Reload the reweight from Gate 0 by recomputing per-bag w_i.
    # First rebuild Durrant marginal and V4.2 val marginal.
    from scripts_helpers import _dummy  # noqa (defensive)


def _rebuild_reweight(val_pool_path, durrant_pairs, method_names=("raw_m","length_pen")):
    """Returns per-bag (weight, {method: (MRR, {bucket -> [beats,...]})})."""
    from collections import Counter
    dur_counts = Counter(durrant_pairs)
    total_dur = sum(dur_counts.values())
    # First scan val to build V4.2 marginal
    v42 = []
    with open(val_pool_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["cstar_slot"] < 0: continue
            cs_slot = rec["cstar_slot"]
            slots = rec["slots"]
            cs_local = None
            for j, s in enumerate(slots):
                if s["slot"] == cs_slot: cs_local = j; break
            if cs_local is None: continue
            cs = slots[cs_local]
            v42.append({"rec": rec, "cs_local": cs_local, "L": int(cs["L"]), "m": int(cs["matches"])})
    v42_counts = Counter((b["L"], b["m"]) for b in v42)
    total_v42 = sum(v42_counts.values())
    alpha = 1.0; n_cells = 17 * 17
    for b in v42:
        p_d = (dur_counts.get((b["L"], b["m"]), 0) + alpha) / (total_dur + alpha * n_cells)
        p_v = (v42_counts.get((b["L"], b["m"]), 0) + alpha) / (total_v42 + alpha * n_cells)
        b["w"] = p_d / max(1e-12, p_v)
    return v42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durrant-cog", required=True)
    ap.add_argument("--durrant-gold", required=True)
    ap.add_argument("--val-pool", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("[collect] Durrant gold pool (L, m) marginal for reweight ...")
    # Rebuild Durrant (L, pool_m) list
    gold = {json.loads(l)["site_id"]: json.loads(l) for l in open(args.durrant_gold)}
    dur_pairs = []
    with open(args.durrant_cog) as f:
        for line in f:
            r = json.loads(line)
            g = gold.get(r["site_id"]);
            if g is None: continue
            a = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if a >= len(ncs): a = 0
            nc = ncs[a]; flank = r["inputs"]["flank"]
            prof = np.zeros((len(nc), 16), dtype=np.float32); val = np.zeros((len(nc), 16), dtype=bool)
            _, feats, mask, cands = build_candidate_arrays(
                nc, flank, prof, val, L_min=DEFAULT_L_MIN, L_max=DEFAULT_L_MAX)
            slot, gm = _find_gold(feats, mask, cands,
                                    g["target_flank_orientation"],
                                    g["target_binding_loop_length"],
                                    g["guide_start_in_nc"],
                                    g["target_flank_start"])
            if slot < 0: continue
            dur_pairs.append((int(cands[slot].L), int(gm)))
    print(f"  Durrant in-pool n={len(dur_pairs)}")

    print("[A1] length_pen + raw_m on Durrant, Tnp-clustered CI ...", flush=True)
    a1 = run_durrant(args.durrant_cog, args.durrant_gold)

    print("[A2 + A3] rebuilding val reweight and taxonomy audit ...", flush=True)
    v42 = _rebuild_reweight(args.val_pool, dur_pairs)
    W = np.asarray([b["w"] for b in v42], dtype=np.float64)
    ESS = float(W.sum() ** 2 / (W ** 2).sum())
    print(f"  n_val={len(W)}  ESS={ESS:.1f}  ({ESS/len(W):.2%})")
    print(f"  weights: min={W.min():.4f}  p25={np.percentile(W,25):.4f}  "
          f"median={np.median(W):.4f}  p75={np.percentile(W,75):.4f}  "
          f"p95={np.percentile(W,95):.4f}  p99={np.percentile(W,99):.4f}  max={W.max():.4f}")
    # Concentration: top-1% of bags carry what fraction of total weight?
    top1_pct = int(0.01 * len(W))
    concentration = float(np.sort(W)[-top1_pct:].sum() / W.sum()) if top1_pct > 0 else 0.0
    print(f"  Top 1% of val bags carry {concentration:.2%} of total sampling weight (gradient-mass concentration risk)")

    print("[A2] weighted taxonomy P(c*>d) on val — raw_m + length_pen ...", flush=True)
    p_beats = {m: defaultdict(list) for m in ("raw_m", "length_pen")}
    weights = {m: defaultdict(list) for m in ("raw_m", "length_pen")}
    for b in v42:
        rec = b["rec"]; slots = rec["slots"]; cs_local = b["cs_local"]
        Ls = np.asarray([int(s["L"]) for s in slots], dtype=np.float32)
        m_arr = np.asarray([float(s["matches"]) for s in slots], dtype=np.float32)
        q_raw = m_arr
        q_lp = m_arr - 0.5 * np.maximum(0.0, Ls - 12.0)
        w_i = b["w"]
        for j, s in enumerate(slots):
            if j == cs_local: continue
            bucket = s["bucket"]
            if bucket == "cstar" or bucket == "unknown": continue
            p_beats["raw_m"][bucket].append((q_raw[cs_local] > q_raw[j], w_i))
            p_beats["length_pen"][bucket].append((q_lp[cs_local] > q_lp[j], w_i))
    print(f"  {'bucket':<32} {'raw_m unw':>10} {'raw_m rw':>9} {'lenpen unw':>11} {'lenpen rw':>10}   {'n':>10}")
    tax_val = {}
    for bucket in ("wrong_orientation","different_region","same_region_longer_L",
                    "same_region_shorter_L","same_region_same_L_wrong_flank","near_gold"):
        row_r = p_beats["raw_m"].get(bucket, [])
        row_l = p_beats["length_pen"].get(bucket, [])
        if not row_r: print(f"  {bucket:<32} n=0"); continue
        vals_r = np.asarray([x[0] for x in row_r], dtype=np.float64)
        ww_r   = np.asarray([x[1] for x in row_r], dtype=np.float64)
        vals_l = np.asarray([x[0] for x in row_l], dtype=np.float64)
        ww_l   = np.asarray([x[1] for x in row_l], dtype=np.float64)
        unw_r = float(vals_r.mean())
        rw_r  = float((vals_r * ww_r).sum() / ww_r.sum())
        unw_l = float(vals_l.mean())
        rw_l  = float((vals_l * ww_l).sum() / ww_l.sum())
        print(f"  {bucket:<32} {unw_r:>10.3f} {rw_r:>9.3f} {unw_l:>11.3f} {rw_l:>10.3f}   {len(row_r):>10}")
        tax_val[bucket] = {"raw_m_unweighted": unw_r, "raw_m_reweighted": rw_r,
                            "length_pen_unweighted": unw_l, "length_pen_reweighted": rw_l,
                            "n": len(row_r)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "A1_durrant":      a1,
            "A2_taxonomy_val": tax_val,
            "A3_weights":      {"ESS": ESS, "ESS_over_N": ESS/len(W),
                                  "min": float(W.min()), "median": float(np.median(W)),
                                  "p95": float(np.percentile(W, 95)),
                                  "p99": float(np.percentile(W, 99)),
                                  "max": float(W.max()),
                                  "top_1pct_gradient_share": concentration},
        }, f, indent=2)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
