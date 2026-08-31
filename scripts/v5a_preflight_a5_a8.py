"""V5A preflight batch A5 + A6 + A7 + A8.

A5: length_pen full R@1/R@4/R@8 on Durrant; Δ R@8 CIs (MIL vs raw_m, MIL vs length_pen);
    complementarity — overlap of length_pen's misses vs MIL's hits.
    MIL attention scores come from the A4 result JSON (per-site MRR only) — for
    A5 we need per-site R@k arrays and Top-8 slot ids. Rerun A4 attention inline.
A6: (α, L0) grid on REWEIGHTED val + on Durrant → two-parameter family ceiling
    (labeled in-sample; formal Gate A bar stays at 0.125 transfer).
A7: top-8 decoy CLASS SHARES on reweighted val vs Durrant.
A8: ESS under weight clipping at p95 / p99 / p99.5; formalize canonical eval
    module (v5a_eval_core.py) already committed.
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

from preprocess.candidates import build_candidate_arrays, DEFAULT_L_MIN, DEFAULT_L_MAX
from scripts.v5a_eval_core import (
    rank_stats, bootstrap_delta_clustered, find_gold_slot, classify_decoy,
    score_length_pen, DECOY_BUCKETS,
)


def build_durrant_records(cog_path, gold_path):
    """Yield per-site records ready for scoring; canonical in-pool definition."""
    gold = {json.loads(l)["site_id"]: json.loads(l) for l in open(gold_path)}
    out = []
    with open(cog_path) as f:
        for line in f:
            r = json.loads(line)
            g = gold.get(r["site_id"]);
            if g is None: continue
            a = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if a >= len(ncs): a = 0
            nc = ncs[a]; flank = r["inputs"]["flank"]
            prof = np.zeros((len(nc), 16), dtype=np.float32)
            val = np.zeros((len(nc), 16), dtype=bool)
            _, feats, mask, cands = build_candidate_arrays(
                nc, flank, prof, val, L_min=DEFAULT_L_MIN, L_max=DEFAULT_L_MAX)
            slot, gm = find_gold_slot(feats, mask, cands,
                                          g["target_flank_orientation"],
                                          g["target_binding_loop_length"],
                                          g["guide_start_in_nc"],
                                          g["target_flank_start"])
            if slot < 0: continue
            out.append({
                "site_id":     r["site_id"],
                "tnp_id":      r["transposase_id"],
                "cs_slot":     int(slot),
                "cs_matches":  float(gm),
                "feats":       feats,
                "mask":        mask,
                "cands":       cands,
                "gold_orient": g["target_flank_orientation"],
                "gold_L":      g["target_binding_loop_length"],
                "gold_nc":     g["guide_start_in_nc"],
                "gold_fl":     g["target_flank_start"],
            })
    return out


def a5_length_pen_full(recs, alpha=0.5, L0=12):
    """Full R@1/R@4/R@8 for raw_m and length_pen on Durrant."""
    print(f"\n=== A5.1 :: full R@k of raw_m and length_pen({alpha}, {L0}) on Durrant ===")
    MRR_raw = []; R1_raw = []; R4_raw = []; R8_raw = []
    MRR_lp  = []; R1_lp  = []; R4_lp  = []; R8_lp  = []
    top8_raw = {}; top8_lp = {}
    tnp = []
    for rec in recs:
        valid = np.where(rec["mask"])[0]
        Ls = np.asarray([rec["cands"][int(i)].L for i in valid], dtype=np.float32)
        m_arr = rec["feats"][valid, 3]
        q_raw = m_arr
        q_lp  = m_arr - alpha * np.maximum(0.0, Ls - L0)
        cs_pos = int(np.where(valid == rec["cs_slot"])[0][0])
        _, R_r, MRR_r = rank_stats(q_raw, cs_pos)
        _, R_l, MRR_l = rank_stats(q_lp,  cs_pos)
        MRR_raw.append(MRR_r); MRR_lp.append(MRR_l)
        R1_raw.append(R_r[1]); R4_raw.append(R_r[4]); R8_raw.append(R_r[8])
        R1_lp .append(R_l[1]); R4_lp .append(R_l[4]); R8_lp .append(R_l[8])
        # Top-8 slot ids (for A5.3 complementarity)
        top8_raw[rec["site_id"]] = valid[np.argsort(-q_raw)[:8]].tolist()
        top8_lp[rec["site_id"]]  = valid[np.argsort(-q_lp)[:8]].tolist()
        tnp.append(rec["tnp_id"])
    def _s(vs): return {"mean": float(np.mean(vs)), "n": len(vs)}
    print(f"  raw_m       R@1={np.mean(R1_raw):.4f}  R@4={np.mean(R4_raw):.4f}  R@8={np.mean(R8_raw):.4f}  MRR={np.mean(MRR_raw):.4f}")
    print(f"  length_pen  R@1={np.mean(R1_lp):.4f}   R@4={np.mean(R4_lp):.4f}   R@8={np.mean(R8_lp):.4f}   MRR={np.mean(MRR_lp):.4f}")
    lo, hi = bootstrap_delta_clustered(tnp, np.asarray(R8_lp), np.asarray(R8_raw))
    print(f"  Δ R@8 (length_pen − raw_m) Tnp-CI = [{lo:+.4f}, {hi:+.4f}]")
    lo, hi = bootstrap_delta_clustered(tnp, np.asarray(MRR_lp), np.asarray(MRR_raw))
    print(f"  Δ MRR (length_pen − raw_m) Tnp-CI = [{lo:+.4f}, {hi:+.4f}]")
    return {
        "raw_m":      {"R@1": np.mean(R1_raw), "R@4": np.mean(R4_raw),
                          "R@8": np.mean(R8_raw), "MRR": np.mean(MRR_raw)},
        "length_pen": {"R@1": np.mean(R1_lp),  "R@4": np.mean(R4_lp),
                          "R@8": np.mean(R8_lp),  "MRR": np.mean(MRR_lp)},
        "top8_raw_m":     top8_raw,
        "top8_length_pen": top8_lp,
        "MRR_lp_arr":  MRR_lp, "MRR_raw_arr": MRR_raw,
        "R8_lp_arr":   R8_lp,  "R8_raw_arr":  R8_raw,
        "tnp":         tnp,
    }


def a5_mil_vs_others(recs, a4_json_path, a5_res):
    """A4 result loaded and used to compute Δ R@8 vs raw_m and length_pen +
    complementarity (fraction of bags length_pen misses at R@8 that MIL catches)."""
    print(f"\n=== A5.2 :: MIL attention vs raw_m, length_pen ===")
    # A4 script had per-record R@k arrays but the JSON only has aggregates.
    # For a proper Δ we need per-record MIL R@8 too — re-run inline is heavy
    # (needs GPU model). Instead compute a4 comparison on the SAME sites where
    # A4 scored, and use its saved arrays. Since A4's script didn't save arrays,
    # fall back on aggregate deltas + note the limitation.
    with open(a4_json_path) as f: a4 = json.load(f)
    print(f"  A4 aggregate (n_sites={a4['n_sites']}):")
    print(f"    MIL         R@1={a4['MIL']['R@1']:.4f}  R@4={a4['MIL']['R@4']:.4f}  R@8={a4['MIL']['R@8']:.4f}  MRR={a4['MIL']['MRR']:.4f}")
    print(f"    raw_m (A4)  R@1={a4['raw_m']['R@1']:.4f}  R@4={a4['raw_m']['R@4']:.4f}  R@8={a4['raw_m']['R@8']:.4f}  MRR={a4['raw_m']['MRR']:.4f}")
    print(f"    MIL R@8 vs raw_m R@8: +{a4['MIL']['R@8'] - a4['raw_m']['R@8']:.4f} absolute, "
          f"{(a4['MIL']['R@8'] / a4['raw_m']['R@8'] - 1)*100:+.1f}% relative")
    print(f"    length_pen R@8={a5_res['length_pen']['R@8']:.4f}   vs MIL {a4['MIL']['R@8']:.4f}")
    print(f"    Δ R@8 (MIL − length_pen) aggregate = {a4['MIL']['R@8'] - a5_res['length_pen']['R@8']:+.4f}")
    print(f"    (per-record CI on MIL Δ requires A4 per-record arrays — flagged for next A4 run to save)")
    return {"a4_agg": a4}


def a5_complementarity(recs, a5_res):
    """A5.3: of bags where length_pen puts gold outside top-8, how many does
    MIL 'catch' (rank ≤ 8)? Requires MIL top-8 per site — placeholder using
    counts of gold_in_lp_top8 & gold_in_MIL_top8 (relies on a rerun of A4
    saving per-record top-K sets)."""
    n = len(recs)
    lp_hits = 0
    for rec in recs:
        top8 = a5_res["top8_length_pen"][rec["site_id"]]
        if rec["cs_slot"] in top8: lp_hits += 1
    lp_hit_rate = lp_hits / n
    print(f"\n=== A5.3 :: complementarity (length_pen R@8 hit rate) ===")
    print(f"  n={n}  length_pen R@8 hits = {lp_hits} ({lp_hit_rate:.2%})")
    print(f"  (MIL complementarity intersection reserved for A4 rerun with saved top-K)")
    return {"n": n, "length_pen_R8_hits": lp_hits, "length_pen_R8_hit_rate": lp_hit_rate}


def a6_grid(recs_dur, val_pool_path, dur_pairs):
    """A6: (α, L0) grid on Durrant + on reweighted val. Report Durrant-optimal
    as the two-parameter family ceiling (labeled in-sample)."""
    print(f"\n=== A6 :: (α, L0) grid — length_pen family ceiling ===")
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    L0s = [8, 9, 10, 11, 12, 13, 14]
    dur_grid = {}
    for a in alphas:
        for L0 in L0s:
            mrs = []
            for rec in recs_dur:
                valid = np.where(rec["mask"])[0]
                Ls = np.asarray([rec["cands"][int(i)].L for i in valid], dtype=np.float32)
                m_arr = rec["feats"][valid, 3]
                q = m_arr - a * np.maximum(0.0, Ls - L0)
                cs_pos = int(np.where(valid == rec["cs_slot"])[0][0])
                _, _, MRR = rank_stats(q, cs_pos)
                mrs.append(MRR)
            dur_grid[(a, L0)] = float(np.mean(mrs))

    best = max(dur_grid.items(), key=lambda kv: kv[1])
    print(f"  Durrant grid best (α, L0) = {best[0]}   MRR = {best[1]:.4f}  (in-sample)")
    print(f"  Ceiling headroom vs (0.5, 12) transfer bar: {best[1] - dur_grid[(0.5, 12)]:+.4f}")
    print(f"  full grid (row=α, col=L0):")
    header_label = "a\\L0"
    print(f"    {header_label:>6} " + " ".join(f"{L0:>7}" for L0 in L0s))
    for a in alphas:
        row = " ".join(f"{dur_grid[(a, L0)]:>7.4f}" for L0 in L0s)
        print(f"    {a:>6.2f} {row}")

    # Reweighted val grid — reuse Gate 0 weights.
    dur_counts = Counter(dur_pairs)
    total_dur = sum(dur_counts.values())
    val = []
    with open(val_pool_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["cstar_slot"] < 0: continue
            cs_local = next((j for j, s in enumerate(rec["slots"])
                                if s["slot"] == rec["cstar_slot"]), None)
            if cs_local is None: continue
            slots = rec["slots"]
            val.append({"rec": rec, "cs_local": cs_local,
                          "L_cs": int(slots[cs_local]["L"]),
                          "m_cs": int(slots[cs_local]["matches"]),
                          "L_arr": np.asarray([int(s["L"]) for s in slots], dtype=np.float32),
                          "m_arr": np.asarray([float(s["matches"]) for s in slots], dtype=np.float32)})
    v42_counts = Counter((b["L_cs"], b["m_cs"]) for b in val)
    total_v42 = sum(v42_counts.values())
    alpha_smoothing = 1.0; n_cells = 17 * 17
    for b in val:
        p_d = (dur_counts.get((b["L_cs"], b["m_cs"]), 0) + alpha_smoothing) / (total_dur + alpha_smoothing * n_cells)
        p_v = (v42_counts.get((b["L_cs"], b["m_cs"]), 0) + alpha_smoothing) / (total_v42 + alpha_smoothing * n_cells)
        b["w"] = p_d / max(1e-12, p_v)

    rw_grid = {}
    for a in alphas:
        for L0 in L0s:
            wgt = 0.0; num = 0.0
            for b in val:
                q = b["m_arr"] - a * np.maximum(0.0, b["L_arr"] - L0)
                _, _, MRR = rank_stats(q, b["cs_local"])
                num += b["w"] * MRR
                wgt += b["w"]
            rw_grid[(a, L0)] = num / max(wgt, 1e-12)
    best_rw = max(rw_grid.items(), key=lambda kv: kv[1])
    print(f"\n  Reweighted val grid best (α, L0) = {best_rw[0]}   MRR = {best_rw[1]:.4f}  (in-sample)")
    print(f"  Reweighted val (0.5, 12) transfer bar: {rw_grid[(0.5, 12)]:.4f}")

    return {
        "durrant_grid":   {f"a={a}_L0={L0}": v for (a, L0), v in dur_grid.items()},
        "durrant_best":   {"alpha": best[0][0], "L0": best[0][1], "MRR": best[1]},
        "reweighted_grid": {f"a={a}_L0={L0}": v for (a, L0), v in rw_grid.items()},
        "reweighted_best": {"alpha": best_rw[0][0], "L0": best_rw[0][1], "MRR": best_rw[1]},
        "val_weights":     [b["w"] for b in val],
    }


def a7_class_shares(recs_dur, val_pool_path):
    """A7: top-8 decoy class shares on Durrant vs REWEIGHTED val."""
    print(f"\n=== A7 :: top-8 decoy class shares (weighted) ===")
    # Durrant: unweighted counts.
    dur_cls = Counter()
    for rec in recs_dur:
        valid = np.where(rec["mask"])[0]
        m_arr = rec["feats"][valid, 3]
        cs_pos = int(np.where(valid == rec["cs_slot"])[0][0])
        order = np.argsort(-m_arr, kind="stable")
        for j in order[:9]:
            if j == cs_pos: continue
            c = rec["cands"][int(valid[j])]
            bucket = classify_decoy(c, rec["gold_orient"], rec["gold_L"],
                                       rec["gold_nc"], rec["gold_fl"])
            dur_cls[bucket] += 1
        # cap at 8 non-cstar
    dur_tot = sum(dur_cls.values()) or 1
    # Reweighted val: weighted by w_i.
    # Load Gate 0 weights (or recompute quickly).
    val_w = json.load(open("/global/scratch/users/kh36969/DL_novel_guide_editor/v5a/amendments_a1_a3.json"))
    # amendments_a1_a3.json's A3 has weights summary but not per-bag weights.
    # Compute per-bag weights on the fly using the same formula as Gate 0.
    from collections import Counter as C
    dur_pairs = []
    for rec in recs_dur: dur_pairs.append((int(rec["cands"][rec["cs_slot"]].L),
                                                int(rec["feats"][rec["cs_slot"], 3])))
    dur_counts = C(dur_pairs); total_dur = sum(dur_counts.values())
    val_cls_w = defaultdict(float); val_tot_w = 0.0
    v42_pairs = []
    with open(val_pool_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["cstar_slot"] < 0: continue
            cs_local = next((j for j, s in enumerate(rec["slots"]) if s["slot"] == rec["cstar_slot"]), None)
            if cs_local is None: continue
            v42_pairs.append((int(rec["slots"][cs_local]["L"]), int(rec["slots"][cs_local]["matches"])))
    v42_counts = C(v42_pairs); total_v42 = sum(v42_counts.values())
    alpha_smoothing = 1.0; n_cells = 17 * 17
    with open(val_pool_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["cstar_slot"] < 0: continue
            slots = rec["slots"]
            cs_local = next((j for j, s in enumerate(slots) if s["slot"] == rec["cstar_slot"]), None)
            if cs_local is None: continue
            L = int(slots[cs_local]["L"]); m = int(slots[cs_local]["matches"])
            p_d = (dur_counts.get((L, m), 0) + alpha_smoothing) / (total_dur + alpha_smoothing * n_cells)
            p_v = (v42_counts.get((L, m), 0) + alpha_smoothing) / (total_v42 + alpha_smoothing * n_cells)
            w = p_d / max(1e-12, p_v)
            # rank by raw m; take top-8 non-cs
            qs = np.asarray([float(s["matches"]) for s in slots], dtype=np.float32)
            order = np.argsort(-qs, kind="stable")
            count = 0
            for j in order:
                if j == cs_local: continue
                if count >= 8: break
                bucket = slots[j].get("bucket", "unknown")
                if bucket == "cstar" or bucket == "unknown": continue
                val_cls_w[bucket] += w
                count += 1
            val_tot_w += w * 8   # 8 slots per bag
    print(f"  {'bucket':<34} {'Durrant':>10} {'reweighted val':>17}")
    a7 = {}
    for k in DECOY_BUCKETS:
        d = dur_cls.get(k, 0) / dur_tot
        v = val_cls_w.get(k, 0.0) / max(val_tot_w, 1e-12)
        print(f"  {k:<34} {d:>10.3f} {v:>17.3f}")
        a7[k] = {"durrant": d, "reweighted_val": v}
    return a7


def a8_clipping(val_pool_path, dur_pairs):
    """A8: ESS under weight clipping at various quantiles."""
    print(f"\n=== A8 :: ESS under weight clipping ===")
    from collections import Counter as C
    val = []
    with open(val_pool_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["cstar_slot"] < 0: continue
            slots = rec["slots"]
            cs_local = next((j for j, s in enumerate(slots) if s["slot"] == rec["cstar_slot"]), None)
            if cs_local is None: continue
            L = int(slots[cs_local]["L"]); m = int(slots[cs_local]["matches"])
            val.append((L, m))
    dur_counts = C(dur_pairs); v42_counts = C(val)
    total_dur = sum(dur_counts.values()); total_v42 = sum(v42_counts.values())
    a = 1.0; n_cells = 17 * 17
    W = np.asarray([
        (dur_counts.get(p, 0) + a) / (total_dur + a * n_cells)
        / max(1e-12, (v42_counts.get(p, 0) + a) / (total_v42 + a * n_cells))
        for p in val], dtype=np.float64)
    print(f"  n_val={len(W)}")
    def _ess(w):
        return float(w.sum() ** 2 / (w ** 2).sum())
    print(f"  {'cap':<15} {'W_max_after':>14} {'ESS':>10} {'ESS/N':>10}")
    result = {"quantiles": {}}
    for q in (None, 0.95, 0.99, 0.995, 0.999):
        if q is None:
            cap = np.inf; ww = W; tag = "no clip"
        else:
            cap = float(np.percentile(W, q * 100)); ww = np.minimum(W, cap); tag = f"p{q*100:.1f}"
        ess = _ess(ww)
        print(f"  {tag:<15} {ww.max():>14.4f} {ess:>10.1f} {ess/len(W):>10.2%}")
        result["quantiles"][tag] = {"cap": None if q is None else cap,
                                       "ESS": ess, "ESS_over_N": ess / len(W)}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durrant-cog", required=True)
    ap.add_argument("--durrant-gold", required=True)
    ap.add_argument("--val-pool", required=True)
    ap.add_argument("--a4-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("[collect] Durrant in-pool records via canonical build ...", flush=True)
    recs_dur = build_durrant_records(args.durrant_cog, args.durrant_gold)
    print(f"  n_in_pool = {len(recs_dur)}", flush=True)

    a5_1 = a5_length_pen_full(recs_dur)
    a5_2 = a5_mil_vs_others(recs_dur, args.a4_json, a5_1)
    a5_3 = a5_complementarity(recs_dur, a5_1)

    dur_pairs = [(int(rec["cands"][rec["cs_slot"]].L),
                    int(rec["feats"][rec["cs_slot"], 3])) for rec in recs_dur]
    a6 = a6_grid(recs_dur, args.val_pool, dur_pairs)
    a7 = a7_class_shares(recs_dur, args.val_pool)
    a8 = a8_clipping(args.val_pool, dur_pairs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = {
        "n_in_pool_canonical": len(recs_dur),
        "A5.1_length_pen_full": {"raw_m": a5_1["raw_m"], "length_pen": a5_1["length_pen"]},
        "A5.2_mil_agg":          a5_2["a4_agg"],
        "A5.3_complementarity":  a5_3,
        "A6_grid":               a6,
        "A7_class_shares":       a7,
        "A8_clipping":           a8,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
