"""(P0) Durrant deploy: run current checkpoint on 65 Tnps via per-site orient loader.

Uses:
  - Current best.pt checkpoint (no retrain)
  - New per-site orient loader (bit-exact verified)
  - Per-site orient inferred from bridge_rna × flank alignment (100% agree with gold,
    Rule A default fwd on 11.4% ties)
  - Existing durrant_positive MatchTable shard (paired_bag naming — bag_id renamed
    from cog_bag)

Reports:
  - PRIMARY: paired Δ(B − A_maj) top-1 on S_maj ≤ 1 stratum (n=13, A=0/13)
  - SECONDARY: overall 65 Tnps
  - SECONDARY: pure ≤2 split by tied-at-max vs strict-below
  - Stratified by S_maj_at_gold
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.channel_b.data import ChannelBDataset, BagInputs, bucket_collate_fn
from model.channel_b.model import ChannelBModel
from model.channel_b.constants import MAX_L
from scripts.v5a_framework.match_table import _compute_site_arrays, load as load_mt

BASE = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold")
SHARD_DIR = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/v5a_framework_cache/durrant_positive")

L_TARGET = 11
M_THRESHOLD = 8

_COMP = str.maketrans("ACGTacgt", "TGCAtgca")
def _rc(s): return s.translate(_COMP)[::-1]


def _max_matches(pattern: str, text: str) -> int:
    L = len(pattern)
    if L == 0 or len(text) < L: return 0
    p = np.frombuffer(pattern.encode(), dtype=np.uint8)
    t = np.frombuffer(text.encode(), dtype=np.uint8)
    return int(max(int(np.sum(p == t[i:i+L])) for i in range(len(t) - L + 1)))


def _infer_site_orient(target: str, flank: str) -> str:
    """Deploy-time per-site orient inference. target = bridge_rna's
    target_binding_loop_specificity (or equivalent). Rule A on ties (default fwd)."""
    if not target or not flank: return "fwd"
    m_fwd = _max_matches(target.upper(), flank.upper())
    m_rc  = _max_matches(_rc(target).upper(), flank.upper())
    if m_fwd > m_rc:  return "fwd"
    if m_rc > m_fwd:  return "rc"
    return "fwd"  # Rule A tie-break


def _paired_diff_bootstrap(a, b, n_boot=5000, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    n = len(a_arr)
    diffs = a_arr - b_arr
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = diffs[idx].mean()
    return float(diffs.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


class DurrantChannelBDataset(ChannelBDataset):
    """ChannelBDataset that skips JSONL loading and takes pre-built Tnp records."""

    def __init__(self, tnp_records: list[dict], shard_dir: Path, orient_source_fn):
        # Bypass parent __init__; set fields manually
        self.cache_dir = None
        self.jsonl_path = None
        self.shard_dir = Path(shard_dir)
        self._mt = None
        self._orient_source_fn = orient_source_fn
        # tnp_records: list of {bag_id, nc_len, label, sites: [{labels, inputs}], gold_start, per_site_orient, gold_target}
        self._bag_index = []
        self._sites_by_bag = {}
        self._extra = {}   # bag_id → {gold_start, per_site_orient, gold_target}
        for i, rec in enumerate(tnp_records):
            bag_id = rec["bag_id"]
            self._bag_index.append((bag_id, i, rec["nc_len"], rec["label"]))
            self._sites_by_bag[bag_id] = rec["sites"]
            self._extra[bag_id] = {"gold_start": rec["gold_start"],
                                     "per_site_orient": rec["per_site_orient"],
                                     "gold_target": rec["gold_target"]}
        self._ensure_mt()


def _synthesize_tnp_records(inference_jsonl: Path, gold_table: dict) -> list[dict]:
    """Build Durrant Tnp records with synth-format labels for ChannelBDataset consumption.
    Handles realbg site_id/bag_id prefix by stripping 'realbg_' for gold + shard lookups."""
    def _lookup_g(sid):
        return gold_table.get(sid) or gold_table.get(sid.replace("durrant_realbg_", "durrant_"))
    by_tnp = defaultdict(list)
    for line in open(inference_jsonl):
        r = json.loads(line)
        g = _lookup_g(r["site_id"])
        if not g or g.get("guide_start_in_nc") is None:
            continue
        by_tnp[r["transposase_id"]].append((r, g))

    records = []
    for tnp, pairs in by_tnp.items():
        # Rename bag_id to match durrant_positive shard convention.
        # Shard uses "durrant_bridge_RNA_..._paired_bag..." (no realbg).
        shard_bag_id = tnp.replace("_cog_", "_paired_").replace("durrant_realbg_", "durrant_")
        # Gold start (shared across sites)
        gold_start = int(pairs[0][1]["guide_start_in_nc"])
        gold_len   = int(pairs[0][1]["guide_end_in_nc"]) - gold_start
        # Target binding loop (per-Tnp)
        target = (pairs[0][1].get("target_binding_loop_specificity") or "").upper()
        # nc — bridge_rna is a substring of nc; verified offset 0 for unpadded
        nc = pairs[0][0]["inputs"]["noncoding_regions"][
            pairs[0][0]["labels"].get("active_noncoding_index", 0)]
        nc_len = len(nc)
        # Majority orient across sites (from gold table)
        orient_counts = Counter(g.get("target_flank_orientation") for _, g in pairs)
        majority_orient = orient_counts.most_common(1)[0][0]
        # Per-site orient via deploy alignment (rebuilt at load time via orient_source_fn)
        per_site_orient = [_infer_site_orient(target, r["inputs"]["flank"]) for r, _ in pairs]
        # Synthesize sites with minimal-required labels blob
        sites = []
        for site_idx, (r, g) in enumerate(pairs):
            site_labels = {
                "canonical_nc": nc,
                "guide_length": gold_len,
                "arch": {"orient": majority_orient, "n_sites": len(pairs),
                          "flank_offset_mode": "consistent",
                          "nc_homology_rate": 1.0},
                "guide_span_in_active_noncoding": [gold_start, gold_start + gold_len],
                "ncrna_length": nc_len,
                "is_positive": 1,
                "active_noncoding_index": 0,
                "num_noncoding_regions": 1,
                "is_planted": True,
            }
            sites.append({"transposase_id": shard_bag_id,
                            "site_id": r["site_id"].replace("_cog_", "_paired_"),
                            "labels": site_labels,
                            "inputs": r["inputs"]})
        records.append({
            "bag_id": shard_bag_id, "orig_tnp": tnp,
            "nc_len": nc_len, "label": 1,
            "sites": sites, "gold_start": gold_start, "gold_len": gold_len,
            "per_site_orient": per_site_orient, "gold_target": target,
        })
    return records


def _S_at_orient(mt, bag_id, orient, n_sites):
    """A_maj per-orient S(p) at L=11 threshold m>=8."""
    n_pos = None
    S = None
    for s_idx in range(n_sites):
        ma = mt.get(bag_id, s_idx, orient, L_TARGET)
        arr = ma.m_max_by_excl.get(0)
        if arr is None: continue
        if n_pos is None:
            n_pos = arr.shape[0]
            S = np.zeros(n_pos, dtype=np.int32)
        Lp = min(n_pos, arr.shape[0])
        S[:Lp] += (arr[:Lp] >= M_THRESHOLD).astype(np.int32)
    return S if S is not None else np.zeros(0, dtype=np.int32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                     default=Path("checkpoints/channel_b/main_lr3e-4/best.pt"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--zero-channels", type=str, default="",
                     help="Comma-separated channel indices to zero at inference. "
                          "Use with --ckpt to a mmax-only checkpoint (which was trained "
                          "with zero-fill on those channels).")
    ap.add_argument("--source-file", type=str, default="durrant_cognate.jsonl",
                     help="Filename under BASE/inference/. Default = unpadded cognate. "
                          "Use durrant_cognate_realbg_ncpad240.jsonl for 31-nt real-bg shift.")
    args = ap.parse_args()
    zero_ch = [int(x) for x in args.zero_channels.split(",") if x.strip()]
    if zero_ch:
        print(f"[P0] ZEROING input channels at inference: {zero_ch}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P0] device={device}  ckpt={args.ckpt}")

    # Load model
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    train_args = blob.get("args", {})
    model = ChannelBModel(hidden=train_args.get("hidden", 128),
                               n_heads=train_args.get("n_heads", 4),
                               n_blocks=train_args.get("n_blocks", 3)).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    # Load Durrant gold
    gold_table = {}
    for line in open(BASE / "annotation/durrant_gold_v1.jsonl"):
        r = json.loads(line)
        gold_table[r["site_id"]] = r

    # Synthesize Tnp records
    src = BASE / "inference" / args.source_file
    print(f"[P0] source: {src}")
    records = _synthesize_tnp_records(src, gold_table)
    print(f"[P0] synthesized {len(records)} Tnp records")

    # orient_source_fn: return per-site inferred orient
    def _orient_fn(bag_id, sites, s_idx, bag_orient):
        # sites has synthesized labels. per-site orient was precomputed and stashed in extra.
        # But we can't access _extra from this closure easily; recompute here (cheap).
        first_lab = sites[0]["labels"]
        # target is same across all sites of a Tnp — recompute from bridge_rna context.
        # Actually: we stashed per_site_orient at record synthesis; retrieve from dataset.
        return dset._extra[bag_id]["per_site_orient"][s_idx]

    dset = DurrantChannelBDataset(records, SHARD_DIR, orient_source_fn=_orient_fn)
    print(f"[P0] dataset ready. shard tnps: {len(dset._mt.tnp_ids)}; our tnps: {len(dset)}")

    loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=bucket_collate_fn,
                            pin_memory=(device.type == "cuda"))

    # Forward
    rows = []
    with torch.no_grad():
        idx_iter = iter(range(len(dset)))
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            if zero_ch:
                x = x.clone(); x[..., zero_ch] = 0.0
            site_mask = batch["site_mask"].to(device, non_blocking=True)
            pos_mask = batch["pos_mask"].to(device, non_blocking=True)
            pred = model(x, site_mask)
            masked = pred * pos_mask.float() + (-1e9) * (~pos_mask).float()
            for i in range(x.shape[0]):
                local_i = next(idx_iter)
                bag_id, _off, nc_len, _label = dset._bag_index[local_i]
                extra = dset._extra[bag_id]
                gold_start = extra["gold_start"]
                per_site_orient = extra["per_site_orient"]
                gold_target = extra["gold_target"]
                # A_maj baseline: per-orient S at majority orient
                orient_counts = Counter(per_site_orient)
                majority_orient = orient_counts.most_common(1)[0][0]
                n_sites = len(per_site_orient)
                S_maj = _S_at_orient(dset._mt, bag_id, majority_orient, n_sites)
                if S_maj.size == 0 or gold_start >= S_maj.size:
                    continue
                a_argmax = int(S_maj.argmax())
                S_maj_at_gold = int(S_maj[gold_start])
                # B: model argmax over valid positions
                b_scores = masked[i][pos_mask[i]].cpu().numpy()
                b_argmax = int(np.argmax(b_scores))
                # Top-1 hit (tie-lenient: gold is at max)
                a_top1_lenient = (int(S_maj[gold_start]) == int(S_maj.max()))
                a_top1_argpart = (a_argmax == gold_start)
                b_top1_lenient = (float(b_scores[gold_start]) == float(b_scores.max())) if gold_start < len(b_scores) else False
                b_top1_argpart = (b_argmax == gold_start)
                is_pure = (len(orient_counts) == 1)
                rows.append({
                    "bag_id": bag_id, "orient_majority": majority_orient,
                    "is_pure": is_pure, "n_sites": n_sites,
                    "gold_start": gold_start,
                    "S_maj_at_gold": S_maj_at_gold,
                    "a_argmax": a_argmax, "b_argmax": b_argmax,
                    "a_top1_argpart": a_top1_argpart, "b_top1_argpart": b_top1_argpart,
                    "a_top1_lenient": a_top1_lenient, "b_top1_lenient": b_top1_lenient,
                })

    n = len(rows)
    print(f"\n[P0] scored n={n} Tnps")

    if n == 0:
        print("no scoreable rows"); return 1

    def _m(rs, k): return float(np.mean([r[k] for r in rs]))

    print(f"\n=== PRIMARY: S_maj ≤ 1 stratum (n=13 expected, A=0/13 by design) ===")
    sub = [r for r in rows if r["S_maj_at_gold"] <= 1]
    n_sub = len(sub)
    if n_sub == 0:
        print("  no S≤1 Tnps");
    else:
        a1 = _m(sub, "a_top1_argpart")
        b1 = _m(sub, "b_top1_argpart")
        n_b_hit = sum(r["b_top1_argpart"] for r in sub)
        print(f"  n_sub={n_sub}  A top-1 (argpart)={a1:.4f}  B top-1 (argpart)={b1:.4f}  B hits={n_b_hit}/{n_sub}")
        if len(sub) >= 2:
            d, lo, hi = _paired_diff_bootstrap([r["b_top1_argpart"] for r in sub],
                                                     [r["a_top1_argpart"] for r in sub])
            print(f"  paired Δ(B−A) = {d:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  {'SIG' if (lo > 0 or hi < 0) else 'ns'}")
        # Verdict: CONFIRM if ≥3 hits, FALSIFY if 0
        if n_b_hit >= 3:
            print(f"  → CONFIRM: B ≥ 3 hits on S≤1 stratum (0 for A). Sub-threshold rescue transfers.")
        elif n_b_hit == 0:
            print(f"  → FALSIFY: B 0 hits on S≤1 stratum. Mechanism does not transfer.")
        else:
            print(f"  → INCONCLUSIVE: B {n_b_hit} hits, expected under noise given n={n_sub}.")

    print(f"\n=== stratified by S_maj_at_gold (tie-lenient bounds shown) ===")
    print(f"  {'S_maj':>5s}  {'n':>3s}  {'A(argpart)':>11s}  {'B(argpart)':>11s}  {'A(lenient)':>11s}  {'B(lenient)':>11s}")
    for s in sorted({r["S_maj_at_gold"] for r in rows}):
        sub = [r for r in rows if r["S_maj_at_gold"] == s]
        print(f"  {s:>5d}  {len(sub):>3d}  {_m(sub,'a_top1_argpart'):>11.4f}  "
              f"{_m(sub,'b_top1_argpart'):>11.4f}  {_m(sub,'a_top1_lenient'):>11.4f}  "
              f"{_m(sub,'b_top1_lenient'):>11.4f}")

    print(f"\n=== overall (all 65 Tnps, majority orient) ===")
    if rows:
        d, lo, hi = _paired_diff_bootstrap([r["b_top1_argpart"] for r in rows],
                                                  [r["a_top1_argpart"] for r in rows])
        print(f"  A top-1 (argpart)={_m(rows,'a_top1_argpart'):.4f}  "
              f"B top-1 (argpart)={_m(rows,'b_top1_argpart'):.4f}  "
              f"Δ={d:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  {'SIG' if (lo > 0 or hi < 0) else 'ns'}")

    # Secondary: pure vs mixed
    print(f"\n=== pure/mixed split (all Tnps) ===")
    for label, sel in [("pure", lambda r: r["is_pure"]),
                        ("mixed", lambda r: not r["is_pure"])]:
        sub = [r for r in rows if sel(r)]
        if not sub: continue
        print(f"  {label:>6s} n={len(sub)}  A={_m(sub,'a_top1_argpart'):.4f}  B={_m(sub,'b_top1_argpart'):.4f}")

    # === Full B_argmax distribution across all 65 Tnps ===
    # If 65 Tnps share the same ncRNA scaffold → identical structure inputs →
    # model's structure attention should produce a FIXED positional attractor.
    # Check whether B's argmax clusters into a small number of "favorite" positions.
    b_argmax_dist = Counter(r["b_argmax"] for r in rows)
    a_argmax_dist = Counter(r["a_argmax"] for r in rows)
    print(f"\n=== B_argmax distribution across all 65 Tnps ===")
    for pos, count in sorted(b_argmax_dist.items(), key=lambda x: -x[1])[:20]:
        pct = count / len(rows) * 100
        star = " ← gold" if pos == 49 else ""
        print(f"  pos {pos:>4d}: {count:>3d} Tnps ({pct:>5.1f}%){star}")
    print(f"  ... unique positions: {len(b_argmax_dist)} out of 65 Tnps")
    print(f"\n=== A_argmax distribution (for reference) ===")
    for pos, count in sorted(a_argmax_dist.items(), key=lambda x: -x[1])[:10]:
        pct = count / len(rows) * 100
        star = " ← gold" if pos == 49 else ""
        print(f"  pos {pos:>4d}: {count:>3d} Tnps ({pct:>5.1f}%){star}")
    print(f"  ... unique positions: {len(a_argmax_dist)}")

    # === CASE-BY-CASE: Tnps where A hits but B misses (per user 2026-09-08) ===
    # Focus on S_maj = 3 and 4 strata where B loses -40pp and -31pp.
    # For each miss: report B's argmax position, distance from gold, S at B's pick.
    print(f"\n=== case-by-case: A hits & B misses on S_maj ∈ {{3, 4}} ===")
    miss_focus = [r for r in rows if r["S_maj_at_gold"] in (3, 4)
                    and r["a_top1_argpart"] and not r["b_top1_argpart"]]
    print(f"  n_focus_misses = {len(miss_focus)}")
    print(f"  {'bag_id':<48s}  {'S_g':>3s}  {'gold':>5s}  {'B_argmax':>8s}  {'|Δ|':>4s}  "
          f"{'S_at_B':>6s}  {'pure':>4s}")
    for r in sorted(miss_focus, key=lambda x: (x["S_maj_at_gold"], x["bag_id"])):
        bag_id = r["bag_id"]
        # Recompute S_maj at bag orient to look up S at B_argmax
        maj_orient = r["orient_majority"]
        S_maj = _S_at_orient(dset._mt, bag_id, maj_orient, r["n_sites"])
        S_at_B = int(S_maj[r["b_argmax"]]) if r["b_argmax"] < S_maj.size else -1
        dist = abs(r["b_argmax"] - r["gold_start"])
        pure_tag = "Y" if r["is_pure"] else "N"
        print(f"  {bag_id[:48]:<48s}  {r['S_maj_at_gold']:>3d}  {r['gold_start']:>5d}  "
              f"{r['b_argmax']:>8d}  {dist:>4d}  {S_at_B:>6d}  {pure_tag:>4s}")

    # Aggregate: distribution of |Δ| and S at B's pick
    if miss_focus:
        dists = [abs(r["b_argmax"] - r["gold_start"]) for r in miss_focus]
        s_at_bs = []
        for r in miss_focus:
            S_maj = _S_at_orient(dset._mt, r["bag_id"], r["orient_majority"], r["n_sites"])
            s_at_bs.append(int(S_maj[r["b_argmax"]]) if r["b_argmax"] < S_maj.size else -1)
        print(f"\n  |Δ| dist:     {Counter(dists)}")
        print(f"  S at B's pick vs S at gold: mean(S_at_B)={np.mean(s_at_bs):.2f}  "
              f"mean(S_at_gold)={np.mean([r['S_maj_at_gold'] for r in miss_focus]):.2f}")
        # Reading: if S_at_B > S_at_gold, B is picking a HIGHER-S peak than gold → not obvious
        #   what B is doing differently from A_argmax (both would prefer higher S).
        # if S_at_B == S_at_gold, B is tie-breaking differently from A_argmax → structure/flank driven.
        # if S_at_B < S_at_gold, B is preferring a LOWER-S position → structure/flank dominates against m_max.
        n_higher = sum(1 for sb, r in zip(s_at_bs, miss_focus) if sb > r["S_maj_at_gold"])
        n_equal  = sum(1 for sb, r in zip(s_at_bs, miss_focus) if sb == r["S_maj_at_gold"])
        n_lower  = sum(1 for sb, r in zip(s_at_bs, miss_focus) if sb < r["S_maj_at_gold"])
        print(f"  B picked position with S: HIGHER than gold's S: {n_higher}/{len(miss_focus)}")
        print(f"                             EQUAL to  gold's S: {n_equal}/{len(miss_focus)}")
        print(f"                             LOWER than gold's S: {n_lower}/{len(miss_focus)}  ← structure/flank-driven misses")

    return 0


if __name__ == "__main__":
    sys.exit(main())
