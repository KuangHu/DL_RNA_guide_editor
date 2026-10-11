"""Per-candidate alignment inspection — step 2 cont'd.

For one bag + one target cluster (nc_pos, L, flank_pos, orient-free):
  1. Replay Phase A→D to identify the ACTUAL supporting-site set
     (not inferred, not guessed).
  2. For every site (supporting + non-supporting) dump:
       - the per-site nc window: nc[nc_pos − flank_nt : nc_pos + L + flank_nt],
         with the guide-window boundaries explicitly marked.
       - the per-site flank window around the chosen hit (orientation-aware).
       - a per-base match line showing guide vs target
         (with revcomp applied if orient='rc').
  3. Cross-site ncRNA alignment of the supporting sites' nc contexts
     via a simple global pairwise match-count map (not a full MSA, but
     enough to see whether the ±flank_nt context aligns column-wise or not).

Does NOT claim homology — presents the evidence for the reviewer to judge.

Usage:
  python scripts/inspect_cluster_alignment.py \
    --jsonl .../adapted.jsonl \
    --bag CDS03597 --nc-pos 1 --L 14 --flank-pos 45 \
    --context 50
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v5a_framework.pipeline import run_bag, PipelineConfig
from scripts.v5a_framework.multi_hit_retention import SPACER_N
from preprocess.alignment import revcomp


def _concat(nc_list: list[str]) -> str:
    out = nc_list[0]
    for extra in nc_list[1:]:
        out = out + "N" * SPACER_N + extra
    return out


def _match_line(a: str, b: str) -> str:
    """|  for match, .  for mismatch, N  for N vs anything."""
    out = []
    for x, y in zip(a, b):
        if x == "N" or y == "N": out.append("N")
        elif x == y: out.append("|")
        else: out.append(".")
    return "".join(out)


def _format_window(seq: str, lo: int, hi: int, hl_lo: int, hl_hi: int,
                     total_len: int) -> str:
    """Format a slice [lo:hi] with the sub-range [hl_lo:hl_hi] marked
    by [ ] on the display row."""
    lo = max(0, lo); hi = min(total_len, hi)
    pre_pad = max(0, hl_lo - lo)
    hl_len = min(hl_hi, hi) - hl_lo
    post = hi - min(hl_hi, hi)
    s = seq[lo:hi]
    # Mark positions with uppercase in bracketed range, lowercase outside
    out = (s[:pre_pad].lower() + "[" + s[pre_pad:pre_pad + hl_len].upper()
           + "]" + s[pre_pad + hl_len:].lower())
    return out, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--nc-pos", type=int, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--flank-pos", type=int, default=None,
                      help="target flank_pos (fwd-frame) to focus on; if omitted, inspect every cluster matching (nc_pos, L)")
    ap.add_argument("--max-offset", type=int, default=5)
    ap.add_argument("--context", type=int, default=50,
                      help="bp of context to extract on each side of the window")
    args = ap.parse_args()

    sites = []
    with args.jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            if r["transposase_id"] == args.bag:
                sites.append(r)
    if not sites:
        print(f"ERROR: bag {args.bag} not found"); return
    K = len(sites)
    print(f"# bag {args.bag}  K={K}")

    # Replay Phase A→D
    cfg = PipelineConfig()
    result = run_bag(args.bag, sites, cfg)
    print(f"# K_valid={result.K_valid}  "
          f"Phase-D-passing clusters: {len(result.passing)}  "
          f"Phase-D-failing: {len(result.failing)}")

    # Find cluster(s) at (nc_pos, L, flank_pos within ±max_offset)
    matches = []
    for cl, bg in result.passing + result.failing:
        if cl.nc_pos != args.nc_pos or cl.L != args.L:
            continue
        if args.flank_pos is not None and abs(cl.flank_pos_median - args.flank_pos) > args.max_offset:
            continue
        matches.append((cl, bg))
    if not matches:
        print(f"# no cluster at (nc_pos={args.nc_pos}, L={args.L}"
              + (f", flank_pos~{args.flank_pos}" if args.flank_pos is not None else "")
              + ")")
        return

    print(f"\n# {len(matches)} matching cluster(s)")
    for cl, bg in matches:
        print(f"\n## cluster: nc_pos={cl.nc_pos}  L={cl.L}  "
              f"flank_pos_median={cl.flank_pos_median}  mad={cl.flank_pos_mad:.1f}  "
              f"n_support={cl.n_support}  mean_m={cl.mean_m_frac:.3f}  "
              f"n_bg_groups={bg.n_local_bg_groups}  passes_D={bg.passes_bg_gate}")
        print(f"## supporting_sites: {sorted(ss.site_idx for ss in cl.supporting_sites)}")

        support_idx = {ss.site_idx: ss for ss in cl.supporting_sites}

        print(f"\n## PER-SITE nc + flank context (support marked with *)")
        for i in range(K):
            is_support = i in support_idx
            marker = "*" if is_support else " "
            nc = _concat(sites[i]["inputs"]["noncoding_regions"])
            flank = sites[i]["inputs"]["flank"]
            # Guide window in nc:
            g_lo = cl.nc_pos
            g_hi = cl.nc_pos + cl.L
            # Context:
            ctx_lo = max(0, g_lo - args.context)
            ctx_hi = min(len(nc), g_hi + args.context)
            guide = nc[g_lo:g_hi] if g_hi <= len(nc) else "(nc_too_short)"
            # Flank + target
            if is_support:
                ss = support_idx[i]
                h = ss.chosen_hit
                t_lo = h.flank_pos
                t_hi = h.flank_pos + cl.L
                tctx_lo = max(0, t_lo - args.context)
                tctx_hi = min(len(flank), t_hi + args.context)
                target = flank[t_lo:t_hi]
                # If RC orientation, the physical pairing is guide vs revcomp(target)
                if h.orient == "rc":
                    tgt_for_compare = revcomp(target)
                else:
                    tgt_for_compare = target
                mline = _match_line(guide.upper(), tgt_for_compare.upper())
                hit_desc = f"orient={h.orient}  flank_pos={h.flank_pos}  m={h.m}/{cl.L}  rank_in_site={ss.rank_within_site}"
            else:
                target = ""; tgt_for_compare = ""; mline = ""; hit_desc = "NOT SUPPORTING THIS CLUSTER"

            print(f"\n  {marker} site {i} ({hit_desc})")
            print(f"      nc[{ctx_lo}:{ctx_hi}]:    {nc[ctx_lo:g_lo].lower()}[{guide}]{nc[g_hi:ctx_hi].lower()}")
            if is_support:
                print(f"      flank[{tctx_lo}:{tctx_hi}]: {flank[tctx_lo:t_lo].lower()}[{target}]{flank[t_hi:tctx_hi].lower()}")
                print(f"      guide vs target: {guide.upper()}")
                print(f"      match line:      {mline}")
                print(f"      target (compared): {tgt_for_compare.upper()}  (orient={h.orient})")

        # Cross-site cross-nc alignment hint: for support sites, print just the
        # ±context nc context stacked, so a reviewer can eyeball column match
        print(f"\n## cross-site nc context stack (supporting sites only, oriented at nc_pos)")
        max_ctx = args.context
        print(f"  (showing nc[{cl.nc_pos - max_ctx}:{cl.nc_pos + cl.L + max_ctx}], "
              f"guide window in brackets)")
        for i in sorted(support_idx):
            nc = _concat(sites[i]["inputs"]["noncoding_regions"])
            g_lo = cl.nc_pos; g_hi = cl.nc_pos + cl.L
            lo = max(0, g_lo - max_ctx)
            hi = min(len(nc), g_hi + max_ctx)
            left_pad = " " * (max_ctx - (g_lo - lo))
            left  = nc[lo:g_lo]
            guide = nc[g_lo:g_hi] if g_hi <= len(nc) else ""
            right = nc[g_hi:hi]
            print(f"  site {i}: {left_pad}{left.lower()}[{guide}]{right.lower()}")


if __name__ == "__main__":
    main()
