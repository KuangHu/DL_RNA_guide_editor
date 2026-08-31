"""D1 + D2 + D3 + pf() bench — decides scope of BPP migration.

D1: T-WT 11-nt MFE loop — does it coincide with annotated guide start (nc=49)?
    If yes: strict-loop rule was correct for T-WT; 4/6 failure is confound 1.
    If no:  strict rule was wrong for T-WT too; must move to accessibility-window.

D2: Refold seekRNA precursor vs a mature-fragment trim.
    ISEc21 mature large seekRNA = nc[7:179] (172 nt) per user's coord.
    For the other 4 seekRNAs (no published mature coord), sweep a symmetric
    145-nt center window as a proxy for the ~145-170 nt mature form.
    Compare loop windows on precursor vs mature.

D3: Max base-pair span distribution — global MFE on all 6 real ncRNAs.
    If max-span > 60 nt widely, the current -L 60 in structure_v42_min
    is blind to outer helices and cache needs rebuilding at global params.

Bench: RNA.fold_compound.pf() + bpp() cost at 177-281 nt.
    Compared to MFE from A1. Determines feasibility of per-bag BPP.

Run: python -m scripts.generator_v5.tests.test_bpp_diagnostics
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np
import RNA

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.v5a_framework.match_table import load as load_mt
from scripts.v5a_framework.tests.aplus_calibration import SEEKRNA_SUPP_T1, normalize_ncrna

MT_POS = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5a_framework_cache/durrant_positive"
GUIDE_L = 11


# ---------- shared helpers ----------

def fold_global(dna: str) -> tuple[str, float]:
    """Global MFE fold on DNA input (converts T->U)."""
    seq = dna.upper().replace("T", "U").replace("N", "N")
    # ViennaRNA tolerates N as fully-ambiguous; keep as-is
    fc = RNA.fold_compound(seq)
    structure, energy = fc.mfe()
    return structure, energy


def loop_runs(structure: str) -> list[tuple[int, int]]:
    """(start, length) for each maximal contiguous run of '.' in dot-bracket."""
    runs = []
    i, n = 0, len(structure)
    while i < n:
        if structure[i] == ".":
            j = i
            while j < n and structure[j] == ".":
                j += 1
            runs.append((i, j - i))
            i = j
        else:
            i += 1
    return runs


def max_pair_span(structure: str) -> int:
    """Longest |i-j| for a paired position, from dot-bracket."""
    stack: list[int] = []
    max_span = 0
    for i, c in enumerate(structure):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if stack:
                j = stack.pop()
                span = i - j
                if span > max_span:
                    max_span = span
    return max_span


# ---------- D1 ----------

def d1_twt_loop_at_49() -> None:
    print("=== D1: T-WT MFE loop at annotated guide start ===")
    mt = load_mt(MT_POS)
    twt_id = next(t for t in mt.tnp_ids if t.startswith("durrant_bridge_RNA_T-WT_D-WT"))
    nc = mt.tnps[twt_id].nc.upper().replace("U", "T")
    gold_pos = [s.gold_nc for s in mt.tnps[twt_id].sites]
    gold_med = int(np.median(gold_pos))
    print(f"  T-WT nc length: {len(nc)}")
    print(f"  gold nc positions (from MatchTable sites): "
          f"median={gold_med}, min={min(gold_pos)}, max={max(gold_pos)}, n={len(gold_pos)}")
    structure, energy = fold_global(nc)
    print(f"  MFE ΔG = {energy:.2f} kcal/mol")
    print(f"  structure ({len(structure)} nt):")
    for k in range(0, len(structure), 60):
        idx = "".join(str((k + i) // 10 % 10) if (k + i) % 10 == 0 else "." for i in range(min(60, len(structure) - k)))
        print(f"    {k:>4d} {structure[k:k+60]}")
    runs = loop_runs(structure)
    long_runs = [(s, L) for s, L in runs if L >= GUIDE_L]
    print(f"  # loop runs >= {GUIDE_L} nt: {len(long_runs)}")
    for s, L in long_runs:
        print(f"    run [{s}:{s+L}] length={L}")
    span = max_pair_span(structure)
    print(f"  max base-pair span: {span} nt")
    # Check gold overlap
    def overlap(rs, rl, g):
        return rs <= g and (g + GUIDE_L) <= rs + rl
    hits = [(s, L) for s, L in long_runs if overlap(s, L, gold_med)]
    print(f"  gold-position {gold_med}..{gold_med+GUIDE_L} inside any qualifying loop run? "
          f"{'YES: ' + str(hits) if hits else 'NO'}")
    if not hits:
        # How structured is the actual gold window?
        gw = structure[gold_med : gold_med + GUIDE_L]
        ss_frac = gw.count(".") / GUIDE_L
        print(f"  gold window structure: {gw}   ss_frac={ss_frac:.2f}")


# ---------- D2 ----------

def d2_seekrna_mature_vs_precursor() -> None:
    print()
    print("=== D2: seekRNA precursor vs mature-fragment fold ===")
    coords = {
        # ISEc21: large seekRNA = 8-179 per user (1-indexed inclusive) -> py slice [7:179]
        "ISEc21_IS110": (7, 179),
    }
    for name, dna in SEEKRNA_SUPP_T1.items():
        seq = dna.upper()
        print(f"\n  --- {name} (precursor {len(seq)} nt) ---")
        struct_pre, e_pre = fold_global(seq)
        runs_pre = [(s, L) for s, L in loop_runs(struct_pre) if L >= GUIDE_L]
        span_pre = max_pair_span(struct_pre)
        ss_pre = struct_pre.count(".") / len(struct_pre)
        print(f"    precursor: ΔG={e_pre:.2f}, #loops>={GUIDE_L}: {len(runs_pre)}, "
              f"max_span={span_pre}, ss_frac={ss_pre:.2f}")

        if name in coords:
            lo, hi = coords[name]
            trims = [(f"mature[{lo}:{hi}]", seq[lo:hi])]
        else:
            # No published mature coord; sweep 3 candidate windows in the 70-172 nt range
            L = len(seq)
            trims = []
            for target_len, tag in [(145, "mid145"), (100, "small100"), (70, "small70")]:
                if L >= target_len:
                    mid = (L - target_len) // 2
                    trims.append((tag, seq[mid : mid + target_len]))
        for tag, sub in trims:
            struct, e = fold_global(sub)
            runs_sub = [(s, L2) for s, L2 in loop_runs(struct) if L2 >= GUIDE_L]
            span = max_pair_span(struct)
            ss = struct.count(".") / len(struct)
            print(f"    {tag} ({len(sub)} nt): ΔG={e:.2f}, #loops>={GUIDE_L}: {len(runs_sub)}, "
                  f"max_span={span}, ss_frac={ss:.2f}")


# ---------- D3 ----------

def d3_pair_span_distribution() -> None:
    print()
    print("=== D3: max base-pair span distribution (global MFE) ===")
    # T-WT
    mt = load_mt(MT_POS)
    twt_id = next(t for t in mt.tnp_ids if t.startswith("durrant_bridge_RNA_T-WT_D-WT"))
    nc = mt.tnps[twt_id].nc.upper().replace("U", "T")
    all_ncs = {"T-WT": nc}
    for k, v in SEEKRNA_SUPP_T1.items():
        all_ncs[k] = v
    print(f"  {'ncRNA':<24s} {'len':>4s} {'max_span':>9s} {'>60?':>5s} {'>=len/2?':>10s}")
    spans = []
    for name, seq in all_ncs.items():
        struct, _ = fold_global(seq)
        s = max_pair_span(struct)
        print(f"    {name:<22s} {len(seq):>4d} {s:>9d} {'yes' if s>60 else 'no':>5s} "
              f"{'yes' if s>=len(seq)//2 else 'no':>10s}")
        spans.append((name, len(seq), s))
    if spans:
        pcts = [s / L for _, L, s in spans]
        print(f"  overall: median max_span/nc_len = {statistics.median(pcts):.2f}, "
              f"mean = {statistics.mean(pcts):.2f}")
        n_over_60 = sum(1 for _, _, s in spans if s > 60)
        print(f"  {n_over_60}/{len(spans)} ncRNAs have max pair span > 60 (would be lost at -L 60)")


# ---------- pf() bench ----------

def bench_pf(n_per_length: int = 30) -> None:
    print()
    print("=== Bench: RNA.fold_compound.pf() cost at 177-281 nt ===")
    import random
    rng = random.Random(0)
    lengths = [177, 200, 225, 250, 281]
    print(f"  {'length':<7s} {'mfe_ms':>8s} {'pf_ms':>8s} {'bpp_ms':>8s} {'per_bag_ms':>12s}")
    for L in lengths:
        mfe_costs, pf_costs, bpp_costs = [], [], []
        for _ in range(n_per_length):
            seq = "".join(rng.choices("ACGU", k=L))
            # MFE (fresh compound to be fair)
            fc = RNA.fold_compound(seq)
            t0 = time.perf_counter()
            fc.mfe()
            mfe_costs.append((time.perf_counter() - t0) * 1000)
            # PF (needs pf() called before bpp())
            fc2 = RNA.fold_compound(seq)
            t0 = time.perf_counter()
            fc2.pf()
            pf_costs.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            fc2.bpp()
            bpp_costs.append((time.perf_counter() - t0) * 1000)
        mfe = statistics.mean(mfe_costs)
        pf = statistics.mean(pf_costs)
        bpp = statistics.mean(bpp_costs)
        print(f"  {L:<7d} {mfe:>8.2f} {pf:>8.2f} {bpp:>8.2f} {mfe+pf+bpp:>12.2f}")
    print()
    # 50K bags cost extrapolation at worst length
    print(f"  extrapolation: 50K bags at 281 nt: "
          f"~{(mfe+pf+bpp) * 50000 / 60000:.1f} min = {(mfe+pf+bpp) * 50000 / 3.6e6:.2f} h")


def main() -> int:
    d1_twt_loop_at_49()
    d2_seekrna_mature_vs_precursor()
    d3_pair_span_distribution()
    bench_pf(n_per_length=30)
    return 0


if __name__ == "__main__":
    sys.exit(main())
