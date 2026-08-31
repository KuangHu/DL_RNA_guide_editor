"""A+ — background competitor-density curves on 11 real ncRNAs.

For each of the ~11 supplementary ncRNA sequences (6 Durrant IS110/IS1111
consensus bridge RNAs + 5 seekRNA IS1111/IS110 NCRs), compute the
background competitor-density curve by pairing the ncRNA with random
draws from the 2,763-flank negative-family pool.

Metric per (ncRNA, planted_m) pair:
  competitor_count = #{ nc positions p : m_max(p, flank) >= planted_m }
  reported as median + p95 across n_draws random flanks

Cross-architecture invariance test: pick the shortest and longest
ncRNAs, match at planted_m = 8, compare their competitor_count. If
comparable at matched m, competitor_count is architecture-invariant
as the generator's orthogonality argument requires.

IUPAC handling: N in a ncRNA never matches — this is conservative
(under-counts on Durrant consensuses which have many Ns). To preserve
comparability, we use uppercase A/C/G/T only; every non-ACGT position
in the consensus is replaced with N and never matches.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocess.alignment import dot_plot, windowed_matches


L_DET = 11
SEED = 0
N_DRAWS = 200                             # random flanks per ncRNA


# --- ncRNAs from Durrant Supp Table 5 (6 consensus bridge RNAs) ---
DURRANT_SUPP_T5 = {
    "ISPpu10":  "nRCYUGnYGGGGYGGRGGGACnnnnnYYGnnUCUGRCYnnnnnRGYACAGACnGURGGAGAnnUCnYCCCRCCCCYCACCnnAGCGCCRAUAARGAAUYYAUCGGYnnRACCGACGAUnnnnRCAAGCCnGCGCnnnGGUGRRCCCCGRCARRYn",
    "ISAar29":  "GGCUGGUCYGGGACYGGCUGRCRRGRnnGRYRYYGGYCGYYYYRnYnnGRnYGUGACUCnnnnnnnnnnUGRCCYnnnnnRGUGYCYUnnnnnnGAnYUGUCCRnnYnnRRnnRGnYGGCCGRYRCCnnYCYGGCCCACYUCGGURRCYUGAUYAGAAGRYUGYCCRnCYYnnnnRRGnYGUGRCYCRUCACAGnnYUGYYCCGRRGUGAY",
    "ISHne5":   "RRYCRCYGnnnGnRRGYnRCnRCYYnCnnnCRGYGGYRRGGGnYGGAUCGnnRnnnYGCnGGYCRnnnnnnGGGCCGnGYUAGAGUnnGAUCRCCCYYRUCUUUYCCnnnnGRCCUGAACGGAUACYYGRGnnYnRGnCCCGnAUGACAAGCAnAGGnCRnGGAAAAGAUG",
    "ISCARN28": "nnnnRYYGAYnnGGYGAYGCGRYnYGRRYnnCGAYnRnRRnAAYCCUCGnnnnnnCUAYnnnGGCRnCYnnnnRGRYRAYCYnCGRYnYUAGARCGAGGCAGnYnnCRCGACGRAYYGAUGnAnnGYGGUAnCCAAYCCRCGRAUAYnAGYnUGAUYCAUCGnYGYnnnnnnnnYnnnnYnnnnnnnnnnnnnnnnnnnn",
    "ISAzs32":  "nYnnGnYnnnnnRGYnRnnnnRnRnYnYYnYCGYnGGGACGnnGGRYnnGGYGAnnYCGYnnnnRnnnYnGnnnnnnnnRnnnnnnnnnRAnnnCGYnnnnYAGAUYGRnnCGCCnYRYYYnYCYGAYnnCAUnRUGYGGCGGCYnnRYGCCGACCRCGRAnRGAAGCRnGnRnYCnRCGRnRnGnYn",
    "ISPa11":   "nYGAUUGCnRRGYRnnnnnRnnnUGAUGGCnRRAYnGGUYRRACCGnnRYnnRnnnARCCYGnnnnnnnYnnnGnnCnnYRAGnnCGnnnnnnYGRUnRGRnnYnnRYnnGCRRAnYCCAUCAnGGnYnGnRnYnnnnnRnYnCRnnnRnARnCCGRAURUAYGnnnGCARUCnYnnCYnnnnnRYYRnnRnnnnnnnnnnRCUnGCAAnnnG",
}

# --- ncRNAs from seekRNA Supp Table 1 (5 NCRs, IS1111 x 4 + IS110 x 1) ---
SEEKRNA_SUPP_T1 = {
    "ISEc11_IS1111": (
        "ACGCTGATACCATTAAACAATGAACTCTTAACAAAAGGGTGAATGCTGAAAGG"
        "TTGCTATGGCGGCCAGAGTGATGACAAAGACAGGTAAGACCGTGACTCACTAA"
        "ACCTGAACAGTATTTTGGGCTTGAAGTCCGCCGTGAAAATAAGGGGTGAGTCG"
        "GCGAATTACATAGGGGCTCGCAGCGTTACGGCTGCAATAAAGCCGGATATAAA"
        "GCTGCAACCTACCCGTCATGTCAAAACAATGGATGCCTTGCAAACGGGATGCG"
        "TTCATATA"
    ),
    "ISKpn4_IS1111": (
        "GGGCGGGGGTAATCAGCAAGACGAGGTGAACTTCCTGCGGTTGTGGTGAGAA"
        "GCGAGACTGTCGGAACAGGCGAGTCCTGCAGCGGAACAAGGCCGATAACAAT"
        "CATGATCCTCGGGATCGTTTGAACGCTTGGCCCCCGCTGCGCGAACTACAGAA"
        "TGGCCCGGGCACTAGCCCACGCAAGGCCGGATATGCGATTGCAACCGCGCTGA"
        "CGACAGGATCGAAAAAAGCTTTCCTCACTGCTTGTGGGGAGAGTCCATATG"
    ),
    "ISPst6_IS1111": (
        "CGGCGAGGTAAATAATGAGGTGAGTGTCTAGTGGTTGTGGTGAGAAGCGAGA"
        "CTGTCGGAACAGGCGAGTCCTGCAGCGGAACAAGGCCGATAACAATGATGAT"
        "CTTCGTGATCGTTTGAACGCTTGGCCCCTGCTGCGCGAACTACAGAATGGCCA"
        "GGGCAATGCCCAGAACAGGCCGGATATACGAATGCAACCGCGCTGACGACAG"
        "GATCGAAAAAAGCTATTCCTCACTGCTTGTGGGGAGAGTCCATATA"
    ),
    "ISPa11_IS1111_seek": (
        "TCGTTCCTGCCACACCGTAGTTGAAACATCCACCACGATTGCTCAGTGAATGAC"
        "AATGATGACGAACCGGTCGAACCGGCCTGCATGAAACCTGGTTTATACGTGGG"
        "CTCCCTGCTGCAGTGAAGCAAAGCCGTTAGGGCGATCAGGTATGCAGGCGCGC"
        "ATTTCATCAGGGCTCGGGAGTTGCAACACCACTCCATGAAGCCGGATATACGG"
        "ATGCAGTCGTACACAGGTTTGAAATCAAGACAAACACTGGCAAACCGGGAGG"
        "AGTCCATATA"
    ),
    "ISEc21_IS110": (
        "AGTAATAATGCCGGTATCAGTTTTTATCATCACTCTGTTTGCTGTTTAACCAGA"
        "CTGGTGTGATTACTGATGCAGTGAAGACCTTCCCGCATCCTGACTCACACAGC"
        "GATCGACCCTTTGTGTCCTGCCCTGGACCTGTCGGTTGCCGGAAGCGCCTTCAT"
        "GCGAGGCGTCTCCTCACCGATGCGCGTGACTCAAGAAGGGCCTGACGGTTTGT"
        "CTCGTTACTGTCCTGTCCGGGTTATCTGTCTGGAGATTCAACTCTGTTTCCTCAC"
        "AGGAGCTCTGTT"
    ),
}


def normalize_ncrna(seq: str) -> str:
    """RNA -> DNA; keep A/C/G/T, replace everything else with N."""
    s = seq.upper().replace("U", "T")
    return "".join(b if b in "ACGT" else "N" for b in s)


def load_flank_pool() -> list[str]:
    """Load the 2,763-family negative-family downstream flanks (already
    verified as real bacterial sequences in earlier probes)."""
    pool = []
    for fam in ("IS10-R", "IS30", "IS903", "ISAjo2", "ISLdl1"):
        p = f"/global/scratch/users/kh36969/DL_novel_guide_editor/real_data/formatted/real_{fam}_sites.jsonl"
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                m = d.get("generator_metadata", {})
                if m.get("flank_side") != "downstream":
                    continue
                fl = d.get("inputs", {}).get("flank")
                if fl and len(fl) == 120:
                    pool.append(fl.upper())
    return pool


def competitor_curve_for_ncrna(nc: str, flanks: list[str], n_draws: int,
                                 seed: int, m_range=range(6, 13),
                                 L: int = L_DET) -> dict:
    """For a given ncRNA + a sample of flanks, compute competitor_count at
    each planted_m in m_range. Returns dict[m] -> (median, p95, mean, p5)
    across draws. competitor_count is #{nc positions : m_max(p) >= m}.
    """
    rng = np.random.default_rng(seed)
    if len(flanks) > n_draws:
        idx = rng.choice(len(flanks), size=n_draws, replace=False)
        sampled = [flanks[i] for i in idx]
    else:
        sampled = flanks

    per_m_counts: dict[int, list[int]] = {m: [] for m in m_range}
    for fl in sampled:
        fwd_dot, rc_dot = dot_plot(nc, fl)
        w_f = windowed_matches(fwd_dot, L)
        w_r = windowed_matches(rc_dot, L)
        if w_f.size == 0 or w_r.size == 0:
            continue
        n_pos = min(w_f.shape[0], w_r.shape[0])
        m_pooled = np.maximum(w_f.max(axis=1)[:n_pos], w_r.max(axis=1)[:n_pos])
        for m in m_range:
            per_m_counts[m].append(int((m_pooled >= m).sum()))
    return {
        m: {
            "n_draws": len(counts),
            "median": float(np.median(counts)),
            "p5": float(np.percentile(counts, 5)),
            "p95": float(np.percentile(counts, 95)),
            "mean": float(np.mean(counts)),
        }
        for m, counts in per_m_counts.items() if counts
    }


def main() -> int:
    print("[A+] Loading 2,763 real-bacterial-flank pool from 5 negative families...", flush=True)
    flanks = load_flank_pool()
    print(f"[A+] pool size: {len(flanks)} downstream flanks (120 nt each)", flush=True)

    all_ncrnas = {}
    for name, seq in DURRANT_SUPP_T5.items():
        norm = normalize_ncrna(seq)
        all_ncrnas[f"Durrant_{name}"] = norm
    for name, seq in SEEKRNA_SUPP_T1.items():
        norm = normalize_ncrna(seq)
        all_ncrnas[f"seekRNA_{name}"] = norm

    print()
    print(f"{'ncRNA':<30s} {'raw len':>8s} {'N count':>8s} {'ACGT len':>9s}")
    for name, seq in all_ncrnas.items():
        raw = len(seq)
        n_count = seq.count("N")
        print(f"  {name:<28s} {raw:>8d} {n_count:>8d} {raw - n_count:>9d}")

    # Add Durrant's actual T-WT bridge RNA from the framework MatchTable
    # for a n=1 experimental anchor within the same code.
    print()
    print("[A+] Loading Durrant T-WT bridge RNA as experimental n=1 anchor...", flush=True)
    from scripts.v5a_framework.match_table import load as load_mt
    mt = load_mt("/global/scratch/users/kh36969/DL_novel_guide_editor/v5a_framework_cache/durrant_positive")
    twt = [t for t in mt.tnp_ids if t.startswith("durrant_bridge_RNA_T-WT_D-WT")][0]
    twt_nc = mt.tnps[twt].nc.upper().replace("U", "T")
    all_ncrnas["Durrant_T-WT_anchor"] = "".join(b if b in "ACGT" else "N" for b in twt_nc)
    print(f"[A+] T-WT nc len={len(all_ncrnas['Durrant_T-WT_anchor'])}", flush=True)

    print()
    print("[A+] Computing background competitor-density curves...", flush=True)
    import time
    t0 = time.time()
    curves: dict[str, dict] = {}
    for name, nc in all_ncrnas.items():
        curves[name] = competitor_curve_for_ncrna(nc, flanks, N_DRAWS,
                                                    seed=hash((SEED, name)) & 0xFFFFFFFF)
    print(f"[A+] wall {time.time()-t0:.1f}s", flush=True)

    # Report competitor-count(m) per ncRNA
    print()
    print("=== Competitor count vs planted_m (median across draws) ===")
    print(f"{'ncRNA':<30s} {'len':>4s} " + " ".join(f"m={m:>2d}" for m in range(6, 13)))
    for name, curve in curves.items():
        row = f"  {name:<28s} {len(all_ncrnas[name]):>4d} "
        for m in range(6, 13):
            if m in curve:
                row += f"{curve[m]['median']:>5.0f}"
            else:
                row += f"    -"
        print(row)

    # Cross-architecture invariance: pick min and max length ncRNAs among
    # A/C/G/T-full-defined ones (seekRNAs), match at m=8
    print()
    print("=== Cross-architecture invariance at m=8 ===")
    seek_names = [n for n in curves if n.startswith("seekRNA_")]
    seek_lens = {n: len(all_ncrnas[n]) for n in seek_names}
    shortest = min(seek_lens, key=seek_lens.get)
    longest = max(seek_lens, key=seek_lens.get)
    print(f"  shortest: {shortest} ({seek_lens[shortest]} nt)")
    print(f"    competitor_count(m=8) median = {curves[shortest][8]['median']:.1f}")
    print(f"  longest:  {longest} ({seek_lens[longest]} nt)")
    print(f"    competitor_count(m=8) median = {curves[longest][8]['median']:.1f}")
    ratio_length = seek_lens[longest] / seek_lens[shortest]
    ratio_competitor = curves[longest][8]['median'] / max(1, curves[shortest][8]['median'])
    print(f"  length ratio: {ratio_length:.2f}x")
    print(f"  competitor-count ratio: {ratio_competitor:.2f}x")
    print(f"  If competitor_count is architecture-invariant at MATCHED planted_m,")
    print(f"  the ratio should track the length ratio (search-space scaling).")
    print(f"  Deviation from length ratio would indicate architecture-dependent difficulty.")

    # T-WT anchor summary
    print()
    print("=== T-WT experimental anchor (n=1) at m=8 vs the 11 supplementary systems ===")
    twt_m8 = curves["Durrant_T-WT_anchor"][8]
    print(f"  T-WT (177 nt bridge RNA): competitor_count(m=8) median = {twt_m8['median']:.1f}, p95 = {twt_m8['p95']:.1f}")
    supp_m8_medians = [curves[n][8]['median'] for n in curves
                        if n != "Durrant_T-WT_anchor" and 8 in curves[n]]
    print(f"  Supplementary ncRNAs (11 systems): median across systems = {float(np.median(supp_m8_medians)):.1f}, "
          f"range [{min(supp_m8_medians):.1f}, {max(supp_m8_medians):.1f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
