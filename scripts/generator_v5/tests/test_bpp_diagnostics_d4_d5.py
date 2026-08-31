"""D4 + D5 — decides θ's role and structure-channel weight.

D4: Global-fold accuracy vs Durrant Supp T5 published dot-bracket
    consensus structures. For each of 6 consensus (sequence + structure),
    sample N IUPAC realizations, fold, compute F1 on paired positions
    against the published dot-bracket. Median F1 across samples per
    consensus; overall aggregate.
    - If F1 >= 0.75: structure prediction is reliable enough as a hard input.
    - 0.55-0.75: use as soft prior with moderate weight.
    - < 0.55: structure channel is weak prior, θ must be soft preference.

D5: T-WT gold-window accessibility percentile.
    For T-WT (nc, 177 nt), compute BPP -> P_ss[i] per position ->
    mean P_ss over each L=11 window on nc. Rank the annotated gold
    window (start=49) among all 167 possible windows.
    Also compute the same for ISEc21 mature[7:179] at its target-
    recognition region (approx positions from user coord).
    - Gold at >= 90th percentile: θ can be a hard threshold.
    - 60-90th: soft preference (Gaussian prior around percentile).
    - < 60th: guide placement rule cannot be based on accessibility alone.

Both diagnostics use RNA.fold_compound.pf() + .bpp().
"""
from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

import numpy as np
import openpyxl
import RNA

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.v5a_framework.match_table import load as load_mt

MT_POS = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5a_framework_cache/durrant_positive"
SUPP_T5 = ("/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/"
           "annotation/2023-09-16026B-s3/2023-09-16026B-SupplementaryTable5.xlsx")
GUIDE_L = 11
N_SAMPLES_PER_CONSENSUS = 30
SEED = 0

IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T", "U": "T",
    "R": "AG", "Y": "CT", "S": "GC", "W": "AT", "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG",
    "N": "ACGT", "n": "ACGT",
}


# ---------- helpers ----------

def iupac_sample(consensus: str, rng: random.Random) -> str:
    """Realize an IUPAC consensus by uniform draw at each ambiguous position."""
    seq = []
    for c in consensus:
        if c in "acgtu":
            c_up = c.upper()
        else:
            c_up = c
        options = IUPAC.get(c_up)
        if options is None:
            options = "ACGT"
        seq.append(rng.choice(options))
    return "".join(seq)


def pairs_from_structure(structure: str) -> set[tuple[int, int]]:
    """Return set of (i, j) with i < j from dot-bracket structure (0-indexed)."""
    stack = []
    pairs = set()
    for i, c in enumerate(structure):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if stack:
                j = stack.pop()
                pairs.add((j, i))
    return pairs


def f1_pairs(pred: set[tuple[int, int]], ref: set[tuple[int, int]]) -> float:
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    tp = len(pred & ref)
    fp = len(pred - ref)
    fn = len(ref - pred)
    return 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 0.0


def bpp_from_fc(fc) -> np.ndarray:
    """Return symmetric BPP as np.ndarray[n,n]. ViennaRNA .bpp() is 1-indexed
    and returns [(i,j,prob), ...] as (n+1, n+1) list."""
    fc.pf()
    bpp = fc.bpp()
    n = len(bpp) - 1
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n + 1):
        row = bpp[i]
        for j in range(i + 1, n + 1):
            p = row[j]
            if p:
                mat[i - 1, j - 1] = p
                mat[j - 1, i - 1] = p
    return mat


def p_ss_per_position(bpp_mat: np.ndarray) -> np.ndarray:
    """P(unpaired i) = 1 - sum_j BPP[i,j]. Clipped to [0, 1]."""
    return np.clip(1.0 - bpp_mat.sum(axis=1), 0.0, 1.0)


def dg_open_u1(bpp_mat: np.ndarray, kT: float = 0.616) -> np.ndarray:
    """dG_open at u=1 (kcal/mol). kT at 37 C ≈ 0.616."""
    p = p_ss_per_position(bpp_mat)
    p = np.clip(p, 1e-10, 1.0)
    return -kT * np.log(p)


def window_mean(arr: np.ndarray, L: int) -> np.ndarray:
    """Return array of length len(arr)-L+1 with means over each L-window."""
    if len(arr) < L:
        return np.zeros(0)
    csum = np.cumsum(arr, dtype=np.float64)
    csum = np.concatenate(([0.0], csum))
    return (csum[L:] - csum[:-L]) / L


# ---------- D4 ----------

def d4_prediction_accuracy() -> None:
    print("=== D4: fold accuracy vs Durrant Supp T5 published dot-brackets ===")
    wb = openpyxl.load_workbook(SUPP_T5, data_only=True)
    ws = wb["6 RNA Structure Cons. Sequences"]
    rows = list(ws.iter_rows(values_only=True))
    # Skip header
    consensuses = []
    for r in rows[1:]:
        name, cons_seq, cons_struct = r[0], r[1], r[2]
        if name and cons_seq and cons_struct:
            consensuses.append((name, cons_seq, cons_struct))
    rng = random.Random(SEED)
    all_f1s: list[float] = []
    print(f"  {'ncRNA':<10s} {'len':>4s} {'N samples':>10s} "
          f"{'F1_med':>8s} {'F1_mean':>8s} {'F1_min':>7s} {'F1_max':>7s} "
          f"{'ref_pairs':>10s} {'pred_pair_med':>14s}")
    for name, cons_seq, cons_struct in consensuses:
        L = len(cons_seq)
        assert L == len(cons_struct), f"length mismatch on {name}"
        ref_pairs = pairs_from_structure(cons_struct)
        f1s: list[float] = []
        pred_pair_counts: list[int] = []
        for _ in range(N_SAMPLES_PER_CONSENSUS):
            realized = iupac_sample(cons_seq, rng)
            rna = realized.upper().replace("T", "U")
            fc = RNA.fold_compound(rna)
            pred_struct, _ = fc.mfe()
            pred_pairs = pairs_from_structure(pred_struct)
            f1s.append(f1_pairs(pred_pairs, ref_pairs))
            pred_pair_counts.append(len(pred_pairs))
        med = statistics.median(f1s)
        mean = statistics.mean(f1s)
        print(f"  {name:<10s} {L:>4d} {len(f1s):>10d} "
              f"{med:>8.3f} {mean:>8.3f} {min(f1s):>7.3f} {max(f1s):>7.3f} "
              f"{len(ref_pairs):>10d} {int(statistics.median(pred_pair_counts)):>14d}")
        all_f1s.extend(f1s)
    print()
    if all_f1s:
        overall_med = statistics.median(all_f1s)
        overall_mean = statistics.mean(all_f1s)
        print(f"  overall (across all {len(all_f1s)} realizations): median F1 = {overall_med:.3f}, "
              f"mean = {overall_mean:.3f}")
        if overall_med >= 0.75:
            verdict = "HIGH: structure prediction is a reliable input, hard θ acceptable"
        elif overall_med >= 0.55:
            verdict = "MEDIUM: soft prior, moderate weight on structure channels"
        else:
            verdict = "LOW: weak prior, θ must be soft preference and structure weight small"
        print(f"  verdict: {verdict}")


# ---------- D5 ----------

def _target_recognition_region_ise21() -> tuple[int, int, int]:
    """ISEc21 mature large seekRNA: nc[7:179] (per user's coord). Guide is
    somewhere within this; use the mid-third as an approximation of the
    target-recognition region (~pos 90-110 in nc coords)."""
    return (7, 179, 90)


def d5_gold_window_percentile() -> None:
    print()
    print("=== D5: T-WT and ISEc21 gold-window accessibility percentile ===")

    # T-WT
    mt = load_mt(MT_POS)
    twt_id = next(t for t in mt.tnp_ids if t.startswith("durrant_bridge_RNA_T-WT_D-WT"))
    twt_nc = mt.tnps[twt_id].nc.upper().replace("U", "T")
    twt_gold_starts = [s.gold_nc for s in mt.tnps[twt_id].sites]
    gold_pos = int(np.median(twt_gold_starts))
    print(f"\n  T-WT nc len {len(twt_nc)}, gold pos {gold_pos} (n={len(twt_gold_starts)} sites)")

    fc = RNA.fold_compound(twt_nc.replace("T", "U"))
    bpp_mat = bpp_from_fc(fc)
    p_ss = p_ss_per_position(bpp_mat)
    dg = dg_open_u1(bpp_mat)
    # window means (higher P_ss / lower dG = more accessible)
    p_ss_win = window_mean(p_ss, GUIDE_L)
    dg_win = window_mean(dg, GUIDE_L)
    n_win = len(p_ss_win)

    gold_p = float(p_ss_win[gold_pos])
    gold_dg = float(dg_win[gold_pos])
    p_rank = float((p_ss_win < gold_p).sum() / n_win)   # accessibility rank
    dg_rank = float((dg_win > gold_dg).sum() / n_win)   # low dG = high rank

    print(f"    n_windows (L={GUIDE_L}) = {n_win}")
    print(f"    gold window mean P_ss   = {gold_p:.3f}  (percentile: {p_rank * 100:.1f}%)")
    print(f"    gold window mean dG     = {gold_dg:.3f}  (percentile: {dg_rank * 100:.1f}%)")
    print(f"    P_ss stats: min={p_ss_win.min():.3f}, p25={np.percentile(p_ss_win, 25):.3f}, "
          f"median={np.median(p_ss_win):.3f}, p75={np.percentile(p_ss_win, 75):.3f}, "
          f"p90={np.percentile(p_ss_win, 90):.3f}, max={p_ss_win.max():.3f}")

    # Top windows and where gold sits among them
    top_by_pss = np.argsort(-p_ss_win)[:10]
    print(f"    top 10 windows by P_ss: {top_by_pss.tolist()}")
    print(f"    gold at start {gold_pos} rank among all windows: "
          f"{int((p_ss_win >= gold_p).sum())} / {n_win}")

    # ISEc21 mature
    from scripts.v5a_framework.tests.aplus_calibration import SEEKRNA_SUPP_T1
    ise21 = SEEKRNA_SUPP_T1["ISEc21_IS110"].upper()
    lo, hi, target_pos_full = _target_recognition_region_ise21()
    mature = ise21[lo:hi]
    # Position in mature coords
    target_pos_mature = target_pos_full - lo
    print(f"\n  ISEc21 mature[{lo}:{hi}] len {len(mature)}, "
          f"approx target region start (mature coord) = {target_pos_mature}")
    fc2 = RNA.fold_compound(mature.replace("T", "U"))
    bpp2 = bpp_from_fc(fc2)
    p_ss2 = p_ss_per_position(bpp2)
    p_ss_win2 = window_mean(p_ss2, GUIDE_L)
    n2 = len(p_ss_win2)
    if 0 <= target_pos_mature < n2:
        gp = float(p_ss_win2[target_pos_mature])
        rk = float((p_ss_win2 < gp).sum() / n2)
        print(f"    n_windows = {n2}")
        print(f"    target-region window mean P_ss = {gp:.3f}  (percentile: {rk * 100:.1f}%)")
        top_by_pss2 = np.argsort(-p_ss_win2)[:10]
        print(f"    top 10 windows by P_ss: {top_by_pss2.tolist()}")
    else:
        print(f"    target_pos_mature = {target_pos_mature} outside window range")

    print()
    print("  === Interpretation ===")
    if p_rank >= 0.90:
        print(f"    T-WT gold at {p_rank*100:.0f}%ile: hard θ acceptable (guides ARE at accessibility peaks)")
    elif p_rank >= 0.60:
        print(f"    T-WT gold at {p_rank*100:.0f}%ile: soft preference (Gaussian around this percentile)")
    else:
        print(f"    T-WT gold at {p_rank*100:.0f}%ile: accessibility is NOT the placement rule; "
              f"need geometry (E_span/H_pair) beyond mean(P_ss)")


def main() -> int:
    d4_prediction_accuracy()
    d5_gold_window_percentile()
    return 0


if __name__ == "__main__":
    sys.exit(main())
