"""D4' — separate MFE-inherent error (a) from IUPAC realization error (b).

Same as D4 but with base-pair-constrained IUPAC sampling: at each (i, j)
in the published dot-bracket consensus, sample a valid RNA pair
(AU/UA/GC/CG/GU/UG) from IUPAC[c_i] x IUPAC[c_j] intersected with the
valid pair set. Unpaired positions still uniform over IUPAC[c].

If F1' is much higher than the F1=0.16 from D4, (b) was the main driver
and structure prediction is a MEDIUM prior (0.55-0.75 → weight matters).
If F1' stays low, (a) dominates and structure channels are truly weak.

Reports:
  - F1' median/mean per consensus
  - fraction of consensus pairs that are unrealizable at the IUPAC codes
    (impossible pair sets — the intersection can be empty; must skip)
"""
from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

import openpyxl
import RNA

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.generator_v5.tests.test_bpp_diagnostics_d4_d5 import (
    IUPAC, pairs_from_structure, f1_pairs,
)

SUPP_T5 = ("/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/"
           "annotation/2023-09-16026B-s3/2023-09-16026B-SupplementaryTable5.xlsx")
N_SAMPLES = 30
SEED = 0

VALID_PAIRS_DNA = {
    ("A", "T"), ("T", "A"),
    ("G", "C"), ("C", "G"),
    ("G", "T"), ("T", "G"),   # wobble
}


def constrained_realize(cons_seq: str, cons_struct: str, rng: random.Random
                        ) -> tuple[str, int]:
    """Return (realized DNA seq, n_pairs_skipped). Paired positions get
    a mutually-consistent pair; unpaired positions get uniform IUPAC.
    A pair is 'skipped' if IUPAC[c_i] x IUPAC[c_j] has no valid pair —
    those positions fall back to independent uniform IUPAC (which will
    almost never form a pair)."""
    L = len(cons_seq)
    pairs = pairs_from_structure(cons_struct)
    seq = [""] * L

    n_skipped = 0
    for i, j in pairs:
        c_i = cons_seq[i].upper().replace("U", "T")
        c_j = cons_seq[j].upper().replace("U", "T")
        opts_i = IUPAC.get(c_i, "ACGT")
        opts_j = IUPAC.get(c_j, "ACGT")
        valid_pairs = [(x, y) for x in opts_i for y in opts_j
                       if (x, y) in VALID_PAIRS_DNA]
        if valid_pairs:
            x, y = rng.choice(valid_pairs)
            seq[i] = x
            seq[j] = y
        else:
            n_skipped += 1
            seq[i] = rng.choice(opts_i)
            seq[j] = rng.choice(opts_j)

    # Unpaired positions
    for k in range(L):
        if seq[k] == "":
            c = cons_seq[k].upper().replace("U", "T")
            opts = IUPAC.get(c, "ACGT")
            seq[k] = rng.choice(opts)

    return "".join(seq), n_skipped


def main() -> int:
    print("=== D4': constrained IUPAC (base-pair complementarity forced) ===")
    wb = openpyxl.load_workbook(SUPP_T5, data_only=True)
    ws = wb["6 RNA Structure Cons. Sequences"]
    rng = random.Random(SEED)
    consensuses = []
    for r in list(ws.iter_rows(values_only=True))[1:]:
        if r[0] and r[1] and r[2]:
            consensuses.append((r[0], r[1], r[2]))

    print(f"  {'ncRNA':<10s} {'len':>4s} {'ref_pairs':>10s} "
          f"{'unrealize/pair':>15s} "
          f"{'F1_med':>8s} {'F1_mean':>8s} {'F1_min':>7s} {'F1_max':>7s}")
    all_f1s = []
    for name, cons_seq, cons_struct in consensuses:
        ref_pairs = pairs_from_structure(cons_struct)
        f1s: list[float] = []
        skipped_counts = []
        for _ in range(N_SAMPLES):
            realized, n_skipped = constrained_realize(cons_seq, cons_struct, rng)
            rna = realized.replace("T", "U")
            fc = RNA.fold_compound(rna)
            pred_struct, _ = fc.mfe()
            pred_pairs = pairs_from_structure(pred_struct)
            f1s.append(f1_pairs(pred_pairs, ref_pairs))
            skipped_counts.append(n_skipped)
        med = statistics.median(f1s)
        mean = statistics.mean(f1s)
        med_skip = statistics.median(skipped_counts)
        print(f"  {name:<10s} {len(cons_seq):>4d} {len(ref_pairs):>10d} "
              f"{med_skip:>15.0f} "
              f"{med:>8.3f} {mean:>8.3f} {min(f1s):>7.3f} {max(f1s):>7.3f}")
        all_f1s.extend(f1s)
    print()
    if all_f1s:
        med = statistics.median(all_f1s)
        mean = statistics.mean(all_f1s)
        print(f"  overall (across {len(all_f1s)} realizations): median F1' = {med:.3f}, "
              f"mean = {mean:.3f}")
        if med >= 0.75:
            v = "HIGH: fold prediction is a reliable input; hard theta acceptable"
        elif med >= 0.55:
            v = "MEDIUM: soft prior, moderate weight on structure channels"
        else:
            v = "LOW: (a) MFE-inherent error dominates; structure prediction is a weak prior"
        print(f"  verdict: {v}")
        print()
        print("  compare to D4 (unconstrained): F1_med = 0.160")
        print(f"  delta from constraint: +{med - 0.160:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
