"""τ=0 anchor regression test.

Historical result on Durrant 65 Tnps (fixed_L11_m8, tau=0):

    X1' v3 log     : "0.338  0.957  0.323"  -> peak-level PPV = 22/23 = 0.9565
    Channel A doc  : "0.955 (21/22)"        -> Tnp-level  PPV = 21/22 = 0.9545

    Both are correct measurements on the SAME data under different definitions:
      peak-level PPV = correct_peaks / total_peaks     (X1' v3 log)
      Tnp-level PPV  = Tnps_with_correct_peak / covered_Tnps  (Channel A doc)

    The framework is byte-equivalent to X1' v3 at the peak level. The +1 peak
    over covered-Tnps count comes from ONE Tnp (bag010) whose S has a
    plateau at nc positions 49 and 50 (identical S=5.0 at both) because at
    tau=0 the aggregate is an integer count over the 5 sites' hits, and
    ties occur with positive probability. The local-max rule is
      "i is a peak iff no j in [i-min_dist, i+min_dist] has S[j] > S[i]"
    which is equivalent to S[i] >= S[j] for all j in the window — so a
    plateau emits every position on it. peak_min_dist=5 is a radius for
    the local-max check, NOT a merge distance for the output, so peaks at
    49 and 50 (distance 1) both survive. X1' v3's _find_peaks uses this
    same rule (verified by diff), and the plateau at bag010 is preserved.

    Corollary for P3: peak-level PPV's DENOMINATOR is inflated at tau=0
    (plateaus common) and unaffected at tau>0 (Gaussian ties measure-zero).
    Cross-tau peak-level comparisons are therefore not apples-to-apples.
    P3 canonical PPV = Tnp-level (tnps_with_correct / covered) — plateau-
    immune.

Tightened assertions: this test enforces EXACT discrete counts (no tolerance).
Any deviation surfaces silent framework drift immediately.

Run:
    python -m scripts.v5a_framework.tests.test_tau0_anchor
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v5a_framework.match_table import (build_positive, load as load_mt)
from v5a_framework.variant import spec_m_threshold_L11, run_variant


DURRANT_COG = "/global/scratch/users/kh36969/DL_novel_guide_editor/R1_baseline/durrant_cog_vs_shuf.jsonl"
DURRANT_GOLD = "/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/annotation/durrant_gold_v1.jsonl"
SHARD_DIR = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5a_framework_cache/durrant_positive"

# Exact discrete-count anchors (no tolerance).
ANCHOR_N_TNPS = 65
ANCHOR_COVERED = 22               # covered_Tnps == 22
ANCHOR_TOTAL_PEAKS = 23           # total peaks emitted (includes 1 tie at bag010)
ANCHOR_PEAKS_CORRECT = 22         # peaks passing IoU>=0.5
ANCHOR_TNPS_WITH_CORRECT = 21     # Tnps with at least one correct peak
ANCHOR_EXACT = 21                 # Tnps whose best peak is within 1 nt of gold_nc

# Derived (for readable reporting only)
ANCHOR_COVERAGE_RATE = ANCHOR_COVERED / ANCHOR_N_TNPS               # 0.3385
ANCHOR_PPV_PEAK_LEVEL = ANCHOR_PEAKS_CORRECT / ANCHOR_TOTAL_PEAKS   # 0.9565
ANCHOR_PPV_TNP_LEVEL = ANCHOR_TNPS_WITH_CORRECT / ANCHOR_COVERED    # 0.9545
ANCHOR_EXACT_RATE = ANCHOR_EXACT / ANCHOR_N_TNPS                    # 0.3231


def _iou(p, L_win, gold_nc, gold_L, thresh=0.5) -> bool:
    a0, a1 = p, p + L_win
    b0, b1 = gold_nc, gold_nc + gold_L
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return union > 0 and inter / union >= thresh


def _build_or_load() -> object:
    if (Path(SHARD_DIR) / "_index.json").exists():
        print(f"[anchor] loading cached MatchTable from {SHARD_DIR}", flush=True)
        return load_mt(SHARD_DIR)
    print(f"[anchor] building MatchTable (this takes ~1 min)...", flush=True)
    return build_positive(DURRANT_COG, DURRANT_GOLD, SHARD_DIR)


def _primary_peak_gold_blind(pks) -> float:
    """Primary peak = centroid of the top-S_all plateau. Gold-blind by design.

    Peeking at gold_nc to pick the tie-break peak (as the earlier version did)
    hides the plateau: bag010 at tau=0 has peaks at 49 and 50 with equal S,
    and gold at 49; gold-aware tie-break would always pick 49 and call it
    exact. That masks the plateau structure. Centroid = mean position of all
    peaks that share the maximal S_all — tau-invariant and gold-independent.
    """
    max_S = max(pk.S_all for pk in pks)
    top = [pk.position for pk in pks if pk.S_all == max_S]
    return sum(top) / len(top)


def compute_anchor_metrics(mt) -> dict:
    spec = spec_m_threshold_L11(m=8, tau=0, S=5)
    peaks_by_tnp = run_variant(mt, spec)

    n_tnps = len(mt.tnp_ids)
    covered = 0
    total_peaks = 0
    peaks_correct = 0
    tnps_with_correct = 0
    exact = 0
    tie_events: list[str] = []
    for tnp_id in mt.tnp_ids:
        pks = peaks_by_tnp.get(tnp_id, [])
        if not pks:
            continue
        covered += 1
        tnp = mt.tnps[tnp_id]
        gold_nc = tnp.sites[0].gold_nc
        gold_L = tnp.sites[0].gold_L
        any_ok = False
        for pk in pks:
            total_peaks += 1
            if _iou(pk.position, pk.L_at_peak, gold_nc, gold_L):
                peaks_correct += 1
                any_ok = True
        if any_ok:
            tnps_with_correct += 1
        if len(pks) > 1:
            tie_events.append(f"{tnp_id}: {[(pk.position, pk.orient) for pk in pks]}")
        primary_pos = _primary_peak_gold_blind(pks)
        if abs(primary_pos - gold_nc) <= 1:
            exact += 1

    return {
        "n_tnps": n_tnps,
        "covered": covered,
        "total_peaks": total_peaks,
        "peaks_correct": peaks_correct,
        "tnps_with_correct": tnps_with_correct,
        "exact": exact,
        "coverage_rate": covered / n_tnps,
        "ppv_peak_level": peaks_correct / max(1, total_peaks),
        "ppv_tnp_level": tnps_with_correct / max(1, covered),
        "exact_rate": exact / n_tnps,
        "tie_events": tie_events,
    }


def main() -> int:
    mt = _build_or_load()
    m = compute_anchor_metrics(mt)
    print(f"[anchor] n_tnps={m['n_tnps']}, covered={m['covered']}, "
          f"total_peaks={m['total_peaks']}, peaks_correct={m['peaks_correct']}, "
          f"tnps_with_correct={m['tnps_with_correct']}, exact={m['exact']}", flush=True)
    print(f"[anchor] coverage_rate = {m['coverage_rate']:.4f} (target {ANCHOR_COVERAGE_RATE:.4f})", flush=True)
    print(f"[anchor] PPV peak-level = {m['ppv_peak_level']:.4f}  "
          f"(target {ANCHOR_PPV_PEAK_LEVEL:.4f} — X1' v3 log)", flush=True)
    print(f"[anchor] PPV Tnp-level  = {m['ppv_tnp_level']:.4f}  "
          f"(target {ANCHOR_PPV_TNP_LEVEL:.4f} — Channel A doc)", flush=True)
    print(f"[anchor] exact_rate    = {m['exact_rate']:.4f} (target {ANCHOR_EXACT_RATE:.4f})", flush=True)
    if m["tie_events"]:
        print(f"[anchor] multi-peak tie events:", flush=True)
        for e in m["tie_events"]:
            print(f"  - {e}", flush=True)

    # Exact discrete-count assertions (no tolerance)
    checks = {
        "n_tnps":            (m["n_tnps"],            ANCHOR_N_TNPS),
        "covered":           (m["covered"],           ANCHOR_COVERED),
        "total_peaks":       (m["total_peaks"],       ANCHOR_TOTAL_PEAKS),
        "peaks_correct":     (m["peaks_correct"],     ANCHOR_PEAKS_CORRECT),
        "tnps_with_correct": (m["tnps_with_correct"], ANCHOR_TNPS_WITH_CORRECT),
        "exact":             (m["exact"],             ANCHOR_EXACT),
    }
    fails = [f"{name}: got {got} != expected {want}"
             for name, (got, want) in checks.items() if got != want]
    if fails:
        print("[anchor] FAIL:", flush=True)
        for f in fails:
            print(f"  - {f}", flush=True)
        return 1
    print("[anchor] PASS (all 6 discrete counts match)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
