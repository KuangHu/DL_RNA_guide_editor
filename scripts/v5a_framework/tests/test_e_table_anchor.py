"""Anchor regression on the Option E table's Durrant-self diagonal.

If build_e_positive_diagonal is semantically equivalent to build_positive
on the diagonal case (each Tnp's flanks × own nc), the historical anchor
must reproduce byte-for-byte: all six discrete counts (65, 22, 23, 22,
21, 21) must hold on the E-table's Durrant-self slice.

This test is 2.4's merge gate. If it fails, the E-table shard build is
semantically off, and no downstream E-table result is trustworthy.

Run:
    python -m scripts.v5a_framework.tests.test_e_table_anchor
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v5a_framework.match_table import load as load_mt
from v5a_framework.e_match_table import (build_e_positive_diagonal, load_e,
                                            DiagonalShim)
from v5a_framework.variant import spec_m_threshold_L11, run_variant


MT_POS_SHARD = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5a_framework_cache/durrant_positive"
E_SHARD = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5a_framework_cache/e_durrant_diagonal"

ANCHOR_N_TNPS = 65
ANCHOR_COVERED = 22
ANCHOR_TOTAL_PEAKS = 23
ANCHOR_PEAKS_CORRECT = 22
ANCHOR_TNPS_WITH_CORRECT = 21
ANCHOR_EXACT = 21


def _iou(p: int, L_win: int, gold_nc: int, gold_L: int, thresh: float = 0.5) -> bool:
    a0, a1 = p, p + L_win
    b0, b1 = gold_nc, gold_nc + gold_L
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return union > 0 and inter / union >= thresh


def _primary_position_gold_blind(pks) -> float:
    max_S = max(pk.S_all for pk in pks)
    top = [pk.position for pk in pks if pk.S_all == max_S]
    return sum(top) / len(top)


def _build_or_load_e() -> object:
    if (Path(E_SHARD) / "_e_index.json").exists():
        print(f"[e-anchor] loading cached E-table diagonal from {E_SHARD}", flush=True)
        return load_e(E_SHARD)
    print(f"[e-anchor] building E-table diagonal (this is fast, ~1 min)...", flush=True)
    mt_pos = load_mt(MT_POS_SHARD)
    return build_e_positive_diagonal(mt_pos, E_SHARD)


def main() -> int:
    emt = _build_or_load_e()
    print(f"[e-anchor] emt.summary = {emt.summary()}", flush=True)

    mt_shim = DiagonalShim(emt)
    spec = spec_m_threshold_L11(m=8, tau=0, S=5)
    peaks_by_tnp = run_variant(mt_shim, spec)

    n_tnps = len(mt_shim.tnp_ids)
    covered = 0
    total_peaks = 0
    peaks_correct = 0
    tnps_with_correct = 0
    exact = 0
    for tnp_id in mt_shim.tnp_ids:
        pks = peaks_by_tnp.get(tnp_id, [])
        if not pks:
            continue
        covered += 1
        tnp = mt_shim.tnps[tnp_id]
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
        primary_pos = _primary_position_gold_blind(pks)
        if abs(primary_pos - gold_nc) <= 1:
            exact += 1

    print(f"[e-anchor] n_tnps={n_tnps}, covered={covered}, total_peaks={total_peaks}, "
          f"peaks_correct={peaks_correct}, tnps_with_correct={tnps_with_correct}, "
          f"exact={exact}", flush=True)

    checks = {
        "n_tnps":            (n_tnps,            ANCHOR_N_TNPS),
        "covered":           (covered,           ANCHOR_COVERED),
        "total_peaks":       (total_peaks,       ANCHOR_TOTAL_PEAKS),
        "peaks_correct":     (peaks_correct,     ANCHOR_PEAKS_CORRECT),
        "tnps_with_correct": (tnps_with_correct, ANCHOR_TNPS_WITH_CORRECT),
        "exact":             (exact,             ANCHOR_EXACT),
    }
    fails = [f"{name}: got {got} != expected {want}"
             for name, (got, want) in checks.items() if got != want]
    if fails:
        print("[e-anchor] FAIL:", flush=True)
        for f in fails:
            print(f"  - {f}", flush=True)
        return 1
    print("[e-anchor] PASS (E-table diagonal byte-equivalent to historical MatchTable)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
