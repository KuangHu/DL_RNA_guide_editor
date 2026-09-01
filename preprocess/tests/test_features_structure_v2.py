"""Unit test for features_structure_v2.

Verifies on the T-WT anchor (n=1, known biology):

  Shape invariants:
    dG_open_u1     shape (nc_len,)
    dG_open_uL_pn  shape (nc_len - L + 1,)
    E_span_win     shape (nc_len - L + 1,)
    H_pair_win     shape (nc_len - L + 1,)
    bpp            shape (nc_len, nc_len), symmetric
    p_ss           shape (nc_len,)

  Value sanity:
    p_ss in (0, 1]                       — clipped
    dG_open_u1 >= 0 for every position   — -RT ln p_ss with p_ss <= 1
    all channels finite where defined    — no NaN in dG or p_ss
    E_span_win, H_pair_win: NaN only where entire window has pair mass < eps

  Reproducibility of D5 gold-window rank:
    T-WT gold at nc position 49 (L=11) should sit in top ~30 windows by
    P_ss window mean — matches D5's rank 28/167.

Runs one fold on T-WT (177 nt), ~90 ms.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocess.features_structure_v2 import compute_features_v2
from scripts.v5a_framework.match_table import load as load_mt

MT_POS = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5a_framework_cache/durrant_positive"


def main() -> int:
    mt = load_mt(MT_POS)
    twt_id = next(t for t in mt.tnp_ids if t.startswith("durrant_bridge_RNA_T-WT_D-WT"))
    nc = mt.tnps[twt_id].nc.upper().replace("U", "T")
    gold_starts = [s.gold_nc for s in mt.tnps[twt_id].sites]
    gold = int(np.median(gold_starts))
    L = 11

    print(f"[test] T-WT nc len {len(nc)}, gold pos {gold}, L={L}")
    feats = compute_features_v2(nc, guide_length=L)

    passed = True

    def check(name: str, cond: bool, why: str = ""):
        nonlocal passed
        mark = "  OK  " if cond else "  FAIL"
        print(f"{mark}  {name}{'  — ' + why if why else ''}")
        if not cond:
            passed = False

    # Shape invariants
    check("dG_open_u1 shape", feats.dG_open_u1.shape == (len(nc),))
    check("dG_open_uL_pn shape",
          feats.dG_open_uL_pn.shape == (len(nc) - L + 1,))
    check("cooperativity_win_pn shape",
          feats.cooperativity_win_pn.shape == (len(nc) - L + 1,))
    check("E_span_win shape",
          feats.E_span_win.shape == (len(nc) - L + 1,))
    check("H_pair_win shape",
          feats.H_pair_win.shape == (len(nc) - L + 1,))
    check("bpp shape", feats.bpp.shape == (len(nc), len(nc)))
    check("p_ss shape", feats.p_ss.shape == (len(nc),))
    check("bpp symmetric",
          np.allclose(feats.bpp, feats.bpp.T))

    # Value sanity
    check("p_ss in (0, 1]",
          (feats.p_ss > 0).all() and (feats.p_ss <= 1).all(),
          f"min={feats.p_ss.min():.3e}, max={feats.p_ss.max():.3e}")
    check("dG_open_u1 >= 0",
          (feats.dG_open_u1 >= 0).all(),
          f"min={feats.dG_open_u1.min():.3f}")
    check("dG_open_u1 finite",
          np.isfinite(feats.dG_open_u1).all())
    check("dG_open_uL_pn finite",
          np.isfinite(feats.dG_open_uL_pn).all())
    check("cooperativity_win_pn finite",
          np.isfinite(feats.cooperativity_win_pn).all())
    # Sanity: cooperativity should NOT be identically zero (that would mean
    # true_uL_pn == mean(u1), i.e. we regressed to the independence bug).
    coop = feats.cooperativity_win_pn
    non_zero_frac = float((np.abs(coop) > 1e-6).mean())
    check("cooperativity != 0 (not independence-collapsed)",
          non_zero_frac > 0.5,
          f"non-zero fraction = {non_zero_frac:.3f}")

    # MATHEMATICAL BOUNDARY CHECK (catches pfl_fold_up indexing regressions).
    # P(all L positions unpaired) <= P(any single position unpaired), so:
    #   dG_open_uL >= -kT * ln(min p_ss over window)
    #   dG_open_uL_pn * L >= dG_open_u1.max() over the window
    # Equivalently: exp(-dG_open_uL_pn * L / kT) <= min(p_ss over win).
    from preprocess.features_structure_v2 import KT_37C_KCAL_PER_MOL
    n_win = len(feats.dG_open_uL_pn)
    csum_pss_min = np.array([feats.p_ss[k : k + L].min() for k in range(n_win)])
    joint_from_dg = np.exp(-feats.dG_open_uL_pn * L / KT_37C_KCAL_PER_MOL)
    boundary_viol = int((joint_from_dg > csum_pss_min + 1e-12).sum())
    check("P(joint) <= min(p_ss) over each window",
          boundary_viol == 0,
          f"violations = {boundary_viol} / {n_win}")

    # NaN in E_span/H_pair only allowed where windowed pair mass < eps
    # (compute the mask independently)
    pair_mass = feats.bpp.sum(axis=1)
    csum = np.concatenate(([0.0], np.cumsum(pair_mass, dtype=np.float64)))
    win_pair_mass = csum[L:] - csum[:-L]
    expected_nan = win_pair_mass < 1e-8

    e_nan_ok = np.array_equal(np.isnan(feats.E_span_win), expected_nan)
    h_nan_ok = np.array_equal(np.isnan(feats.H_pair_win), expected_nan)
    check("E_span_win NaN pattern matches pair-mass",
          e_nan_ok,
          f"n_nan={int(np.isnan(feats.E_span_win).sum())} vs expected {int(expected_nan.sum())}")
    check("H_pair_win NaN pattern matches pair-mass",
          h_nan_ok)
    check("windowed_valid mask matches NaN pattern",
          np.array_equal(feats.windowed_valid, ~expected_nan))

    # D5 anchor: gold window should be in the top ~30 by P_ss window mean
    p_ss = feats.p_ss
    csum_p = np.concatenate(([0.0], np.cumsum(p_ss, dtype=np.float64)))
    p_ss_win = (csum_p[L:] - csum_p[:-L]) / L
    if 0 <= gold < len(p_ss_win):
        gold_p = p_ss_win[gold]
        n_above = int((p_ss_win >= gold_p).sum())
        check("D5 anchor: gold rank in top 30",
              n_above <= 30,
              f"rank {n_above} / {len(p_ss_win)}")
        # Also cross-check the percentile
        pct = float((p_ss_win < gold_p).mean())
        check("D5 anchor: gold percentile in [0.80, 0.90]",
              0.80 <= pct <= 0.90,
              f"percentile = {pct:.3f}")
    else:
        check("D5 anchor gold in range", False, f"gold={gold} out of window range")

    # Report the four channels at the gold window
    print()
    print(f"[report] channels at gold window (start={gold}):")
    print(f"  dG_open_u1[{gold}]          = {feats.dG_open_u1[gold]:.3f} kcal/mol")
    print(f"  dG_open_uL_pn[{gold}] (per-nt) = "
          f"{feats.dG_open_uL_pn[gold]:.3f} kcal/mol/nt")
    print(f"  cooperativity_win_pn[{gold}] = "
          f"{feats.cooperativity_win_pn[gold]:+.3f} kcal/mol/nt "
          f"(positive = anti-cooperative)")
    print(f"  E_span_win[{gold}]          = {feats.E_span_win[gold]:.2f} nt")
    print(f"  H_pair_win[{gold}]          = {feats.H_pair_win[gold]:.3f} nats")

    print()
    print(f"[report] channel-level medians (nan-safe):")
    print(f"  dG_open_u1     med = {np.median(feats.dG_open_u1):.3f}")
    print(f"  dG_open_uL_pn  med = {np.median(feats.dG_open_uL_pn):.3f}")
    print(f"  E_span_win     med = {np.nanmedian(feats.E_span_win):.2f}, "
          f"n_nan = {int(np.isnan(feats.E_span_win).sum())}")
    print(f"  H_pair_win     med = {np.nanmedian(feats.H_pair_win):.3f}, "
          f"n_nan = {int(np.isnan(feats.H_pair_win).sum())}")

    print()
    print("[test] PASS" if passed else "[test] FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
