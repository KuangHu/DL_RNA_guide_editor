"""Item 5 (a) — 6-anchor old vs new p_ss comparison.

Runs RNAplfold -W 120 -L 60 -u 1 on the 6 anchors (T-WT + 5 seekRNA),
parses P(unpaired at u=1) per position, then compares to
features_structure_v2.p_ss on the same 6 sequences at global params.

Reports:
  - per-position |Δp_ss| distribution per ncRNA
  - localization: positions with largest Δ, are they participating in
    long-range pairs (span > 60) in the new global fold?
  - T-WT gold[49] percentile in OLD (from RNAplfold u=1) vs NEW p_ss

Confirms: -L 60 hides the outermost helix (D3), which affects P_ss at
the positions that were pairing partners across span > 60.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import RNA

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from preprocess.features_structure_v2 import compute_features_v2
from scripts.v5a_framework.match_table import load as load_mt
from scripts.v5a_framework.tests.aplus_calibration import SEEKRNA_SUPP_T1

MT_POS = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5a_framework_cache/durrant_positive"
GUIDE_L = 11


def rnaplfold_u1(seq: str, W: int = 120, L: int = 60) -> np.ndarray:
    """Run RNAplfold -W W -L L -u 1 and parse P(unpaired) per position."""
    rna = seq.upper().replace("T", "U")
    with tempfile.TemporaryDirectory() as tmpd:
        cmd = ["RNAplfold", "-W", str(W), "-L", str(L), "-u", "1"]
        p = subprocess.run(
            cmd, cwd=tmpd,
            input=f">tmp\n{rna}\n", text=True,
            capture_output=True,
        )
        if p.returncode != 0:
            raise RuntimeError(f"RNAplfold failed: {p.stderr}")
        lunp = Path(tmpd) / "tmp_lunp"
        if not lunp.exists():
            raise RuntimeError(f"RNAplfold produced no _lunp file: {list(Path(tmpd).iterdir())}")
        vals = np.zeros(len(rna), dtype=np.float64)
        with open(lunp) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                if len(parts) < 2:
                    continue
                try:
                    i = int(parts[0])
                except ValueError:
                    continue
                v = parts[1]
                if v == "NA":
                    vals[i - 1] = np.nan
                else:
                    vals[i - 1] = float(v)
    return vals


def max_pair_span(seq: str) -> int:
    """Global MFE and its max base-pair span."""
    fc = RNA.fold_compound(seq.upper().replace("T", "U"))
    structure, _ = fc.mfe()
    stack = []
    m = 0
    for i, c in enumerate(structure):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            j = stack.pop()
            m = max(m, i - j)
    return m


def main() -> int:
    mt = load_mt(MT_POS)
    twt_id = next(t for t in mt.tnp_ids if t.startswith("durrant_bridge_RNA_T-WT_D-WT"))
    twt_nc = mt.tnps[twt_id].nc.upper().replace("U", "T")
    twt_gold_starts = [s.gold_nc for s in mt.tnps[twt_id].sites]
    twt_gold = int(np.median(twt_gold_starts))

    anchors = [("T-WT", twt_nc)]
    for k, v in SEEKRNA_SUPP_T1.items():
        anchors.append((k, v.upper()))

    print(f"[5a] {'anchor':<22s} {'len':>4s} {'max_span':>9s} "
          f"{'ss_mean_old':>12s} {'ss_mean_new':>12s} "
          f"{'|Δ| med':>9s} {'|Δ| p95':>9s} {'r_spear':>9s}")
    from scipy.stats import spearmanr
    per_anchor = {}
    for name, seq in anchors:
        p_ss_old = rnaplfold_u1(seq)         # (len,), NaN at boundary
        feats = compute_features_v2(seq, guide_length=GUIDE_L)
        p_ss_new = feats.p_ss
        assert len(p_ss_old) == len(p_ss_new) == len(seq)
        mask = ~np.isnan(p_ss_old)
        diff = np.abs(p_ss_new - p_ss_old)
        med = float(np.median(diff[mask]))
        p95 = float(np.percentile(diff[mask], 95))
        r, _ = spearmanr(p_ss_old[mask], p_ss_new[mask])
        span = max_pair_span(seq)
        print(f"  {name:<22s} {len(seq):>4d} {span:>9d} "
              f"{float(p_ss_old[mask].mean()):>12.3f} "
              f"{float(p_ss_new.mean()):>12.3f} "
              f"{med:>9.3f} {p95:>9.3f} {r:>9.3f}")
        per_anchor[name] = {
            "p_ss_old": p_ss_old,
            "p_ss_new": p_ss_new,
            "diff": diff,
            "span": span,
            "mask": mask,
        }

    # T-WT gold[49] percentile old vs new
    print()
    print("=== T-WT gold[49:60] window mean p_ss: OLD vs NEW ===")
    p_ss_old = per_anchor["T-WT"]["p_ss_old"]
    p_ss_new = per_anchor["T-WT"]["p_ss_new"]
    # Compute window means
    def win_mean(arr):
        n = len(arr) - GUIDE_L + 1
        out = np.full(n, np.nan)
        for i in range(n):
            w = arr[i : i + GUIDE_L]
            wm = ~np.isnan(w)
            if wm.sum() < GUIDE_L:
                continue
            out[i] = w[wm].mean()
        return out
    old_win = win_mean(p_ss_old)
    new_win = win_mean(p_ss_new)
    # Gold percentile
    valid_old = ~np.isnan(old_win)
    valid_new = ~np.isnan(new_win)
    if valid_old[twt_gold] and valid_new[twt_gold]:
        pct_old = float((old_win[valid_old] < old_win[twt_gold]).mean())
        pct_new = float((new_win[valid_new] < new_win[twt_gold]).mean())
        rank_old_str = f"{int((old_win[valid_old] > old_win[twt_gold]).sum()) + 1} / {int(valid_old.sum())}"
        rank_new_str = f"{int((new_win[valid_new] > new_win[twt_gold]).sum()) + 1} / {int(valid_new.sum())}"
        print(f"  gold pos {twt_gold}")
        print(f"    OLD (-W 120 -L 60): p_ss_win = {float(old_win[twt_gold]):.3f}, "
              f"percentile = {pct_old * 100:.1f}%, rank = {rank_old_str}")
        print(f"    NEW (global):        p_ss_win = {float(new_win[twt_gold]):.3f}, "
              f"percentile = {pct_new * 100:.1f}%, rank = {rank_new_str}")
        print(f"    Δ percentile: {(pct_new - pct_old) * 100:+.1f} pp")

    # Localize largest per-position differences on T-WT
    print()
    print("=== Localization: where do OLD and NEW diverge most on T-WT? ===")
    twt_diff = per_anchor["T-WT"]["diff"]
    valid = per_anchor["T-WT"]["mask"]
    order = np.argsort(-twt_diff)
    print("  top 10 positions by |Δp_ss|:")
    print(f"    {'pos':>4s} {'p_ss_old':>10s} {'p_ss_new':>10s} {'|Δ|':>7s}")
    shown = 0
    for pos in order:
        if not valid[pos]:
            continue
        print(f"    {int(pos):>4d} {p_ss_old[pos]:>10.3f} {p_ss_new[pos]:>10.3f} {twt_diff[pos]:>7.3f}")
        shown += 1
        if shown >= 10:
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
