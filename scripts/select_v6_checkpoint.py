"""Constrained V6 checkpoint selection.

Reads every epoch_XX.pt in `checkpoints/v6_stageB1_v2/`, gates by the five
guardrails, then selects the highest-AUROCpair epoch. Ties within 0.005
AUROCpair break in favor of the EARLIEST epoch (closer to V5.2 init, less
domain-transfer risk).

Guardrails (all must pass; NaN metrics are ignored not failed):
    nc_top1                             ≥ 0.90
    wrong_orientation_consistency AUROC ≥ 0.95
    wrong_position_consistency AUROC    ≥ 0.80
    wrong_structure_role_consistency    ≥ 0.73
    AUPRC (overall)                     ≥ 0.85

Output: prints a passing-epoch table and writes:
    checkpoints/v6_selected/best.pt      (copy of chosen epoch_XX.pt)
    checkpoints/v6_selected/selection.json (metadata: chosen epoch + metrics)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')

import torch

CKPT_DIR = Path('/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/checkpoints/v6_stageB1_v2')
OUT_DIR = Path('/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/checkpoints/v6_selected')

GUARDRAILS = {
    'nc_top1':                                    0.90,
    'auroc[wrong_orientation_consistency]':       0.95,
    'auroc[wrong_position_consistency]':          0.80,
    'auroc[wrong_structure_role_consistency]':    0.73,
    'auprc':                                       0.85,
}
TIE_TOL = 0.005    # AUROCpair within this counts as tied → prefer earlier epoch


def main():
    epochs = sorted(CKPT_DIR.glob('epoch_*.pt'))
    if not epochs:
        print(f'No epoch_*.pt in {CKPT_DIR}', file=sys.stderr)
        sys.exit(1)

    rows = []
    for pt in epochs:
        ck = torch.load(pt, map_location='cpu', weights_only=False)
        st = ck.get('val_stats', {})
        rows.append({
            'path': pt,
            'epoch': ck['epoch'],
            'auprc': st.get('auprc', float('nan')),
            'auroc': st.get('auroc', float('nan')),
            'hard_auroc': st.get('auroc_hard_only', float('nan')),
            'nc_top1': st.get('nc_top1', float('nan')),
            'w_orient': st.get('auroc[wrong_orientation_consistency]', float('nan')),
            'w_pos': st.get('auroc[wrong_position_consistency]', float('nan')),
            'w_struct': st.get('auroc[wrong_structure_role_consistency]', float('nan')),
            'auroc_pair': st.get('pair_final_auroc_pos', float('nan')),
            'dmedian': st.get('delta_final_pos_median', float('nan')),
            'dq10': st.get('delta_final_pos_q10', float('nan')),
            'pgt0': st.get('delta_final_pos_frac_gt_0', float('nan')),
            'pgt1': st.get('delta_final_pos_frac_gt_1', float('nan')),
            'val_stats': st,
        })

    print(f'{len(rows)} epoch checkpoints found. Applying guardrails:')
    for k, v in GUARDRAILS.items():
        print(f'  {k:<48} >= {v}')
    print()

    print(f'  {"ep":>2}  {"AUROC":>6} {"AUPRC":>6} {"HARD":>6} '
          f'{"nc_top1":>7} {"w_orient":>8} {"w_pos":>6} {"w_struct":>8} '
          f'{"AUROCpair":>10} {"Δmed":>7} {"Q10":>7} {"P>0":>5} {"P>1":>5}   pass')
    passing = []
    for r in rows:
        checks = []
        def _ok(val, threshold):
            if val is None: return True
            import math
            if isinstance(val, float) and math.isnan(val): return True
            return val >= threshold
        pass_flags = {k: _ok(r['val_stats'].get(k), v) for k, v in GUARDRAILS.items()}
        all_pass = all(pass_flags.values())
        marker = '✓' if all_pass else '✗'
        which_fail = ','.join(k.replace('auroc[', '').replace(']', '').replace('wrong_', 'w_')[:8]
                                for k, ok in pass_flags.items() if not ok)
        print(f'  {r["epoch"]:>2}  {r["auroc"]:>6.4f} {r["auprc"]:>6.4f} '
              f'{r["hard_auroc"]:>6.4f} {r["nc_top1"]:>7.3f} '
              f'{r["w_orient"]:>8.3f} {r["w_pos"]:>6.3f} {r["w_struct"]:>8.3f} '
              f'{r["auroc_pair"]:>10.4f} {r["dmedian"]:>+7.3f} {r["dq10"]:>+7.3f} '
              f'{r["pgt0"]:>5.3f} {r["pgt1"]:>5.3f}   {marker}'
              + (f'  fail: {which_fail}' if not all_pass else ''))
        if all_pass:
            passing.append(r)

    if not passing:
        print('\n[SELECT] No epoch passes all guardrails.')
        sys.exit(2)

    # Pick highest AUROCpair; ties within TIE_TOL → earlier epoch
    max_auroc_pair = max(p['auroc_pair'] for p in passing)
    tied = [p for p in passing if p['auroc_pair'] >= max_auroc_pair - TIE_TOL]
    tied.sort(key=lambda p: p['epoch'])   # earliest epoch first
    winner = tied[0]
    print()
    print(f'[SELECT] Passing epochs: {[p["epoch"] for p in passing]}')
    print(f'[SELECT] Max AUROCpair: {max_auroc_pair:.4f}')
    print(f'[SELECT] Within tie tol ({TIE_TOL}): {[p["epoch"] for p in tied]}')
    print(f'[SELECT] Winner (earliest tied): ep{winner["epoch"]}')
    print(f'         AUROCpair={winner["auroc_pair"]:.4f}, '
          f'Δfinal_median={winner["dmedian"]:+.3f}, '
          f'AUPRC={winner["auprc"]:.4f}, w_pos={winner["w_pos"]:.3f}, '
          f'w_struct={winner["w_struct"]:.3f}')

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(winner['path'], OUT_DIR / 'best.pt')
    with open(OUT_DIR / 'selection.json', 'w') as f:
        json.dump({
            'chosen_epoch': winner['epoch'],
            'source_ckpt': str(winner['path']),
            'guardrails': GUARDRAILS,
            'auroc_pair': winner['auroc_pair'],
            'val_stats': winner['val_stats'],
            'tie_tolerance': TIE_TOL,
            'tied_epochs': [p['epoch'] for p in tied],
            'all_passing_epochs': [p['epoch'] for p in passing],
        }, f, indent=2)
    print(f'\n[SAVE] {OUT_DIR / "best.pt"}  (copy of ep{winner["epoch"]})')
    print(f'[SAVE] {OUT_DIR / "selection.json"}')


if __name__ == '__main__':
    main()
