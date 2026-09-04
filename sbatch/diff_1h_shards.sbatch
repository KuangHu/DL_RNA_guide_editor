#!/bin/bash
#SBATCH --job-name=diff_1h
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0:10:00
#SBATCH --output=logs/diff_1h_%j.out
#SBATCH --error=logs/diff_1h_%j.err
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
/global/home/users/kh36969/.conda/envs/opfi/bin/python -u <<'PY'
import numpy as np, sys
sys.path.insert(0, '.')
from scripts.v5a_framework.match_table import load as load_mt

MT_PRE = load_mt('/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/channel_a/mt_v6_5k_hom10_1c_pin5v2')
MT_NEW = load_mt('/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/channel_a/mt_1h_pin_all')

common = set(MT_PRE.tnp_ids) & set(MT_NEW.tnp_ids)
print(f"MT_PRE tnps={len(MT_PRE.tnp_ids)} MT_NEW tnps={len(MT_NEW.tnp_ids)} common={len(common)}", flush=True)

n_checked = 0
n_diffs = 0
diff_tnps = []
for tid in list(common)[:200]:
    tnp = MT_PRE.tnps[tid]
    for s in tnp.sites[:5]:
        for orient in ("fwd", "rc"):
            for L in (11,):
                pre = MT_PRE.get(tid, s.site_idx, orient, L).m_max_by_excl[0]
                new = MT_NEW.get(tid, s.site_idx, orient, L).m_max_by_excl[0]
                n_checked += 1
                if not np.array_equal(pre, new):
                    n_diffs += 1
                    if tid not in diff_tnps:
                        diff_tnps.append(tid)
                    if len(diff_tnps) <= 3:
                        d = np.where(pre != new)[0]
                        print(f"  DIFF tnp={tid} site={s.site_idx} orient={orient} L={L}: {len(d)} positions differ")
                        for p in d[:3]:
                            print(f"    pos {p}: pre={pre[p]} new={new[p]}")
print(f"\nchecked {n_checked} (site,orient,L) triples")
print(f"n_diffs = {n_diffs}, distinct diff Tnps = {len(diff_tnps)}")
if not n_diffs:
    print("  ~ byte-identical ~")
PY
