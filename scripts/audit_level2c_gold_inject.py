"""LEVEL 2C — Gold candidate injection through V6.

For each site, keep the normal 96-candidate pool from build_candidate_arrays,
then FORCE one slot in the (orient='fwd', L=11) group to be the "gold" candidate:
  nc_start = the LTG-loop position in this bRNA
  flank_start = 0    (junction anchor; downstream flank convention)
  L = 11
  orient = 'fwd'
  matches = whatever the actual alignment gives at that (nc_start, flank_start)

The LTG-loop position is defined per-bRNA as the position where a sliding L=11
fwd alignment between the bRNA DNA sequence and the bRNA's TBL specificity
string has the maximum match count. THIS USES ANNOTATION — only for this
diagnostic, not for downstream training.

Injection strategy:
  If the gold candidate is already in the top-4 for (fwd, L=11), leave as-is.
  Otherwise, replace the k=3 (worst-rank) slot in that group with the gold.
  All 88 other slots unchanged. Total K = 96 (same as current V6 pool).

Then re-forward V6 and re-forward V5.2 on both cognate and shuffled splits,
compare to their un-injected baseline scores.

If injected V6 AUROC ≫ 0.55 → proposal starvation is the dominant loss.
If injected V6 AUROC still ≈ 0.55 → learned cand-MIL / pair_head cannot use
the gold even when it's present.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')

import numpy as np
import openpyxl
import torch

from model.v1 import V1Config, V1Model
from preprocess.candidates import (
    build_candidate_arrays, Candidate, _fill_candidate_slot,
    DEFAULT_L_MIN, DEFAULT_L_MAX, DEFAULT_ORIENTATIONS,
    TOP_K_PER_COMBO_DEFAULT, PATCH_WIDTH_DEFAULT,
)
from preprocess.site import StructureCache
from preprocess.tnp_dataset import TnpGroupedDataset, collate_tnp_batch

BASE = Path('/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/inference')
CKPTS = {
    'V5.2': '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/checkpoints/v5_2_stageA_from_v4/best.pt',
    'V6':   '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/checkpoints/v6_selected/best.pt',
}
SPLITS = {
    'cognate':  {'jsonl': BASE / 'durrant_cognate.jsonl',
                 'cache': BASE / 'struct' / 'durrant_cognate_u16.index.json'},
    'shuffled': {'jsonl': BASE / 'durrant_shuffled.jsonl',
                 'cache': BASE / 'struct' / 'durrant_shuffled_u16.index.json'},
}
TABLE2 = '/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/unpacked/2023-09-16026B-s3/2023-09-16026B-SupplementaryTable2.xlsx'
OUT = BASE / 'level2c_scores.jsonl'


BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}


def load_ltg_specs():
    """Return {bRNA_name: TBL_specificity_string_first_11}."""
    wb = openpyxl.load_workbook(TABLE2, data_only=True)
    ws = wb['Key bridge RNAs Used']
    rows = list(ws.iter_rows(values_only=True))
    out = {}
    for r in rows[1:]:
        if not r or r[0] is None:
            continue
        name = r[0]
        tbl = r[1]
        if not tbl or tbl == 'N/A':
            continue
        out[name] = tbl[:11]  # trim to 11 bp
    wb.close()
    return out


def find_ltg_position(brna_dna: str, tbl_spec: str) -> tuple[int, int]:
    """Slide the 11-bp spec across the bRNA, return (best_nc_start, matches)."""
    L = 11
    if len(brna_dna) < L or len(tbl_spec) != L:
        return -1, -1
    best_pos, best_m = -1, -1
    tbl_arr = np.asarray([BASE_MAP.get(c, 4) for c in tbl_spec.upper()], dtype=np.int8)
    for i in range(len(brna_dna) - L + 1):
        window = brna_dna[i:i + L].upper()
        win_arr = np.asarray([BASE_MAP.get(c, 4) for c in window], dtype=np.int8)
        m = int(((tbl_arr == win_arr) & (tbl_arr < 4)).sum())
        if m > best_m:
            best_m = m
            best_pos = i
    return best_pos, best_m


def slot_index(orient_idx: int, L: int, k: int,
                L_min=DEFAULT_L_MIN, top_k=TOP_K_PER_COMBO_DEFAULT) -> int:
    """Layout: for orient in [fwd, rc]: for L in [L_min..L_max]: k in [0..top_k)."""
    n_L = DEFAULT_L_MAX - L_min + 1
    return orient_idx * n_L * top_k + (L - L_min) * top_k + k


def build_and_inject(nc, flank, structure_profile, structure_valid,
                     gold_nc_start, ltg_pos, brna_name, cache_debug=None):
    """Run normal build_candidate_arrays, then overwrite slot for
    (fwd, L=11, k=3) with the gold candidate if it's not already in top-4.
    Return patches, feats, mask, and whether we actually injected."""
    patches, feats, mask, cands = build_candidate_arrays(
        nc=nc, flank=flank,
        structure_profile=structure_profile,
        structure_valid=structure_valid,
        top_k_per_combo=TOP_K_PER_COMBO_DEFAULT,
        L_min=DEFAULT_L_MIN, L_max=DEFAULT_L_MAX,
        orientations=DEFAULT_ORIENTATIONS,
        patch_width=PATCH_WIDTH_DEFAULT, nc_max=350,
    )

    # Compute gold matches at (fwd, L=11, nc_start=gold_nc_start, flank_start=0)
    L = 11
    flank_start = 0
    if gold_nc_start < 0 or gold_nc_start + L > len(nc) or len(flank) < L:
        return patches, feats, mask, False, None

    nc_win = nc[gold_nc_start:gold_nc_start + L].upper()
    flank_win = flank[:L].upper()
    gold_matches = sum(1 for a, b in zip(nc_win, flank_win) if a == b and a != 'N')

    # Check if the gold is already in the top-4 for (fwd, L=11)
    fwd_L11_slots = [slot_index(0, L, k) for k in range(TOP_K_PER_COMBO_DEFAULT)]
    already_in = False
    for slot in fwd_L11_slots:
        c = cands[slot]
        if c is None: continue
        if c.orient == 'fwd' and c.L == L and c.nc_start == gold_nc_start and c.flank_start == 0:
            already_in = True
            break

    injected = False
    if not already_in:
        # Overwrite worst-rank slot (k=3) with gold
        gold = Candidate(orient='fwd', L=L, nc_start=gold_nc_start,
                          flank_start=flank_start, matches=gold_matches)
        target_slot = slot_index(0, L, TOP_K_PER_COMBO_DEFAULT - 1)  # k=3
        # Clear old patch + feats before refilling
        patches[target_slot] = 0.0
        feats[target_slot] = 0.0
        # Build arrays for _fill_candidate_slot
        nc_codes = np.asarray([BASE_MAP.get(c, 4) for c in nc.upper()], dtype=np.int8)
        flank_codes = np.asarray([BASE_MAP.get(c, 4) for c in flank.upper()], dtype=np.int8)
        _fill_candidate_slot(
            patches=patches, feats=feats, mask=mask, slot=target_slot,
            nc_codes=nc_codes, flank_codes=flank_codes,
            structure_profile=structure_profile,
            structure_valid=structure_valid,
            cand=gold,
            patch_width=PATCH_WIDTH_DEFAULT, nc_max=350,
        )
        mask[target_slot] = True
        injected = True

    return patches, feats, mask, injected, gold_matches


def build_batch_from_split(jsonl_path, cache_index, ltg_specs, ltg_positions):
    """Yield (bag_id, per-site arrays, nc_region_mask, injected_flags, gold_match_counts).
    Reads the JSONL directly; groups records by transposase_id."""
    from collections import defaultdict
    cache = StructureCache(cache_index)
    # Group records by transposase_id
    by_tnp = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            by_tnp[r['transposase_id']].append(r)

    for tnp, recs in by_tnp.items():
        patches_list, feats_list, mask_list, nc_masks_list = [], [], [], []
        injected_flags, gold_match_counts = [], []
        for rec in recs:
            nc = rec['inputs']['noncoding_regions'][rec['labels']['active_noncoding_index']]
            flank = rec['inputs']['flank']
            brna_name = rec['generator_metadata']['is_id']
            ltg_pos = ltg_positions.get(brna_name, -1)
            # Get structure for this site's active NC
            struct, valid = cache.get(rec['site_id'],
                                        slot=rec['labels']['active_noncoding_index'],
                                        nc_len=len(nc))
            patches, feats, mask, injected, gold_m = build_and_inject(
                nc, flank, struct.astype(np.float32), valid.astype(bool),
                gold_nc_start=ltg_pos, ltg_pos=ltg_pos, brna_name=brna_name,
            )
            injected_flags.append(injected)
            gold_match_counts.append(gold_m if gold_m is not None else -1)
            patches_list.append(patches)
            feats_list.append(feats)
            mask_list.append(mask)
            # nc_region_mask is per-site (N_nc,) boolean — True where NC slot is populated.
            # Durrant records have exactly 1 NC → mask = [True, False, False] for N_nc=3.
            nc_masks_list.append(np.array([True, False, False], dtype=bool))
        yield tnp, patches_list, feats_list, mask_list, np.stack(nc_masks_list), \
              injected_flags, gold_match_counts


def score_split(ckpt_name, split_name, injected_data, device):
    """injected_data is a list of (tnp, patches_list, feats_list, mask_list, nc_region_mask, ...)."""
    ckpt = torch.load(CKPTS[ckpt_name], map_location=device, weights_only=False)
    cfg = V1Config(**ckpt['cfg'])
    model = V1Model(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    rows = []
    for tup in injected_data:
        (tnp, patches_list, feats_list, mask_list, nc_region_mask,
         injected_flags, gold_match_counts) = tup
        patches = np.stack(patches_list, axis=0)   # (n_sites, K, W, C)
        feats = np.stack(feats_list, axis=0)       # (n_sites, K, F)
        mask = np.stack(mask_list, axis=0)         # (n_sites, K)
        # Pad to N_nc=3 NC slots (Durrant records have 1 NC; pad slots 1,2 with zeros/False)
        N_NC = 3
        n_sites, K, W, C = patches.shape
        F = feats.shape[-1]
        patches_pad = np.zeros((n_sites, N_NC, K, W, C), dtype=patches.dtype)
        feats_pad = np.zeros((n_sites, N_NC, K, F), dtype=feats.dtype)
        mask_pad = np.zeros((n_sites, N_NC, K), dtype=mask.dtype)
        patches_pad[:, 0] = patches
        feats_pad[:, 0] = feats
        mask_pad[:, 0] = mask
        patches_t = torch.from_numpy(patches_pad).unsqueeze(0).to(device)
        feats_t = torch.from_numpy(feats_pad).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask_pad).unsqueeze(0).to(device)
        # nc_region_mask input is now (n_sites, N_nc) bool — just add B=1
        nc_arr = nc_region_mask if isinstance(nc_region_mask, np.ndarray) else nc_region_mask.numpy()
        nc_t = torch.from_numpy(nc_arr).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(patches_t, feats_t, mask_t, nc_t)
        rows.append({
            'ckpt': ckpt_name, 'split': split_name, 'tnp_id': tnp,
            'n_sites': int(patches.shape[0]),
            'n_injected': int(sum(injected_flags)),
            'gold_matches_med': float(np.median([g for g in gold_match_counts if g >= 0])),
            'logit': float(out['logit'].item()),
            'base_logit': float(out['base_logit'].item()),
            'score': float(torch.sigmoid(out['logit']).item()),
        })
    del model
    torch.cuda.empty_cache()
    return rows


def _auroc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if labels.sum() == 0 or (~labels).sum() == 0:
        return float('nan')
    order = np.argsort(-scores, kind='mergesort')
    y = labels[order]
    tps = np.cumsum(y); fps = np.cumsum(~y)
    tps = np.concatenate([[0], tps]); fps = np.concatenate([[0], fps])
    tpr = tps / max(1, tps[-1]); fpr = fps / max(1, fps[-1])
    return float(np.trapezoid(tpr, fpr))


def main():
    device = torch.device('cuda')

    # 1) Load LTG specificities + find LTG positions per bRNA
    ltg_specs = load_ltg_specs()
    print(f'[table2] {len(ltg_specs)} bRNAs with TBL specificity')

    # Use one representative site from cognate JSONL per bRNA to get its bRNA DNA sequence
    brna_dna = {}
    for line in open(SPLITS['cognate']['jsonl']):
        rec = json.loads(line)
        b = rec['generator_metadata']['is_id']
        if b not in brna_dna:
            brna_dna[b] = rec['inputs']['noncoding_regions'][
                rec['labels']['active_noncoding_index']]

    ltg_positions = {}
    for brna, dna in brna_dna.items():
        if brna not in ltg_specs:
            continue
        pos, m = find_ltg_position(dna, ltg_specs[brna])
        ltg_positions[brna] = pos
        print(f'  {brna:<40} spec={ltg_specs[brna]}  ltg_pos={pos}  matches_to_spec={m}/11')

    # 2) Build injected candidate arrays + score
    print(f'\n[build+score]')
    all_rows = []
    for ckpt in CKPTS:
        for split in SPLITS:
            t0 = time.time()
            injected_data = list(build_batch_from_split(
                SPLITS[split]['jsonl'], SPLITS[split]['cache'],
                ltg_specs, ltg_positions,
            ))
            rows = score_split(ckpt, split, injected_data, device)
            n_inj_total = sum(r['n_injected'] for r in rows)
            n_sites_total = sum(r['n_sites'] for r in rows)
            print(f'  [{ckpt}][{split}] {len(rows)} bags, '
                  f'{n_inj_total}/{n_sites_total} sites injected, '
                  f'elapsed {time.time()-t0:.1f}s')
            all_rows.extend(rows)

    with OUT.open('w') as f:
        for r in all_rows:
            f.write(json.dumps(r) + '\n')

    print(f'\n[out] {OUT}')

    # 3) Report cognate vs shuffled per ckpt
    print(f'\n{"="*95}')
    print(f'  LEVEL 2C — Gold-injected candidate pool: cognate vs shuffled')
    print(f'{"="*95}\n')
    print(f'  {"ckpt":<6} {"cog_n":>5} {"cog_med":>8} {"cog_mean":>9} '
          f'{"shu_n":>5} {"shu_med":>9} {"shu_mean":>10}   {"AUROC":>7}')
    from collections import defaultdict
    by = defaultdict(list)
    for r in all_rows: by[(r['ckpt'], r['split'])].append(r)
    for ck in CKPTS:
        cog = np.asarray([r['score'] for r in by[(ck, 'cognate')]])
        shu = np.asarray([r['score'] for r in by[(ck, 'shuffled')]])
        if not len(cog) or not len(shu): continue
        au = _auroc(np.concatenate([cog, shu]),
                    np.concatenate([np.ones(len(cog)), np.zeros(len(shu))]))
        print(f'  {ck:<6} {len(cog):>5} {np.median(cog):>+8.3f} {cog.mean():>+9.3f} '
              f'{len(shu):>5} {np.median(shu):>+9.3f} {shu.mean():>+10.3f}   {au:>7.4f}')

    print(f'\n  Reference (from earlier Durrant run, no injection):')
    print(f'    V5.2 baseline AUROC = 0.5714')
    print(f'    V6   baseline AUROC = 0.5498')
    print(f'    Oracle Hamming spec-vs-target AUROC = 0.9801')
    print(f'    Oracle raw junction matches         = 0.8207')


if __name__ == '__main__':
    main()
