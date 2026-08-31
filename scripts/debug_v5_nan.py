"""Reproduce V5 NaN with real batches; localize which stage produces NaN.

Loads val_v4_no_l3 in the same way the trainer does. Runs a few batches
through V5 (use_dispersion=True), inspects disp_phi and each intermediate
tensor, prints the culprit tensor when NaN first appears.
"""
import sys
sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')

import torch
from torch.utils.data import DataLoader
from model.v1 import V1Config, V1Model, _compute_dispersion_features
from preprocess.site import StructureCache
from preprocess.tnp_dataset import TnpGroupedDataset, collate_tnp_batch, make_torch_tnp_dataset

BASE = '/global/scratch/users/kh36969/DL_novel_guide_editor'


def check_nan(name, t):
    if t is None:
        print(f'  {name}: None')
        return False
    if torch.is_tensor(t):
        n_nan = torch.isnan(t).sum().item()
        n_inf = torch.isinf(t).sum().item()
        stat = f'shape={tuple(t.shape)} dtype={t.dtype} min={t.float().min().item():.3g} max={t.float().max().item():.3g}'
        if n_nan or n_inf:
            print(f'  {name}: NaN={n_nan} Inf={n_inf}  {stat}')
            return True
        else:
            print(f'  {name}: ok  {stat}')
    return False


def main():
    torch.manual_seed(0)
    device = torch.device('cuda')
    cfg = V1Config(use_dispersion=True, disp_hidden=32)
    model = V1Model(cfg).to(device)
    model.train()

    # Load a few batches (same setup as training)
    cache = StructureCache(f'{BASE}/structure/val_v4_u16.index.json')
    ds = TnpGroupedDataset(f'{BASE}/splits/val_v4_no_l3.jsonl', cache,
                            site_subsample_size=16, rng_seed=0)
    dl = DataLoader(make_torch_tnp_dataset(ds), batch_size=8, shuffle=True,
                    num_workers=2, collate_fn=lambda x: collate_tnp_batch(x, to_torch=True))

    for bi, b in enumerate(dl):
        if bi >= 8:
            break
        b_dev = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in b.items()}
        print(f'\n===== batch {bi} =====')

        # Forward with autocast (as trainer does)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(b_dev['candidate_patches'], b_dev['candidate_features'],
                         b_dev['candidate_mask'], b_dev['nc_region_mask'])

        got_nan = False
        got_nan |= check_nan('cand_raw', out['cand_raw'])
        got_nan |= check_nan('nc_attn', out['nc_attn'])
        got_nan |= check_nan('disp_phi', out['disp_phi'])
        got_nan |= check_nan('logit', out['logit'])

        if got_nan or bi < 2:
            # Detailed inspection of disp_phi computation
            print('  --- manual dispersion computation ---')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                phi = _compute_dispersion_features(
                    out['cand_raw'].detach(),
                    b_dev['candidate_features'].detach(),
                    b_dev['candidate_mask'],
                    out['nc_attn'].detach(),
                )
            print(f'  disp_phi (float32): {phi}')
            for i, name in enumerate(['pos_MAD', 'pos_STD', 'pos_IQR', 'ncstart_STD', 'L_STD', 'orient_H']):
                col = phi[:, i]
                n_nan = torch.isnan(col).sum().item()
                n_inf = torch.isinf(col).sum().item()
                print(f'    {name}: min={col.min().item():.4g} max={col.max().item():.4g} nan={n_nan} inf={n_inf}')

        if got_nan:
            print('=== NaN FOUND, stopping ===')
            break


if __name__ == '__main__':
    main()
