"""Evaluate best V1 checkpoint on the held-out test.jsonl."""
import argparse, json, time, math, os, sys
sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')
import numpy as np, torch
from torch.utils.data import DataLoader
from model.v1 import V1Config, V1Model
from preprocess.site import StructureCache
from preprocess.tnp_dataset import TnpGroupedDataset, collate_tnp_batch, make_torch_tnp_dataset
from training.metrics import tnp_metrics, stratified_auroc, candidate_recall, nc_selection_accuracy

BASE = '/global/scratch/users/kh36969/DL_novel_guide_editor'
CKPT = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/checkpoints/v1_1178692/best.pt'

def _to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}

def group_map(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            t = r['transposase_id']
            if t not in out:
                out[t] = 'positive' if r['labels'].get('is_positive') else (r['labels'].get('violation_profile') or 'unknown')
    return out

device = torch.device('cuda')
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
cfg = V1Config(**ckpt['cfg'])
model = V1Model(cfg).to(device)
model.load_state_dict(ckpt['model'])
model.eval()
print(f'loaded ckpt: epoch {ckpt["epoch"]}, saved AUPRC={ckpt["auprc"]:.4f}')

cache = StructureCache(f'{BASE}/structure/test_u16.index.json')
ds = TnpGroupedDataset(f'{BASE}/splits/test.jsonl', cache, site_subsample_size=50, rng_seed=0)
print(f'test: {len(ds)} tnps')
dl = DataLoader(
    make_torch_tnp_dataset(ds),
    batch_size=8, shuffle=False, num_workers=4,
    collate_fn=lambda x: collate_tnp_batch(x, to_torch=True),
    persistent_workers=True, pin_memory=True,
)
gmap = group_map(f'{BASE}/splits/test.jsonl')

t0 = time.time()
scores, labels, tnp_ids = [], [], []
cand_at_active, true_slot_all, active_all, nc_attn_all = [], [], [], []
with torch.no_grad():
    for batch in dl:
        batch = _to_device(batch, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(batch['candidate_patches'], batch['candidate_features'],
                         batch['candidate_mask'], batch['nc_region_mask'])
        scores.append(torch.sigmoid(out['logit']).float().cpu().numpy())
        labels.append(batch['is_positive'].cpu().numpy())
        tnp_ids.extend(list(batch['tnp_id']))
        cr = out['cand_raw'].float().cpu().numpy()
        na = out['nc_attn'].float().cpu().numpy()
        ac = batch['active_nc_index'].cpu().numpy()
        ts = batch['true_slot_idx'].cpu().numpy()
        B, S, N, K = cr.shape
        for bi in range(B):
            for si in range(S):
                if int(ac[bi, si]) < 0 or int(ts[bi, si]) < 0: continue
                cand_at_active.append(cr[bi, si, int(ac[bi, si])])
                true_slot_all.append(int(ts[bi, si]))
                active_all.append(int(ac[bi, si]))
                nc_attn_all.append(na[bi, si])

scores = np.concatenate(scores); labels = np.concatenate(labels)
groups = np.asarray([gmap[t] for t in tnp_ids])
m = tnp_metrics(scores, labels)
strat = stratified_auroc(scores, labels, groups)
cand = candidate_recall(np.stack(cand_at_active, 0), true_slot_all, ks=(1,5,10))
nc = nc_selection_accuracy(np.stack(nc_attn_all, 0), active_all)
print(f'test eval done in {time.time()-t0:.1f}s')
print(f'  n_tnp_pos={m["n_pos"]}  n_tnp_neg={m["n_neg"]}')
print(f'  AUROC={m["auroc"]:.4f}  AUPRC={m["auprc"]:.4f}')
print(f'  R@1={cand["recall@1"]:.3f}  R@5={cand["recall@5"]:.3f}  R@10={cand["recall@10"]:.3f}   (n={cand["n"]})')
print(f'  NC top-1: {nc["nc_top1"]:.3f}   (n={nc["n"]})')
for k in sorted(strat):
    print(f'  {k}: {strat[k]:.4f}')
