"""Evaluate best V1-on-V3 checkpoint on test_v3.jsonl."""
import json, time, sys, os
sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')
import numpy as np, torch
from torch.utils.data import DataLoader
from model.v1 import V1Config, V1Model
from preprocess.site import StructureCache
from preprocess.tnp_dataset import TnpGroupedDataset, collate_tnp_batch, make_torch_tnp_dataset
from training.metrics import tnp_metrics, stratified_auroc, candidate_recall, nc_selection_accuracy, _auroc, _auprc
from training.train_v1 import _violation_profile_by_tnp, _tnp_strength_by_tnp, EASY_PROFILES

BASE = '/global/scratch/users/kh36969/DL_novel_guide_editor'
CKPT = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/checkpoints/v1_on_v3/best.pt'

device = torch.device('cuda')
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
cfg = V1Config(**ckpt['cfg'])
model = V1Model(cfg).to(device); model.load_state_dict(ckpt['model']); model.eval()
print(f'loaded ckpt: epoch {ckpt["epoch"]}, val AUPRC={ckpt["auprc"]:.4f}')

cache = StructureCache(f'{BASE}/structure/test_v3_u16.index.json')
ds = TnpGroupedDataset(f'{BASE}/splits/test_v3.jsonl', cache, site_subsample_size=50, rng_seed=0)
print(f'test tnps={len(ds)}')
dl = DataLoader(make_torch_tnp_dataset(ds), batch_size=8, shuffle=False, num_workers=4,
                collate_fn=lambda x: collate_tnp_batch(x, to_torch=True),
                persistent_workers=True, pin_memory=True)
gmap = _violation_profile_by_tnp(f'{BASE}/splits/test_v3.jsonl')
smap = _tnp_strength_by_tnp(f'{BASE}/splits/test_v3.jsonl')

t0 = time.time()
scores, labels, tnp_ids = [], [], []
cand_at_active, true_slot_all, active_all, nc_attn_all = [], [], [], []
with torch.no_grad():
    for b in dl:
        b = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(b['candidate_patches'], b['candidate_features'], b['candidate_mask'], b['nc_region_mask'])
        scores.append(torch.sigmoid(out['logit']).float().cpu().numpy())
        labels.append(b['is_positive'].cpu().numpy())
        tnp_ids.extend(list(b['tnp_id']))
        cr = out['cand_raw'].float().cpu().numpy(); na = out['nc_attn'].float().cpu().numpy()
        ac = b['active_nc_index'].cpu().numpy(); ts = b['true_slot_idx'].cpu().numpy()
        B, S, N, K = cr.shape
        for bi in range(B):
            for si in range(S):
                if int(ac[bi,si]) < 0 or int(ts[bi,si]) < 0: continue
                cand_at_active.append(cr[bi, si, int(ac[bi,si])])
                true_slot_all.append(int(ts[bi,si])); active_all.append(int(ac[bi,si]))
                nc_attn_all.append(na[bi, si])

scores = np.concatenate(scores); labels = np.concatenate(labels)
groups = np.asarray([gmap[t] for t in tnp_ids])
m = tnp_metrics(scores, labels)
strat = stratified_auroc(scores, labels, groups)
cand = candidate_recall(np.stack(cand_at_active, 0), true_slot_all, ks=(1,5,10))
nc = nc_selection_accuracy(np.stack(nc_attn_all, 0), active_all)

# Hard-only
labels_bool = labels.astype(bool)
hard_mask = labels_bool | np.array([g not in EASY_PROFILES for g in groups])
s_h, y_h = scores[hard_mask], labels_bool[hard_mask]
hard_auroc = _auroc(s_h, y_h); hard_auprc = _auprc(s_h, y_h)

# Weak-positive recall
strengths = np.asarray([smap.get(t, 'unknown') for t in tnp_ids])
called_pos = scores > 0.5
recall_by_strength = {}
for lvl in ('strong','moderate','weak'):
    mask = labels_bool & (strengths == lvl)
    if mask.any():
        recall_by_strength[lvl] = (float(called_pos[mask].mean()), int(mask.sum()))

print(f'test eval done in {time.time()-t0:.1f}s')
print()
print(f'  n_tnp_pos={m["n_pos"]}  n_tnp_neg={m["n_neg"]}')
print(f'  AUROC={m["auroc"]:.4f}    AUPRC={m["auprc"]:.4f}')
print(f'  HARD_AUROC={hard_auroc:.4f}    HARD_AUPRC={hard_auprc:.4f}   (n_hard_neg={int((~y_h).sum())})')
print(f'  R@1={cand["recall@1"]:.3f}  R@5={cand["recall@5"]:.3f}  R@10={cand["recall@10"]:.3f}   (n={cand["n"]})')
print(f'  NC top-1: {nc["nc_top1"]:.3f}   (n={nc["n"]})')
print()
print(f'  Weak-positive recall @ threshold 0.5:')
for lvl, (r, n) in recall_by_strength.items():
    print(f'    {lvl:<10} {r:.4f}   (n={n})')
print()
print(f'  Per-profile AUROC:')
for k in sorted(strat):
    print(f'    {k[6:-1]:<45} {strat[k]:.4f}')
