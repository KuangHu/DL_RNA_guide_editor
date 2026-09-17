"""Non-RNA-guided negative control on real_data/negative_top10.

10 DDE-family FASTA pairs (empty/filled), grouped by insert_md5 →
natural multi-site bags. Score v7real_main; expect scores near
model's twin/shuffle level (LOW), not positive level.

Per site:
  flank    = empty[flank_L - 60 : flank_L + 60]        (120bp, junction at 60)
  nc       = filled[flank_L : flank_L + ins_len]       (the insertion, as
                                                          single-region nc)

Per bag (grouped by insert_md5, filtered to n_sites ∈ [3, 8]):
  all sites share the SAME nc (same insert_md5 = same insert sequence),
  each site has its own flank (its own genome context around the
  insertion junction).

Reports per-family bag_max_score distribution + is-family-distinguishable
from positive corpus AUROC (Durrant WT as positive comparator).
"""
from __future__ import annotations
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.channel_b.data import ChannelBDataset, bucket_collate_fn
from model.channel_b.model import ChannelBModel
from scripts.build_v7_shard import build_v7_shard


NEG_DIR = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/real_data/negative_top10")
FAMILIES = ["IS1", "IS3", "IS4", "IS5", "IS6", "IS66", "IS256", "IS481",
                "IS1595", "ISL3"]


def parse_fasta(path):
    """Yield (id, header_dict, sequence) tuples."""
    with open(path) as f:
        cur_id, cur_head, cur_seq = None, None, []
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if cur_id is not None:
                    yield cur_id, cur_head, "".join(cur_seq)
                # parse header
                parts = line[1:].split()
                cur_id = parts[0]
                cur_head = {}
                for tok in parts[1:]:
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        cur_head[k] = v
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            yield cur_id, cur_head, "".join(cur_seq)


def load_family(fam):
    """Return list of records: {id, flank_L, ins_len, empty_seq, insert_seq, insert_md5}."""
    empty_path = NEG_DIR / f"{fam}_empty.fna"
    filled_path = NEG_DIR / f"{fam}_filled.fna"
    empty_by_id = {i: (h, s) for i, h, s in parse_fasta(empty_path)}
    recs = []
    for i, h, s in parse_fasta(filled_path):
        if i not in empty_by_id: continue
        _, empty_s = empty_by_id[i]
        try:
            flank_L = int(h["flank_L"])
            ins_len = int(h["ins_len"])
            insert_md5 = h["insert_md5"]
        except (KeyError, ValueError):
            continue
        # extract insert from filled
        if len(s) < flank_L + ins_len: continue
        # v7-real training used nc_len U[100, 250]; DDE inserts are 500-3000bp.
        # Truncate to 250bp to match training regime (matched conditions,
        # comparable score distributions).
        insert_seq = s[flank_L:flank_L + ins_len][:250]
        # extract 120bp flank from empty around junction position flank_L
        if len(empty_s) < flank_L + 60 or flank_L < 60: continue
        flank_120 = empty_s[flank_L - 60:flank_L + 60]
        if len(flank_120) != 120: continue
        # Bases only (drop any Ns for cleanliness — v7-real training was ACGT)
        if any(b not in "ACGT" for b in flank_120): continue
        if any(b not in "ACGTN" for b in insert_seq): continue
        recs.append({
            "id": i, "insert_md5": insert_md5,
            "flank_L": flank_L, "ins_len": ins_len,
            "flank_120": flank_120, "insert_seq": insert_seq,
            "species": h.get("species", "unknown"),
        })
    return recs


def group_and_sample(recs, min_sites=3, max_sites=8, n_bags=200, rng=None):
    by_md5 = defaultdict(list)
    for r in recs:
        by_md5[r["insert_md5"]].append(r)
    eligible = [(md5, sites) for md5, sites in by_md5.items()
                    if min_sites <= len(sites) <= max_sites]
    if rng is not None:
        sample = rng.sample(eligible, min(n_bags, len(eligible)))
    else:
        sample = eligible[:n_bags]
    return sample, len(eligible)


def emit_jsonl_bag(bag_id, sites, fout):
    """Emit v7-real-schema records for a bag; all sites share nc = first
    site's insert_seq (they're identical by md5)."""
    nc = sites[0]["insert_seq"]
    for i, r in enumerate(sites):
        rec = {
            "site_id":       f"{bag_id}_site_{i:04d}",
            "transposase_id": bag_id,
            "ncrna_id":       f"{bag_id}_ncrna",
            "inputs": {
                "flank":              r["flank_120"],
                "noncoding_regions":  [nc],
            },
            "labels": {
                "is_positive":                 False,
                "target_position_in_flank":    None,
                "guide_length":                11,
                "num_noncoding_regions":       1,
                "active_noncoding_index":      0,
                "guide_span_in_active_noncoding": None,
                "ncrna_length":                len(nc),
                "arch": {
                    "n_sites":                     len(sites),
                    "nc_multi_region_scoring":     "concat_with_N_spacer",
                    "nc_homology_rate":            1.0,
                },
            },
        }
        fout.write(json.dumps(rec) + "\n")


def score_bags(model, device, jsonl_path, shard_dir):
    if shard_dir.exists():
        import shutil; shutil.rmtree(shard_dir)
    build_v7_shard(jsonl_path, shard_dir, min_sites=1, family_label=jsonl_path.stem)
    ds = ChannelBDataset(jsonl_path, shard_dir, preload_mt=True, cache_dir=None)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2,
                            collate_fn=bucket_collate_fn, pin_memory=True)
    scores = []
    for batch in loader:
        x = batch["x"].to(device); sm = batch["site_mask"].to(device)
        pm = batch["pos_mask"].to(device)
        with torch.no_grad():
            pred = model(x, sm)
        pm_score = (pred * pm.float() + (-1e9) * (~pm).float()).max(dim=1).values
        for i in range(pm_score.shape[0]):
            scores.append(float(pm_score[i].item()))
    return scores


def score_positive_corpus(model, device, jsonl_path, cache_root, out_root, n_bags=200,
                                seed=0):
    """Score bags from the standard v7-real val pool as positive comparator."""
    from scripts.channel_b_train import build_split
    cfg_json = json.loads(Path("config/channel_b_v7real_corpora.json").read_text())
    batch_root = Path(cfg_json["batch_root"]); shard_root = Path(cfg_json["shard_root"])
    # Just pos50k val bags
    ds = ChannelBDataset(batch_root / "v7real_pos50k.jsonl",
                              shard_root / "v7real_pos50k_shard",
                              preload_mt=True,
                              cache_dir=cache_root / "pos50k")
    _tr, va = build_split(ds, "main", seed=seed)
    rng = random.Random(seed)
    picks = rng.sample(list(va.indices), min(n_bags, len(va.indices)))
    scores = []
    for local_i in picks:
        item = ds[local_i]
        x = item.x.unsqueeze(0).to(device)
        sm = item.site_mask.unsqueeze(0).to(device)
        pm = torch.ones((1, x.shape[2]), dtype=torch.bool, device=device)
        # actually pos_mask is per-position; the loader gives us item.x already
        # padded to nc_len_eff so all positions are valid → pm all True.
        with torch.no_grad():
            pred = model(x, sm).squeeze(0)
        scores.append(float(pred.max().item()))
    return scores


def summarize(name, scores):
    if not scores:
        print(f"  {name}: NO SCORES")
        return
    arr = np.array(scores)
    q = np.percentile(arr, [5, 25, 50, 75, 95])
    print(f"  {name:<12s}  n={len(arr):>4d}  "
              f"p05={q[0]:+.3f}  p25={q[1]:+.3f}  p50={q[2]:+.3f}  "
              f"p75={q[3]:+.3f}  p95={q[4]:+.3f}  mean={arr.mean():+.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                     default=Path("checkpoints/channel_b/v7real_main/best.pt"))
    ap.add_argument("--n-bags-per-family", type=int, default=200)
    ap.add_argument("--n-bags-positive", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", type=Path,
                     default=Path("/global/scratch/users/kh36969/tmp/nrg_ctrl_top10"))
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    hp = ckpt.get("hparams", {})
    model = ChannelBModel(hidden=hp.get("hidden", 128),
                                n_heads=hp.get("n_heads", 4),
                                n_blocks=hp.get("n_blocks", 3)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[nrg-top10] device={device}  ckpt={args.ckpt}")

    rng = random.Random(args.seed)
    all_scores = {}
    for fam in FAMILIES:
        try:
            recs = load_family(fam)
            print(f"\n[{fam}] {len(recs)} records parsed")
            sample, n_eligible = group_and_sample(recs, 3, 8,
                                                            args.n_bags_per_family, rng)
            print(f"  {n_eligible} bags with n_sites ∈ [3, 8]; sampling {len(sample)}")
            jl = args.out_root / f"{fam}.jsonl"
            with open(jl, "w") as f:
                for i, (md5, sites) in enumerate(sample):
                    emit_jsonl_bag(f"{fam}_bag_{i:04d}", sites, f)
            shard_dir = args.out_root / f"{fam}_shard"
            scores = score_bags(model, device, jl, shard_dir)
            all_scores[fam] = scores
            print(f"  scored {len(scores)} bags")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [{fam}] FAIL: {type(e).__name__}: {e}")

    # v7-real val pos50k as positive comparator
    print(f"\n[POS_pos50k] scoring positive comparator from v7real val pool ...")
    try:
        cache_root = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                             "channel_b_cache_v7real")
        pos_scores = score_positive_corpus(model, device, None, cache_root,
                                                    args.out_root, args.n_bags_positive,
                                                    args.seed)
        all_scores["POS_pos50k"] = pos_scores
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  POS FAIL: {type(e).__name__}: {e}")

    print(f"\n=== BAG SCORE DISTRIBUTION ===")
    for fam in FAMILIES + ["POS_pos50k"]:
        summarize(fam, all_scores.get(fam, []))

    # AUROC per family vs POS
    if "POS_pos50k" in all_scores and all_scores["POS_pos50k"]:
        pos = np.array(all_scores["POS_pos50k"])
        print(f"\n=== AUROC (positive-label = POS_pos50k) ===")
        for fam in FAMILIES:
            if fam not in all_scores or not all_scores[fam]: continue
            neg = np.array(all_scores[fam])
            y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
            s = np.concatenate([pos, neg])
            auc = float(roc_auc_score(y, s))
            print(f"  POS vs {fam:<8s}  AUROC = {auc:.4f}  (n_pos={len(pos)} n_neg={len(neg)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
