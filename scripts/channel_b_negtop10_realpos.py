"""Sanity extension to negtop10: score Durrant WT as REAL positive
comparator vs 10 DDE families. Also report n_sites + nc_len marginals
per corpus to expose any synth-vs-real confound.

Question: does the 0.956-0.977 AUROC(pos50k vs DDE) reflect
"RNA-guided vs non-RNA-guided" or "synth vs real"?
Answer test: Durrant WT is real natural bridge RNA data; if
AUROC(Durrant-WT vs DDE) is also ≥ 0.90, confound rejected.
"""
from __future__ import annotations
import argparse
import json
import random
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
from model.channel_b.constants import MAX_L
from scripts.build_v7_shard import build_v7_shard


NEG_DIR = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/real_data/negative_top10")
FAMILIES = ["IS1", "IS3", "IS4", "IS5", "IS6", "IS66", "IS256", "IS481",
                "IS1595", "ISL3"]
DURRANT_WT_JSONL = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                              "IS110_gold/inference/durrant_wt_realbg_v2.jsonl")


# ------- DDE parsing (same as negtop10 script, truncated insert) -------
def parse_fasta(path):
    with open(path) as f:
        cur_id, cur_head, cur_seq = None, None, []
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if cur_id is not None:
                    yield cur_id, cur_head, "".join(cur_seq)
                parts = line[1:].split()
                cur_id = parts[0]
                cur_head = {}
                for tok in parts[1:]:
                    if "=" in tok:
                        k, v = tok.split("=", 1); cur_head[k] = v
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            yield cur_id, cur_head, "".join(cur_seq)


def load_dde_family(fam):
    empty_path = NEG_DIR / f"{fam}_empty.fna"
    filled_path = NEG_DIR / f"{fam}_filled.fna"
    empty_by_id = {i: (h, s) for i, h, s in parse_fasta(empty_path)}
    recs = []
    for i, h, s in parse_fasta(filled_path):
        if i not in empty_by_id: continue
        _, empty_s = empty_by_id[i]
        try:
            flank_L = int(h["flank_L"]); ins_len = int(h["ins_len"])
            insert_md5 = h["insert_md5"]
        except (KeyError, ValueError):
            continue
        if len(s) < flank_L + ins_len: continue
        insert_seq = s[flank_L:flank_L + ins_len][:250]   # truncate
        if len(empty_s) < flank_L + 60 or flank_L < 60: continue
        flank_120 = empty_s[flank_L - 60:flank_L + 60]
        if len(flank_120) != 120: continue
        if any(b not in "ACGT" for b in flank_120): continue
        if any(b not in "ACGTN" for b in insert_seq): continue
        recs.append({"id": i, "insert_md5": insert_md5,
                        "flank_L": flank_L, "ins_len": ins_len,
                        "flank_120": flank_120, "insert_seq": insert_seq})
    return recs


def emit_dde_bag(bag_id, sites, fout):
    nc = sites[0]["insert_seq"]
    for i, r in enumerate(sites):
        rec = {
            "site_id":       f"{bag_id}_site_{i:04d}",
            "transposase_id": bag_id,
            "ncrna_id":       f"{bag_id}_ncrna",
            "inputs": {"flank": r["flank_120"], "noncoding_regions": [nc]},
            "labels": {
                "is_positive":                 False,
                "target_position_in_flank":    None,
                "guide_length":                11,
                "num_noncoding_regions":       1,
                "active_noncoding_index":      0,
                "guide_span_in_active_noncoding": None,
                "ncrna_length":                len(nc),
                "arch": {"n_sites": len(sites),
                            "nc_multi_region_scoring": "concat_with_N_spacer",
                            "nc_homology_rate": 1.0},
            },
        }
        fout.write(json.dumps(rec) + "\n")


# ------- Durrant WT bag builder (drop-in) -------
def load_durrant_wt():
    return [json.loads(l) for l in open(DURRANT_WT_JSONL)]


def emit_durrant_bag(bag_id, sites, fout):
    """All WT records share nc = IS621 [nc_region_1, nc_region_2].
    Preserve multi-region schema."""
    for i, r in enumerate(sites):
        new = dict(r)
        new["site_id"] = f"{bag_id}_site_{i:04d}"
        new["transposase_id"] = bag_id
        L = dict(new["labels"])
        arch = dict(L.get("arch") or {})
        arch["n_sites"] = len(sites)
        arch["nc_multi_region_scoring"] = "concat_with_N_spacer"
        L["arch"] = arch
        new["labels"] = L
        fout.write(json.dumps(new) + "\n")


# ------- scoring helper -------
def score_from_jsonl(model, device, jsonl_path, shard_dir):
    if shard_dir.exists():
        import shutil; shutil.rmtree(shard_dir)
    build_v7_shard(jsonl_path, shard_dir, min_sites=1, family_label=jsonl_path.stem)
    ds = ChannelBDataset(jsonl_path, shard_dir, preload_mt=True, cache_dir=None)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2,
                            collate_fn=bucket_collate_fn, pin_memory=True)
    scores, ns_list, nc_len_list = [], [], []
    for batch in loader:
        x = batch["x"].to(device); sm = batch["site_mask"].to(device)
        pm = batch["pos_mask"].to(device)
        with torch.no_grad():
            pred = model(x, sm)
        pm_score = (pred * pm.float() + (-1e9) * (~pm).float()).max(dim=1).values
        ns_batch = sm.sum(dim=1).cpu().numpy()
        for i in range(pm_score.shape[0]):
            scores.append(float(pm_score[i].item()))
            ns_list.append(int(ns_batch[i]))
            nc_len_list.append(int(pm[i].sum().item()))
    return scores, ns_list, nc_len_list


def score_pos50k_val(model, device, cache_root, n_bags, seed):
    from scripts.channel_b_train import build_split
    cfg = json.loads(Path("config/channel_b_v7real_corpora.json").read_text())
    batch_root = Path(cfg["batch_root"]); shard_root = Path(cfg["shard_root"])
    ds = ChannelBDataset(batch_root / "v7real_pos50k.jsonl",
                              shard_root / "v7real_pos50k_shard",
                              preload_mt=True, cache_dir=cache_root / "pos50k")
    _tr, va = build_split(ds, "main", seed=seed)
    rng = random.Random(seed)
    picks = rng.sample(list(va.indices), min(n_bags, len(va.indices)))
    scores, ns_list, nc_len_list = [], [], []
    for local_i in picks:
        item = ds[local_i]
        x = item.x.unsqueeze(0).to(device)
        sm = item.site_mask.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x, sm).squeeze(0)
        scores.append(float(pred.max().item()))
        ns_list.append(int(item.site_mask.sum().item()))
        nc_len_list.append(int(item.x.shape[1]))
    return scores, ns_list, nc_len_list


def summarize(name, scores, ns_list, nc_len_list):
    if not scores:
        print(f"  {name}: NO SCORES"); return
    sc = np.array(scores); ns = np.array(ns_list); nl = np.array(nc_len_list)
    print(f"  {name:<15s}  n={len(sc):>3d}  score p50={np.median(sc):+.3f} "
              f"p95={np.percentile(sc,95):+.3f}  "
              f"n_sites p50={np.median(ns):.0f} min={ns.min()} max={ns.max()}  "
              f"nc_len_eff p50={np.median(nl):.0f} min={nl.min()} max={nl.max()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                     default=Path("checkpoints/channel_b/v7real_main/best.pt"))
    ap.add_argument("--n-bags-per-family", type=int, default=50)
    ap.add_argument("--n-bags-pos50k", type=int, default=200)
    ap.add_argument("--n-bags-durrant", type=int, default=50)
    ap.add_argument("--durrant-n-sites-range", type=str, default="3,8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", type=Path,
                     default=Path("/global/scratch/users/kh36969/tmp/nrg_ctrl_realpos"))
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
    print(f"[nrg-realpos] device={device}")

    rng = random.Random(args.seed)
    all_scores, all_ns, all_nl = {}, {}, {}

    # DDE families
    for fam in FAMILIES:
        try:
            recs = load_dde_family(fam)
            by_md5 = defaultdict(list)
            for r in recs: by_md5[r["insert_md5"]].append(r)
            eligible = [(m, s) for m, s in by_md5.items() if 3 <= len(s) <= 8]
            sample = rng.sample(eligible, min(args.n_bags_per_family, len(eligible)))
            jl = args.out_root / f"{fam}.jsonl"
            with open(jl, "w") as f:
                for i, (md5, sites) in enumerate(sample):
                    emit_dde_bag(f"{fam}_bag_{i:04d}", sites, f)
            shard = args.out_root / f"{fam}_shard"
            s, ns, nl = score_from_jsonl(model, device, jl, shard)
            all_scores[fam] = s; all_ns[fam] = ns; all_nl[fam] = nl
            print(f"  [{fam}] scored {len(s)} bags")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [{fam}] FAIL: {e}")

    # Durrant WT — REAL RNA-guided positive
    ns_lo, ns_hi = [int(x) for x in args.durrant_n_sites_range.split(",")]
    try:
        wt_recs = load_durrant_wt()
        rng2 = random.Random(args.seed + 1)
        jl = args.out_root / "DurrantWT.jsonl"
        with open(jl, "w") as f:
            for i in range(args.n_bags_durrant):
                K = rng2.randint(ns_lo, ns_hi)
                picks = rng2.sample(wt_recs, K)
                emit_durrant_bag(f"DurrantWT_bag_{i:04d}", picks, f)
        shard = args.out_root / "DurrantWT_shard"
        s, ns, nl = score_from_jsonl(model, device, jl, shard)
        all_scores["DurrantWT"] = s
        all_ns["DurrantWT"] = ns
        all_nl["DurrantWT"] = nl
        print(f"  [DurrantWT] scored {len(s)} bags")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  [DurrantWT] FAIL: {e}")

    # pos50k synth positive
    try:
        cache_root = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                             "channel_b_cache_v7real")
        s, ns, nl = score_pos50k_val(model, device, cache_root,
                                             args.n_bags_pos50k, args.seed)
        all_scores["pos50k_synth"] = s
        all_ns["pos50k_synth"] = ns
        all_nl["pos50k_synth"] = nl
        print(f"  [pos50k_synth] scored {len(s)} bags")
    except Exception as e:
        import traceback; traceback.print_exc()

    print(f"\n=== DISTRIBUTIONS ===")
    for name in FAMILIES + ["DurrantWT", "pos50k_synth"]:
        summarize(name, all_scores.get(name, []), all_ns.get(name, []), all_nl.get(name, []))

    # AUROC against BOTH positive comparators
    print(f"\n=== AUROC (real+real: DurrantWT vs DDE) ===")
    if "DurrantWT" in all_scores:
        pos = np.array(all_scores["DurrantWT"])
        for fam in FAMILIES:
            if fam not in all_scores: continue
            neg = np.array(all_scores[fam])
            y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
            s = np.concatenate([pos, neg])
            auc = float(roc_auc_score(y, s))
            print(f"  DurrantWT vs {fam:<8s}  AUROC = {auc:.4f}  "
                      f"(n_pos={len(pos)} n_neg={len(neg)})")
    print(f"\n=== AUROC (synth+real: pos50k_synth vs DDE) — reference ===")
    if "pos50k_synth" in all_scores:
        pos = np.array(all_scores["pos50k_synth"])
        for fam in FAMILIES:
            if fam not in all_scores: continue
            neg = np.array(all_scores[fam])
            y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
            s = np.concatenate([pos, neg])
            auc = float(roc_auc_score(y, s))
            print(f"  pos50k_synth vs {fam:<8s}  AUROC = {auc:.4f}  "
                      f"(n_pos={len(pos)} n_neg={len(neg)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
