"""Batch-probe: score every top-N candidate that is NOT IS110/GII, emit
per-candidate quality metrics + rank "looks real" candidates.

Metrics per candidate:
  - top_p, bag_max
  - peak sharpness: top1 / top2 ratio
  - per-site m_max at top_p: mean, min (all K sites should be > baseline)
  - zero-ablation deltas: structure, flank_bg, orient
  - localization: fraction of score retained in ±15 bp window
  - dominant ORF length, insertion size, n_events

Ranking heuristic ("real-ness score"):
  + peak_ratio (top1 vs top2): tight singular peak → looks localized
  + all-K m_max floor: all K sites must show >0.4 (cross-site coherence)
  + structure_drop: structure channels contribute > 0.5
  + localization_frac: score in ±15 bp is > 0.5
  + size in [700, 2500] bp: RNA-guided IS-family sizes get bonus
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.channel_b.data import ChannelBDataset
from model.channel_b.model import ChannelBModel
from model.channel_b.constants import CHANNELS, Ls, MAX_L


CH_IDX = {c: i for i, c in enumerate(CHANNELS)}
CH_MMAX = [CH_IDX[f"m_max_L{L}"] for L in Ls]
CH_FDEV = [CH_IDX[f"flank_dev_L{L}"] for L in Ls]
CH_STRUCT = [CH_IDX[c] for c in ("dG_open_uL_pn", "H_pair_win",
                                       "cooperativity_win_pn", "E_span_win",
                                       "structure_valid")]
CH_FBG = [CH_IDX["flank_bg_identity"]]

ADAPTED_JL = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                       "v84_evals/fna_ins_discovery/adapted.jsonl")
SHARD = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                  "v84_evals/fna_ins_discovery/adapted_shard")
CKPT = Path("checkpoints/channel_b/v84_main/best.pt")
CACHE = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/"
                  "fna_ins_discovery/cache")
PFAM_TSV = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                     "v84_evals/fna_ins_discovery/pfam_by_cluster.tsv")
INSERT_CDS = Path("/global/scratch/users/kh36969/fna_ins_discovery/"
                       "cds_v3/insert_cds.tsv")


def is_is110_or_gii(pfam: str) -> bool:
    """Return True if pfam string indicates IS110 or Group II intron."""
    hits = set(re.findall(r"([A-Za-z0-9_\-]+)\(", pfam)) if pfam else set()
    is110 = {"DEDD_Tnp_IS110", "Transposase_20"}
    gii = {"GIIM"}
    return bool(hits & is110) or bool(hits & gii)


def probe_one(model, device, ds, idx: int) -> dict:
    b = ds[idx]
    x = b.x
    sm = b.site_mask
    K_real = int(sm.sum().item())
    x_g = x.unsqueeze(0).to(device)
    sm_g = sm.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x_g, sm_g).squeeze(0).cpu().numpy()
    top_p = int(pred.argmax())
    top_v = float(pred.max())

    # Second-best peak (non-overlapping)
    scored = pred.copy()
    scored[max(0, top_p - 11):min(len(pred), top_p + 12)] = -1e9
    second_p = int(scored.argmax())
    second_v = float(scored.max())
    peak_ratio = top_v / max(second_v, 1e-6)

    # Per-site m_max at top_p — take max over L per site
    per_site_mmax = []
    for s in range(K_real):
        m_vals = [float(x[s, top_p, CH_MMAX[i]].item()) for i in range(len(Ls))]
        per_site_mmax.append(max(m_vals))
    m_mean = statistics.mean(per_site_mmax)
    m_min = min(per_site_mmax)

    # Structure ablation
    x_abl = x.clone()
    x_abl[..., CH_STRUCT] = 0.0
    with torch.no_grad():
        p2 = model(x_abl.unsqueeze(0).to(device), sm_g).squeeze(0).cpu().numpy()
    struct_drop = top_v - float(p2.max())

    # Localization: keep only nc[top_p-15, top_p+26)
    x_loc = x.clone()
    lo, hi = max(0, top_p - 15), min(x.shape[1], top_p + 26)
    mask_out = torch.ones(x.shape[1], dtype=torch.bool)
    mask_out[lo:hi] = False
    x_loc[:, mask_out, :] = 0.0
    with torch.no_grad():
        p3 = model(x_loc.unsqueeze(0).to(device), sm_g).squeeze(0).cpu().numpy()
    loc_frac = float(p3.max()) / max(top_v, 1e-6)

    return {
        "top_p": top_p, "top_v": top_v,
        "second_v": second_v, "peak_ratio": peak_ratio,
        "K": K_real,
        "m_mean_at_top": m_mean, "m_min_at_top": m_min,
        "struct_drop": struct_drop,
        "loc_frac": loc_frac,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=100)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"# batch probe of top-{args.top_n} candidates (excluding IS110 & GII)")
    print(f"# device={device}\n")

    # Load pfam + top-N
    top_rows = []
    with open(PFAM_TSV) as f:
        header = next(f)
        for i, line in enumerate(f):
            if i >= args.top_n:
                break
            parts = line.rstrip("\n").split("\t")
            cid, ns, tp, bmax = parts[0], int(parts[1]), int(parts[2]), float(parts[3])
            pfam = parts[5] if len(parts) >= 6 else ""
            top_rows.append({"cid": cid, "n_sites": ns, "top_p_orig": tp,
                                 "bag_max_orig": bmax, "pfam": pfam,
                                 "rank_in_top100": i + 1})

    excluded = [r for r in top_rows if is_is110_or_gii(r["pfam"])]
    to_probe = [r for r in top_rows if not is_is110_or_gii(r["pfam"])]
    print(f"# {len(top_rows)} top rows; {len(excluded)} IS110/GII excluded; "
          f"probing {len(to_probe)} clusters\n")

    # Metadata: size + ORF from insert_cds
    meta = {}
    with open(INSERT_CDS) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cid = row["cds_cluster_id"]
            meta.setdefault(cid, []).append({
                "inserted_len": int(row["inserted_len"]),
                "orf_len_aa": int(row["dominant_orf_len_aa"]),
                "species": row["species"],
                "dominant_orf": row["dominant_orf"],
            })

    # Load dataset (with cache — score with model)
    ds = ChannelBDataset(ADAPTED_JL, SHARD, preload_mt=True,
                            cache_dir=CACHE, sort_by_nc_len=False)
    idx_by_cid = {ds._bag_index[i][0]: i for i in range(len(ds))}

    # Load model
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    hp = ck.get("args", {})
    model = ChannelBModel(hidden=hp.get("hidden", 128),
                            n_heads=hp.get("n_heads", 4),
                            n_blocks=hp.get("n_blocks", 3)).to(device).eval()
    model.load_state_dict(ck["model"])

    # Probe each
    results = []
    for r in to_probe:
        cid = r["cid"]
        idx = idx_by_cid.get(cid)
        if idx is None:
            print(f"  [SKIP missing] {cid}")
            continue
        try:
            p = probe_one(model, device, ds, idx)
        except Exception as e:
            print(f"  [FAIL {cid}] {type(e).__name__}: {e}")
            continue
        events = meta.get(cid, [])
        med_ins = int(statistics.median(e["inserted_len"] for e in events)) if events else 0
        med_orf = int(statistics.median(e["orf_len_aa"] for e in events)) if events else 0
        n_species = len({e["species"] for e in events})
        n_events = len(events)
        # Composite "real-ness" score — heuristic
        realness = (
            (1.0 if p["peak_ratio"] >= 2.0 else p["peak_ratio"] / 2.0)   # sharp peak
            + (1.0 if p["m_min_at_top"] >= 0.5 else p["m_min_at_top"] * 2)  # all K sites match
            + (1.0 if p["struct_drop"] >= 0.5 else max(0, p["struct_drop"] / 0.5))  # structure real
            + (1.0 if p["loc_frac"] >= 0.6 else p["loc_frac"] / 0.6)  # local
            + (0.5 if 700 <= med_ins <= 2500 else 0)  # IS-size band bonus
            + (0.3 if n_species >= 2 else 0)   # multi-species bonus
        )
        results.append({**r, **p, "med_ins": med_ins, "med_orf": med_orf,
                              "n_species": n_species, "n_events": n_events,
                              "realness": realness})

    # Sort by realness descending
    results.sort(key=lambda x: -x["realness"])

    print(f"## RANKED by 'looks-real' heuristic (max 4.8)\n")
    print(f"  {'rank':>5s}  {'cds':<10s}  {'bag_max':>7s}  {'K':>2s}  "
              f"{'top_p':>5s}  {'peak':>6s}  {'m_min':>5s}  {'m_mean':>6s}  "
              f"{'strDrop':>7s}  {'locFr':>5s}  {'size':>5s}  {'orf':>4s}  "
              f"{'spc':>3s}  {'evt':>4s}  {'RS':>5s}  {'pfam':<40s}")
    for r in results:
        pfam_short = ",".join(re.findall(r"([A-Za-z0-9_\-]+)\(", r["pfam"])[:2]) or "-"
        print(f"  #{r['rank_in_top100']:>3d}   {r['cid']:<10s}  "
                f"{r['top_v']:>7.3f}  {r['K']:>2d}  "
                f"{r['top_p']:>5d}  {r['peak_ratio']:>6.2f}  "
                f"{r['m_min_at_top']:>5.2f}  {r['m_mean_at_top']:>6.2f}  "
                f"{r['struct_drop']:>7.2f}  {r['loc_frac']:>5.2f}  "
                f"{r['med_ins']:>4d}bp  {r['med_orf']:>3d}aa  "
                f"{r['n_species']:>3d}  {r['n_events']:>4d}  "
                f"{r['realness']:>5.2f}  {pfam_short:<40s}")

    print(f"\n# batch probe complete")


if __name__ == "__main__":
    main()
