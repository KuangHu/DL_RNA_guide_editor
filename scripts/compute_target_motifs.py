"""For each candidate with a known v84 top_p, compute the per-site TARGET MOTIF
in the flank — the position in each site's 120bp flank where the guide at
nc[top_p..top_p+L] best matches.

Reads match_table shards (same ones the model uses — flank_argmax_by_excl=0,
the deploy-legal, orient-ambiguous version), extracts per-site:
  - flank_argmax (position in flank where best match landed)
  - m_max (number of matches at that position)
  - flank sequence (120bp joined, from the site record)
  - target_seq = flank[argmax..argmax+L]

Writes `target_motifs.tsv` per candidate folder.

Usage:
  python3 scripts/compute_target_motifs.py --cds CDS01040
  python3 scripts/compute_target_motifs.py --all
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.channel_b.data import ChannelBDataset
from model.channel_b.constants import Ls

CAND_ROOT = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/"
                        "fna_ins_discovery/candidate")
ADAPTED_JL = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                        "v84_evals/fna_ins_discovery/adapted.jsonl")
SHARD = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                   "v84_evals/fna_ins_discovery/adapted_shard")


def parse_top_p(cand_dir: Path) -> int | None:
    pt = cand_dir / "v84_probe.txt"
    if not pt.exists():
        return None
    import re
    m = re.search(r"^# TOP: pos=(\d+)", pt.read_text(), re.M)
    return int(m.group(1)) if m else None


def extract_target_motifs(ds: ChannelBDataset, cid: str, top_p: int) -> list[dict]:
    """For the bag with bag_id=cid, read per-site flank_argmax at nc[top_p]
    across all Ls and return per-site target-motif records."""
    bag_sites = ds._sites_by_bag.get(cid, [])
    if not bag_sites:
        return []

    records = []
    for s_idx, site in enumerate(bag_sites):
        flank = site.get("inputs", {}).get("flank", "")
        site_id = site.get("site_id", f"site_{s_idx}")
        if not flank:
            continue
        row = {"site_idx": s_idx, "site_id": site_id, "flank_len": len(flank)}
        for L_i, L in enumerate(Ls):
            try:
                ma_fwd = ds._mt.get(cid, s_idx, "fwd", L)
                ma_rc = ds._mt.get(cid, s_idx, "rc", L)
            except Exception as e:
                row[f"L{L}"] = f"ERR:{e}"
                continue
            m_fwd = ma_fwd.m_max_by_excl.get(0)
            m_rc = ma_rc.m_max_by_excl.get(0)
            a_fwd = ma_fwd.flank_argmax_by_excl.get(0)
            a_rc = ma_rc.flank_argmax_by_excl.get(0)
            if m_fwd is None or m_rc is None or a_fwd is None or a_rc is None:
                row[f"L{L}_argmax"] = None
                continue
            if top_p >= min(len(m_fwd), len(m_rc)):
                row[f"L{L}_argmax"] = None
                continue
            # pick winning orient at top_p
            if m_fwd[top_p] >= m_rc[top_p]:
                argmax = int(a_fwd[top_p])
                m_val = int(m_fwd[top_p])
                orient = "fwd"
            else:
                argmax = int(a_rc[top_p])
                m_val = int(m_rc[top_p])
                orient = "rc"
            target_seq = flank[argmax:argmax + L] if argmax + L <= len(flank) else flank[argmax:]
            row[f"L{L}_argmax"] = argmax
            row[f"L{L}_m"] = m_val
            row[f"L{L}_orient"] = orient
            row[f"L{L}_seq"] = target_seq
        records.append(row)
    return records


def write_tsv(records: list[dict], out: Path) -> None:
    if not records:
        return
    fields = ["site_idx", "site_id", "flank_len"]
    for L in Ls:
        fields += [f"L{L}_argmax", f"L{L}_m", f"L{L}_orient", f"L{L}_seq"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in fields})


def summarize(records: list[dict]) -> dict:
    """Report consensus of flank_argmax across sites (for L=14)."""
    if not records:
        return {}
    argmaxes = [r.get("L14_argmax") for r in records if r.get("L14_argmax") is not None]
    if not argmaxes:
        return {"n_sites_with_argmax": 0}
    seqs = [r.get("L14_seq", "") for r in records if r.get("L14_seq")]
    return {
        "n_sites_with_argmax": len(argmaxes),
        "argmax_min": min(argmaxes),
        "argmax_median": int(np.median(argmaxes)),
        "argmax_max": max(argmaxes),
        "argmax_std": float(np.std(argmaxes)),
        "n_unique_seqs": len(set(seqs)),
        "example_seqs": seqs[:4],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cds", default="")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    cids = []
    if args.all:
        for d in sorted(CAND_ROOT.iterdir()):
            if d.is_dir() and d.name.startswith("CDS"):
                cids.append(d.name)
    else:
        cids = [args.cds]

    # Load dataset once (ChannelBDataset preload_mt=True loads match tables)
    print(f"# loading ChannelBDataset from {SHARD}")
    ds = ChannelBDataset(ADAPTED_JL, SHARD, preload_mt=True,
                                cache_dir=None, sort_by_nc_len=False)
    print(f"# dataset: {len(ds)} bags")

    for cid in cids:
        folder = CAND_ROOT / cid
        if not folder.exists():
            print(f"  [SKIP {cid}] no folder")
            continue
        top_p = parse_top_p(folder)
        if top_p is None:
            print(f"  [SKIP {cid}] no v84_probe.txt / top_p")
            continue
        print(f"\n## {cid}: top_p={top_p}")
        records = extract_target_motifs(ds, cid, top_p)
        write_tsv(records, folder / "target_motifs.tsv")
        s = summarize(records)
        print(f"  → {len(records)} sites written to target_motifs.tsv")
        if s:
            print(f"  summary L=14: {s}")


if __name__ == "__main__":
    main()
