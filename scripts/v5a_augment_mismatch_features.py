"""V5A-3a0 data-prep: augment mining records with per-candidate mismatch features.

Streams positives_v42.jsonl and hard_decoys_full.jsonl in parallel (verified
same order), and for each c* + each of the 12 decoys computes the per-position
match/mismatch pattern by reconstructing the alignment from (orient, L,
nc_start, flank_start) against the record's nc + flank.

Adds to c* and each decoy dict:
  m                   (matches, redundant with existing 'matches')
  mm_count            = L - matches
  mm_frac_5p          fraction of mismatches in first L//2 positions
  mm_frac_3p          fraction of mismatches in last L//2 positions
  mm_at_pos_0         1 if first guide position is a mismatch
  mm_at_pos_last      1 if last guide position is a mismatch

Only inference-visible features. No absolute RNA coordinate. No planted labels.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

_COMP = str.maketrans("ACGTacgt", "TGCAtgca")


def _revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def _match_pattern(nc: str, flank: str, orient: str, L: int,
                    nc_start: int, flank_start: int) -> list[int]:
    """Return list of 0/1 per guide position (1 = match). Length L."""
    if nc_start < 0 or nc_start + L > len(nc): return []
    if flank_start < 0 or flank_start + L > len(flank): return []
    guide = nc[nc_start:nc_start + L].upper().replace("U", "T")
    target = flank[flank_start:flank_start + L].upper()
    if orient == "rc":
        target = _revcomp(target)
    if len(guide) != L or len(target) != L: return []
    return [1 if guide[i] == target[i] else 0 for i in range(L)]


def _summarize(pattern: list[int]) -> dict:
    L = len(pattern)
    if L == 0:
        return {"mm_count": 0, "mm_frac_5p": 0.0, "mm_frac_3p": 0.0,
                 "mm_at_pos_0": 0, "mm_at_pos_last": 0}
    mm = [1 - x for x in pattern]
    mm_count = sum(mm)
    half = L // 2
    n_5p = sum(mm[:half]) if half > 0 else 0
    n_3p = sum(mm[L - half:]) if half > 0 else 0
    return {
        "mm_count":       int(mm_count),
        "mm_frac_5p":     float(n_5p) / max(1, half),
        "mm_frac_3p":     float(n_3p) / max(1, half),
        "mm_at_pos_0":    int(mm[0]),
        "mm_at_pos_last": int(mm[-1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-jsonl", required=True)
    ap.add_argument("--mining-jsonl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n_ok = n_dropped = 0
    with open(args.pos_jsonl) as fp, open(args.mining_jsonl) as fm, open(args.out, "w") as fo:
        for pl, ml in zip(fp, fm):
            pr = json.loads(pl); mr = json.loads(ml)
            assert pr["site_id"] == mr["site_id"], (pr["site_id"], mr["site_id"])
            active_nc = pr["labels"].get("active_noncoding_index", 0) or 0
            ncs = pr["inputs"]["noncoding_regions"]
            if active_nc >= len(ncs): active_nc = 0
            nc = ncs[active_nc]
            flank = pr["inputs"]["flank"]

            cs = mr["cstar"]
            if cs["slot"] >= 0:
                pat = _match_pattern(nc, flank, cs["orient"], int(cs["L"]),
                                       int(cs["nc_start"]), int(cs["flank_start"]))
                cs.update(_summarize(pat))
            for d in mr["decoys"]:
                pat = _match_pattern(nc, flank, d["orient"], int(d["L"]),
                                       int(d["nc_start"]), int(d["flank_start"]))
                d.update(_summarize(pat))
            fo.write(json.dumps(mr) + "\n")
            n_ok += 1
            if n_ok % 25000 == 0:
                print(f"  progress: {n_ok}", flush=True)

    print(f"[done] wrote {n_ok} records to {args.out}", flush=True)


if __name__ == "__main__":
    main()
