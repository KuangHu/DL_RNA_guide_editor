"""Build the permanent Durrant IS110 gold-candidate annotation table.

Sources:
  - Durrant 2024 Supplementary Table 2, "Key bridge RNAs Used" sheet
      columns: Name, Target_Binding_Loop_Specificity, Donor_Binding_Loop_Specificity, Sequence
  - Curated cognate JSONL: IS110_gold/inference/durrant_cognate.jsonl (325 records)

For each cognate record, map the panel's transposase_id back to a bridge RNA
Name in Table 2, then use its Target_Binding_Loop_Specificity (TBL, a short
DNA sequence) to search:

  1. In the flank: find where the TBL matches → gold (flank_start, orient, L)
  2. In the noncoding region: find where the RNA-form of TBL matches → gold
     (guide_start_in_nc)

Output: one JSON record per cognate site_id with:

  site_id
  transposase_id
  bridge_rna_name
  bridge_rna_sequence               (RNA; from Supp. Table 2)
  target_binding_loop_specificity   (DNA; the target sequence in the flank)
  target_binding_loop_length        (L, typically 11 or 14)
  active_nc_index
  guide_start_in_nc / guide_end_in_nc   (guide region within ncRNA)
  target_flank_start                (best-match position in the flank)
  target_flank_orientation          ('fwd' or 'rc')
  target_flank_matches              (# of exact matches at target_flank_start; L for a perfect find)
  match_status                      ('exact' | 'partial' | 'not_found')
  provenance                        (row indices in the sources, for audit)

Deliberately does NOT modify the cognate JSONL. Gold is a separate asset.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd


def _rna_to_dna(s: str) -> str:
    return s.upper().replace("U", "T")


def _dna_to_rna(s: str) -> str:
    return s.upper().replace("T", "U")


_COMP = str.maketrans("ACGTacgt", "TGCAtgca")


def _revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def _find_best_match(query: str, target: str) -> tuple[int, int]:
    """Slide `query` across `target`, return (best_pos, matches) for the highest-match position.

    Returns (-1, 0) if `target` is shorter than `query`.
    """
    L = len(query)
    if len(target) < L:
        return -1, 0
    q_arr = np.frombuffer(query.encode(), dtype=np.uint8)
    t_arr = np.frombuffer(target.encode(), dtype=np.uint8)
    n_pos = len(target) - L + 1
    best_pos, best_matches = -1, -1
    for i in range(n_pos):
        m = int((t_arr[i:i + L] == q_arr).sum())
        if m > best_matches:
            best_matches = m
            best_pos = i
    return best_pos, best_matches


def _tnp_to_bridge_name(tnp_id: str) -> str | None:
    """Extract the bridge RNA name from a Durrant panel transposase_id.

    Example transposase_id:
        'durrant_bridge_RNA_1_7bp_RTG_D-WT_cog_bag012'
    Should return:
        'bridge_RNA_1_7bp_RTG_D-WT'   (matches Supp Table 2 Name)

    Panel names use underscores while Supp Table 2 may have hyphens. We build
    a canonical form (all underscores) for a lookup in the caller.
    """
    if "durrant_" not in tnp_id:
        return None
    # Strip 'durrant_' prefix and _cog/_shu/_paired_bagNNN and _bagNNN suffix
    body = tnp_id.split("durrant_", 1)[1]
    for token in ("_cog_", "_shu_", "_paired_"):
        if token in body:
            body = body.split(token, 1)[0]
            break
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supp2-xlsx", required=True)
    ap.add_argument("--cognate-jsonl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("[load] reading Supp Table 2 'Key bridge RNAs Used'", flush=True)
    brna_df = pd.read_excel(args.supp2_xlsx, sheet_name="Key bridge RNAs Used")
    print(f"  n_bridge_rnas={len(brna_df)}", flush=True)

    # Build lookup canonicalized to all-underscore names.
    def _canon(n: str) -> str:
        return n.replace("-", "_") if isinstance(n, str) else n

    brna_lookup: dict[str, dict] = {}
    for i, row in brna_df.iterrows():
        name = row["Name"]
        if not isinstance(name, str):
            continue
        canon = _canon(name)
        tbl = row.get("Target_Binding_Loop_Specificity")
        seq = row.get("Sequence")
        brna_lookup[canon] = {
            "raw_name":       name,
            "canon_name":     canon,
            "tbl_dna":        (str(tbl).upper() if isinstance(tbl, str) else None),
            "bridge_rna_seq": (str(seq).upper() if isinstance(seq, str) else None),
            "supp2_row":      int(i),
        }
    print(f"  bridge RNA canonical names: {len(brna_lookup)}", flush=True)

    print(f"[load] scanning cognate jsonl", flush=True)
    n_read = n_missing = n_found = 0
    n_flank_exact = n_flank_partial = n_flank_not = 0
    records_out = []
    with open(args.cognate_jsonl) as f:
        for line in f:
            r = json.loads(line)
            n_read += 1
            sid = r["site_id"]
            tnp = r["transposase_id"]
            active_nc = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if active_nc >= len(ncs):
                active_nc = 0
            nc = ncs[active_nc]
            flank = r["inputs"]["flank"]
            bridge_name = _tnp_to_bridge_name(tnp)
            if bridge_name is None:
                n_missing += 1
                continue
            canon = _canon(bridge_name)
            entry = brna_lookup.get(canon)
            if entry is None or entry["tbl_dna"] is None:
                n_missing += 1
                continue

            tbl_dna = entry["tbl_dna"]
            L = len(tbl_dna)
            # 1) find TBL in flank (fwd + rc)
            pos_fwd, m_fwd = _find_best_match(tbl_dna, flank)
            pos_rc,  m_rc  = _find_best_match(_revcomp(tbl_dna), flank)
            if m_fwd >= m_rc:
                orient = "fwd"
                flank_start = pos_fwd
                flank_matches = m_fwd
            else:
                orient = "rc"
                flank_start = pos_rc
                flank_matches = m_rc

            # 2) find TBL (RNA-form) in the ncRNA (bridge RNA in Supp2 is written with U's;
            #    the panel's noncoding_regions may already be DNA). Search the DNA form.
            nc_dna = nc.upper().replace("U", "T")
            guide_pos_nc, guide_m_nc = _find_best_match(tbl_dna, nc_dna)

            match_status = ("exact" if flank_matches == L else
                             ("partial" if flank_matches >= max(1, L - 2) else "not_found"))
            if match_status == "exact":
                n_flank_exact += 1
            elif match_status == "partial":
                n_flank_partial += 1
            else:
                n_flank_not += 1
            n_found += 1

            records_out.append({
                "site_id":                       sid,
                "transposase_id":                tnp,
                "bridge_rna_name":               entry["raw_name"],
                "bridge_rna_canonical":          canon,
                "bridge_rna_sequence":           entry["bridge_rna_seq"],
                "target_binding_loop_specificity": tbl_dna,
                "target_binding_loop_length":    L,
                "active_nc_index":               active_nc,
                "guide_start_in_nc":             int(guide_pos_nc),
                "guide_end_in_nc":               int(guide_pos_nc + L) if guide_pos_nc >= 0 else -1,
                "guide_matches_in_nc":           int(guide_m_nc),
                "target_flank_start":            int(flank_start),
                "target_flank_orientation":      orient,
                "target_flank_matches":          int(flank_matches),
                "target_binding_loop_L":         L,
                "flank_length":                  len(flank),
                "nc_length":                     len(nc),
                "match_status":                  match_status,
                "provenance": {
                    "supp2_row":     entry["supp2_row"],
                    "supp2_source":  args.supp2_xlsx,
                    "cognate_jsonl": args.cognate_jsonl,
                },
            })

    print(f"[done] read={n_read}  missing_mapping={n_missing}  annotated={n_found}", flush=True)
    print(f"  flank match status: exact={n_flank_exact}  partial={n_flank_partial}  not_found={n_flank_not}",
          flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in records_out:
            f.write(json.dumps(r) + "\n")
    print(f"[out] {args.out}  ({len(records_out)} rows)", flush=True)


if __name__ == "__main__":
    main()
