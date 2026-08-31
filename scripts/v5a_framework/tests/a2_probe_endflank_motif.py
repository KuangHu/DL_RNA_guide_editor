"""A2 — identify the fixed-position artifact at flank positions [101, 110).

The 218x FP concentration under shift_matched=9 exclusion says the FP
comes from a fixed 9-nt slice at the far end of the (extracted, possibly
RC'd) flank. This script dumps that slice for the 16 sticky Tnps and
looks for a shared motif or a substring pattern that would name the
artifact.

Also dumps the nc-side sequences that are actually being matched by
those flank positions, to see whether the FP mechanism is:
  - a repeated pipeline-boundary motif matching a common ncRNA feature
  - or something else
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocess.alignment import dot_plot, windowed_matches
from v5a_framework.e_match_table import load_e


IS10R = "/global/scratch/users/kh36969/DL_novel_guide_editor/real_data/formatted/real_IS10-R_sites.jsonl"
E_DIAGONAL = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5a_framework_cache/e_durrant_diagonal"

L_FIX = 11
M_THRESH = 8
S_THRESH = 5
END_START, END_END = 101, 110  # the 9-nt slice of concern


def load_paired_downstream(path):
    ins = defaultdict(dict)
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            m = d.get("generator_metadata", {})
            key = (d["transposase_id"], m.get("insertion_start"), m.get("sample_id"))
            side = m.get("flank_side")
            rc = m.get("reverse_complemented")
            fl = d.get("inputs", {}).get("flank")
            if side in ("upstream", "downstream") and fl:
                ins[key][side] = (fl, rc)
    tnp_dn = defaultdict(list)
    for (tnp, ins_start, sid), p in ins.items():
        if "upstream" in p and "downstream" in p:
            fl_dn, rc = p["downstream"]
            tnp_dn[tnp].append((fl_dn, rc))
    return {t: v for t, v in tnp_dn.items() if len(v) >= 5}


def identity(a, b):
    n = min(len(a), len(b))
    return sum(1 for x, y in zip(a, b) if x == y) / n if n else 0.0


def cluster_single_linkage(flanks_and_rc, thresh):
    n = len(flanks_and_rc)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry: parent[rx] = ry
    for i in range(n):
        for j in range(i + 1, n):
            if identity(flanks_and_rc[i][0], flanks_and_rc[j][0]) >= thresh:
                union(i, j)
    clus = defaultdict(list)
    for i in range(n):
        clus[find(i)].append(i)
    return list(clus.values())


def get_a1p_flanks():
    """Return the same 27 Tnps × 5 dedup-representative flanks A1' used."""
    tnp_dn = load_paired_downstream(IS10R)
    sorted_ids = sorted(tnp_dn.keys())
    rng = np.random.default_rng(0)
    picked = list(rng.choice(sorted_ids, size=min(30, len(sorted_ids)), replace=False))
    out = {}
    for tnp in picked:
        pool = tnp_dn[tnp]
        clus = cluster_single_linkage(pool, 0.90)
        clus_sorted = sorted(clus, key=len, reverse=True)
        reps = [pool[c[0]] for c in clus_sorted]
        if len(reps) < 5:
            continue
        out[tnp] = reps[:5]  # [(flank_str, rc_flag), ...]
    return out


def main() -> int:
    print("[A2] loading A1' Tnp/flank selection", flush=True)
    tnp_flanks = get_a1p_flanks()
    emt = load_e(E_DIAGONAL)
    print(f"[A2] {len(tnp_flanks)} Tnps, {len(emt.nc_source_tnp_ids)} Durrant ncs", flush=True)

    # 1) Dump the [101, 110) 9-mer of every flank across every Tnp; count frequency.
    print()
    print(f"=== 9-mer at flank positions [{END_START}, {END_END}) across 27 x 5 = 135 flanks ===")
    kmer_freq = Counter()
    per_tnp_kmers = defaultdict(list)
    per_tnp_rc = defaultdict(list)
    for tnp, flanks in tnp_flanks.items():
        for fl_str, rc in flanks:
            slc = fl_str[END_START:END_END]
            kmer_freq[slc] += 1
            per_tnp_kmers[tnp].append(slc)
            per_tnp_rc[tnp].append(rc)
    print(f"  distinct 9-mers observed: {len(kmer_freq)}")
    print(f"  top 15 most frequent 9-mers:")
    for kmer, count in kmer_freq.most_common(15):
        print(f"    {kmer}: {count}")
    print(f"  frequency at count=1: {sum(1 for c in kmer_freq.values() if c == 1)}"
          f"    (expected if random: ~135)")

    # 2) Now identify which Tnps are "sticky" — need to detect FP inline.
    #    Rerun the S_all detection to know which (Tnp, nc) pairs fired.
    print()
    print(f"=== Redoing S_all detection to identify sticky Tnps + hit positions ===")
    fires_by_tnp = Counter()
    hit_positions_by_tnp = defaultdict(list)   # list of (nc_id, nc_pos)
    for tnp in tnp_flanks:
        flanks5 = [x[0] for x in tnp_flanks[tnp]]
        for nc_tnp in emt.nc_source_tnp_ids:
            nc = emt.nc_source_ncs[nc_tnp]
            nc_len_pos = len(nc) - L_FIX + 1
            fired = False
            for orient_name, orient_dot_key in [("fwd", 0), ("rc", 1)]:
                m_5 = np.zeros((5, nc_len_pos), dtype=np.int32)
                for si, fl in enumerate(flanks5):
                    dots = dot_plot(nc, fl)
                    win = windowed_matches(dots[orient_dot_key], L_FIX)
                    if win.size:
                        m = win.max(axis=1)
                        m_5[si, :len(m)] = m
                # detect
                hits_lists = [set(int(p) for p in np.where(m_5[s] >= M_THRESH)[0])
                                for s in range(5)]
                # Manual local-max detection (avoid importing to keep clean)
                S = np.zeros(nc_len_pos)
                for h in hits_lists:
                    for p in h:
                        if 0 <= p < nc_len_pos:
                            S[p] += 1
                peak_positions = []
                for i in range(nc_len_pos):
                    if S[i] < S_THRESH: continue
                    is_max = True
                    for j in range(max(0, i-5), min(nc_len_pos, i+6)):
                        if j != i and S[j] > S[i]:
                            is_max = False; break
                    if is_max:
                        peak_positions.append(i)
                if peak_positions:
                    fired = True
                    for p in peak_positions:
                        hit_positions_by_tnp[tnp].append((nc_tnp, orient_name, p))
            if fired:
                fires_by_tnp[tnp] += 1
    print(f"  total S_all fires: {sum(fires_by_tnp.values())}")
    print(f"  per-Tnp distribution: {dict(fires_by_tnp)}")

    # 3) For the sticky Tnps: dump their [101, 110) 9-mers + the nc-side seqs at hit positions
    print()
    sticky = [t for t, c in fires_by_tnp.most_common() if c > 10]
    print(f"=== Sticky Tnps (>10 fires) : {len(sticky)} ===")
    for tnp in sticky:
        n_fires = fires_by_tnp[tnp]
        kmers = per_tnp_kmers[tnp]
        rc = per_tnp_rc[tnp]
        print(f"\n  {tnp}  fires={n_fires}/65  rc_flags={rc}")
        print(f"    5 flanks' [{END_START},{END_END}) 9-mers:")
        for i, km in enumerate(kmers):
            print(f"      site {i}: {km}   (rc={rc[i]})")
        # composition of that slice, aggregated
        pos_freqs = np.zeros((9, 4))
        base_idx = {'A':0,'C':1,'G':2,'T':3}
        for km in kmers:
            for pos, b in enumerate(km):
                if b in base_idx:
                    pos_freqs[pos, base_idx[b]] += 1
        pos_freqs /= max(1, len(kmers))
        modal = pos_freqs.max(axis=1)
        modal_bases = [ "ACGT"[i] for i in pos_freqs.argmax(axis=1) ]
        print(f"    modal 9-mer  : {''.join(modal_bases)}  modal_freq per pos: {modal.round(2).tolist()}")

        # Also: what nc positions was this Tnp hitting? Distribution of nc positions
        hits = hit_positions_by_tnp[tnp]
        pos_counts = Counter(p for (_, _, p) in hits)
        top3 = pos_counts.most_common(3)
        print(f"    top nc positions hit: {top3}")
        # For the top position, show the nc sequence there (from one nc)
        if top3:
            top_pos = top3[0][0]
            for nc_id, orient, p in hits[:3]:
                if p == top_pos:
                    nc = emt.nc_source_ncs[nc_id]
                    print(f"      nc {nc_id[:35]}... orient={orient} pos={p}  nc[p:p+11] = {nc[p:p+11]}")
                    break

    # 4) Check if the modal 9-mer across sticky Tnps is shared
    print()
    print("=== Shared boundary motif across all 135 flanks ===")
    all_kmers = list(kmer_freq.keys())
    top_kmer = kmer_freq.most_common(1)[0]
    print(f"  Most-frequent 9-mer across ALL 135 slices: {top_kmer[0]!r} × {top_kmer[1]}")
    print(f"  Top 5 by count: {kmer_freq.most_common(5)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
