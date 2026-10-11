"""Boundary-stability analysis — step 1 of the reviewer-set next-step order.

For each supporting site of a CDS03597 cluster, compute:

  (A) In the OCCUPIED allele's coordinate frame, where do the two 14-mer
      cores sit? core_a = the per-site guide nc[nc_pos : nc_pos + L];
      core_b = RC(core_a) occurrence that was matched by the Phase-C hit
      at flank_pos. Report their absolute positions in the occupied
      allele.

  (B) Sweep the insertion boundary across the full ambiguity window
      [junction_L_abs - j_amb .. junction_L_abs + j_amb] (in units of
      the occupied allele). At each candidate boundary pick, record
      whether core_a and core_b fall INSIDE the "insert" interval
      [b .. b + inserted_len]. Report the fraction of best boundaries
      (and the extreme boundaries) under which the assignment flips.

  (C) In the OCCUPIED allele frame, measure the contiguous
      Watson-Crick-reverse-complement match length between the two
      cores (extend from the core positions in both directions; stop
      at first mismatch). This is the "same-direction repeat length
      in occupied" that the reviewer asked for.

Writes `occupied_core_stability.tsv` at the parent dir with one row per
site, carrying all three measurements.

Usage:
  python scripts/boundary_stability.py \
    --jsonl /global/scratch/.../adapted.jsonl \
    --evidence-dir evidence/CDS03597/nc1_L14_fp45_v2 \
    --bag CDS03597 --nc-pos 1 --L 14 --flank-pos 45
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v5a_framework.pipeline import run_bag, PipelineConfig
from scripts.v5a_framework.multi_hit_retention import SPACER_N
from preprocess.alignment import revcomp


def read_fna(path: Path) -> str:
    seq = []
    with path.open() as f:
        for line in f:
            if line.startswith(">"): continue
            seq.append(line.rstrip("\n"))
    return "".join(seq).upper()


def concat_nc(nc_list):
    out = nc_list[0]
    for extra in nc_list[1:]:
        out = out + "N" * SPACER_N + extra
    return out


def find_core_in_occupied(core: str, occupied: str) -> list[int]:
    """Return all 1-based positions where core occurs in occupied (forward)."""
    hits = []
    i = 0
    while True:
        j = occupied.find(core, i)
        if j < 0: break
        hits.append(j)
        i = j + 1
    return hits


def contig_match_direct(a_seq: str, a_pos: int, b_seq: str, b_pos: int,
                          core_len: int) -> dict:
    """DIRECT (same-strand) repeat extension. Core: a_seq[a_pos:a_pos+core_len]
    == b_seq[b_pos:b_pos+core_len]. Extend to both sides keeping both pointers
    moving in the same direction; stop at first mismatch or edge."""
    left_ext = 0
    while True:
        ia = a_pos - 1 - left_ext
        ib = b_pos - 1 - left_ext
        if ia < 0 or ib < 0: break
        if a_seq[ia] != b_seq[ib]: break
        left_ext += 1
    right_ext = 0
    while True:
        ia = a_pos + core_len + right_ext
        ib = b_pos + core_len + right_ext
        if ia >= len(a_seq) or ib >= len(b_seq): break
        if a_seq[ia] != b_seq[ib]: break
        right_ext += 1
    return {"core_len": core_len, "left_ext": left_ext, "right_ext": right_ext,
             "total": core_len + left_ext + right_ext}


def contig_match_inverted(a_seq: str, a_pos: int, b_seq: str, b_pos: int,
                           core_len: int) -> dict:
    """INVERTED (opposite-strand) repeat extension. Anchor: a[a_pos:a_pos+L]
    == revcomp(b[b_pos:b_pos+L]). Extend outward (apart) in both directions."""
    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    left_ext = 0
    while True:
        ia = a_pos - 1 - left_ext
        ib = b_pos + core_len + left_ext
        if ia < 0 or ib >= len(b_seq): break
        if a_seq[ia] != comp.get(b_seq[ib], "N"): break
        left_ext += 1
    right_ext = 0
    while True:
        ia = a_pos + core_len + right_ext
        ib = b_pos - 1 - right_ext
        if ia >= len(a_seq) or ib < 0: break
        if a_seq[ia] != comp.get(b_seq[ib], "N"): break
        right_ext += 1
    return {"core_len": core_len, "left_ext": left_ext, "right_ext": right_ext,
             "total": core_len + left_ext + right_ext}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--evidence-dir", type=Path, required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--nc-pos", type=int, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--flank-pos", type=int, required=True)
    ap.add_argument("--max-offset", type=int, default=5)
    args = ap.parse_args()

    # Load the bag + replay to get the supporting-site set and per-site hits
    sites = []
    with args.jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("transposase_id") == args.bag:
                sites.append(r)
    cfg = PipelineConfig()
    result = run_bag(args.bag, sites, cfg)
    cl, bg = None, None
    for ccl, cbg in result.passing + result.failing:
        if ccl.nc_pos == args.nc_pos and ccl.L == args.L and \
                abs(ccl.flank_pos_median - args.flank_pos) <= args.max_offset:
            cl, bg = ccl, cbg; break
    if cl is None:
        raise SystemExit("cluster not found")
    support_idx = {ss.site_idx: ss for ss in cl.supporting_sites}

    out = args.evidence_dir / "occupied_core_stability.tsv"
    cols = ["site_idx", "locus_id",
            "repeat_type",  # "direct" / "inverted" / "unknown"
            "guide_core", "target_core_raw",
            "core_a_pos_in_occupied", "core_b_pos_in_occupied",
            "insert_start_in_long",  # the pipeline's chosen boundary (0-based)
            "insert_end_in_long",
            "boundary_min", "boundary_max",  # ambiguity window edges in occupied frame
            "core_a_inside_at_chosen", "core_b_inside_at_chosen",
            "core_a_stable_across_window",
            "core_b_stable_across_window",
            "occupied_repeat_total_bp",   # core + left_ext + right_ext
            "occupied_repeat_core_bp",
            "occupied_repeat_left_ext_bp",
            "occupied_repeat_right_ext_bp",
            "notes",
            ]
    with out.open("w") as f:
        f.write("\t".join(cols) + "\n")

        for sidx in sorted(support_idx):
            site = sites[sidx]
            nc = concat_nc(site["inputs"]["noncoding_regions"])
            flank = site["inputs"]["flank"]
            ss = support_idx[sidx]
            hit = ss.chosen_hit

            # Per-site cores (as the pipeline computed them)
            # guide_core: nc[nc_pos : nc_pos + L] — forward-strand on the nc.
            # target_core_raw: flank[flank_pos : flank_pos + L] — forward-strand
            #                   on the flank as extracted.
            # If orient=='rc', revcomp(target_core_raw) == guide_core by hit
            # definition. In the OCCUPIED allele's forward strand both the
            # guide (inside insert) and the target (on host side) sit at
            # different genomic positions; the match is between forward and
            # reverse-complement.
            guide_core = nc[args.nc_pos : args.nc_pos + args.L].upper()
            target_core_raw = flank[hit.flank_pos : hit.flank_pos + args.L].upper()

            # Load occupied allele + loci evidence
            site_dir = args.evidence_dir / f"site_{sidx}"
            occupied = read_fna(site_dir / "occupied.fna")
            loci_fields = {}
            with (site_dir / "loci_evidence.tsv").open() as lf:
                lf.readline()
                for line in lf:
                    k, v = line.rstrip("\n").split("\t", 1)
                    loci_fields[k] = v

            # Chosen insert boundaries in occupied-allele 0-based coords
            start_val = loci_fields.get("insert_start_in_long_bp", "")
            if "(" in start_val:
                b0 = int(start_val.split()[0])  # fallback form "139 (fallback: ...)"
            else:
                b0 = int(start_val)
            ins_len = int(loci_fields["inserted_len"])
            b1 = b0 + ins_len

            # Ambiguity window: full width = junction_ambiguity_bp around the
            # chosen junction. junction_L_abs/R_abs are genome coords; the
            # occupied-frame center = b0. Width = j_amb_bp (symmetric).
            j_amb = int(loci_fields["junction_ambiguity_bp"])
            bmin = max(0, b0 - j_amb // 2)
            bmax = min(len(occupied) - ins_len, b0 + j_amb - (j_amb // 2))

            # Locate the two cores in occupied.
            # Try FOUR candidate representations in occupied forward strand:
            #   guide_core, revcomp(guide_core),
            #   target_core_raw, revcomp(target_core_raw).
            # For orient="rc", revcomp(target_core_raw) == guide_core (duplicate).
            # The "two cores" are two distinct positions in occupied that
            # correspond to the same 14-mer the pairing displayed.
            hits_guide = find_core_in_occupied(guide_core, occupied)
            hits_guide_rc = find_core_in_occupied(revcomp(guide_core).upper(),
                                                   occupied)
            hits_target = find_core_in_occupied(target_core_raw, occupied)
            hits_target_rc = find_core_in_occupied(revcomp(target_core_raw).upper(),
                                                    occupied)
            # The actual present-on-forward-strand hits:
            fwd_hits = sorted(set(hits_guide) | set(hits_target_rc))

            # Determine repeat type:
            # - DIRECT: 2 forward-strand copies of the same 14-mer (both =
            #   guide_core, or both = target_core_raw).
            # - INVERTED: 1 copy of X on fwd, 1 copy of revcomp(X) on fwd.
            # For orient="rc" with flank extracted on same strand as insert:
            # if both forward positions have the SAME 14-mer → direct repeat.
            # If one is guide_core and the other is revcomp(guide_core) →
            # inverted repeat.
            n_as_guide = len(hits_guide)
            n_as_guide_rc = len(hits_guide_rc)
            n_as_target = len(hits_target)
            n_as_target_rc = len(hits_target_rc)

            exp_a = b0 + args.nc_pos
            if n_as_guide >= 2:
                # DIRECT REPEAT on forward strand
                repeat_type = "direct"
                core_a_pos = min(hits_guide, key=lambda p: abs(p - exp_a))
                other = [p for p in hits_guide if p != core_a_pos]
                core_b_pos = min(other, key=lambda p: abs(p - b0)) if other else -1
                core_a_src = f"guide_core fwd (direct repeat) exp={exp_a}"
                core_b_src = f"guide_core fwd 2nd occurrence"
            elif n_as_guide == 1 and n_as_guide_rc >= 1:
                # INVERTED REPEAT on forward strand
                repeat_type = "inverted"
                core_a_pos = hits_guide[0]
                core_b_pos = min(hits_guide_rc, key=lambda p: abs(p - b0))
                core_a_src = f"guide_core fwd (inverted repeat anchor)"
                core_b_src = f"revcomp(guide_core) fwd"
            else:
                repeat_type = "unknown"
                core_a_pos = hits_guide[0] if hits_guide else -1
                core_b_pos = -1
                core_a_src = f"n_guide_fwd={n_as_guide} n_guide_rc_fwd={n_as_guide_rc}"
                core_b_src = f"n_target_fwd={n_as_target} n_target_rc_fwd={n_as_target_rc}"

            def inside(pos: int, b: int) -> bool:
                return b <= pos < b + ins_len

            def stable_assignment(pos: int) -> bool:
                if pos < 0: return False
                inside_at_min = inside(pos, bmin)
                inside_at_max = inside(pos, bmax)
                inside_at_chosen = inside(pos, b0)
                return inside_at_min == inside_at_max == inside_at_chosen

            # Measure contiguous match in occupied allele according to the
            # detected repeat type:
            if core_a_pos >= 0 and core_b_pos >= 0:
                if repeat_type == "direct":
                    m = contig_match_direct(occupied, core_a_pos,
                                              occupied, core_b_pos, args.L)
                elif repeat_type == "inverted":
                    m = contig_match_inverted(occupied, core_a_pos,
                                                occupied, core_b_pos, args.L)
                else:
                    m = {"core_len": args.L, "left_ext": 0, "right_ext": 0,
                          "total": args.L}
            else:
                m = {"core_len": args.L, "left_ext": -1, "right_ext": -1,
                      "total": -1}

            row = [str(sidx),
                   loci_fields.get("locus_id", ""),
                   repeat_type,
                   guide_core, target_core_raw,
                   str(core_a_pos), str(core_b_pos),
                   str(b0), str(b1),
                   str(bmin), str(bmax),
                   str(inside(core_a_pos, b0)) if core_a_pos >= 0 else "n/a",
                   str(inside(core_b_pos, b0)) if core_b_pos >= 0 else "n/a",
                   str(stable_assignment(core_a_pos)),
                   str(stable_assignment(core_b_pos)),
                   str(m["total"]),
                   str(m["core_len"]), str(m["left_ext"]), str(m["right_ext"]),
                   f"n_guide_fwd={n_as_guide}, n_guide_rc_fwd={n_as_guide_rc}, "
                   f"n_target_fwd={n_as_target}, n_target_rc_fwd={n_as_target_rc}; "
                   f"{core_a_src}; {core_b_src}"]
            f.write("\t".join(row) + "\n")

            print(f"[site {sidx}] repeat_type={repeat_type}  "
                  f"guide={guide_core}  target_raw={target_core_raw}  "
                  f"orient={hit.orient}")
            print(f"  fwd-strand occurrence counts: guide={n_as_guide}  "
                  f"guide_rc={n_as_guide_rc}  target={n_as_target}  "
                  f"target_rc={n_as_target_rc}")
            print(f"  core_a_pos={core_a_pos}  inside_at_chosen={inside(core_a_pos,b0) if core_a_pos>=0 else 'n/a'}  "
                  f"stable_across_window={stable_assignment(core_a_pos)}")
            print(f"  core_b_pos={core_b_pos}  inside_at_chosen={inside(core_b_pos,b0) if core_b_pos>=0 else 'n/a'}  "
                  f"stable_across_window={stable_assignment(core_b_pos)}")
            print(f"  insert interval in occupied: [{b0}..{b1})  "
                  f"ambiguity window for boundary: [{bmin}..{bmax})  "
                  f"(width={bmax-bmin}bp = j_amb {loci_fields['junction_ambiguity_bp']})")
            print(f"  occupied-frame {repeat_type}-repeat extent:  "
                  f"L={m['core_len']}  left={m['left_ext']}  right={m['right_ext']}  "
                  f"TOTAL={m['total']} bp")
            print()

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
