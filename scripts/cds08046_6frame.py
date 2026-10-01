"""6-frame + frameshift analysis of CDS08046 events.

CDS08046 was flagged by Prodigal as a 63aa ORF with partial=10 (5'-truncated) and
start_type=Edge — the visible ORF is a fragment. This script:

  1) Extracts the raw insertion nucleotide sequences of 6 CDS08046 events.
  2) 6-frame translates each and finds the longest ORF per frame.
  3) Simulates -1 and +1 programmed frameshifts at every position (walks the
     sequence, shifts frame after each codon, sees if an extended ORF appears)
     — common in IS200/IS605 TnpB and Group II introns.
  4) Writes each event's best "extended ORF" to a fasta for blastp against
     the local Cas12f/TnpB/Y1_Tnp reference.
"""
from __future__ import annotations
import sys
from pathlib import Path

CODON = {
    "TTT":"F","TTC":"F","TTA":"L","TTG":"L",
    "CTT":"L","CTC":"L","CTA":"L","CTG":"L",
    "ATT":"I","ATC":"I","ATA":"I","ATG":"M",
    "GTT":"V","GTC":"V","GTA":"V","GTG":"V",
    "TCT":"S","TCC":"S","TCA":"S","TCG":"S",
    "CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    "ACT":"T","ACC":"T","ACA":"T","ACG":"T",
    "GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*",
    "CAT":"H","CAC":"H","CAA":"Q","CAG":"Q",
    "AAT":"N","AAC":"N","AAA":"K","AAG":"K",
    "GAT":"D","GAC":"D","GAA":"E","GAG":"E",
    "TGT":"C","TGC":"C","TGA":"*","TGG":"W",
    "CGT":"R","CGC":"R","CGA":"R","CGG":"R",
    "AGT":"S","AGC":"S","AGA":"R","AGG":"R",
    "GGT":"G","GGC":"G","GGA":"G","GGG":"G",
}

COMP = str.maketrans("ACGTN", "TGCAN")


def revcomp(s: str) -> str:
    return s.translate(COMP)[::-1]


def translate(seq: str, frame: int) -> str:
    """frame ∈ {0,1,2}"""
    seq = seq[frame:]
    out = []
    for i in range(0, len(seq) - 2, 3):
        c = seq[i:i+3]
        if len(c) < 3:
            break
        out.append(CODON.get(c.upper(), "X"))
    return "".join(out)


def longest_orfs(prot: str, min_len: int = 30) -> list[tuple[int, int, str]]:
    """Return (start_aa, end_aa, seq) for every stop-to-stop ORF ≥ min_len aa."""
    orfs = []
    cur_start = 0
    cur = []
    for i, a in enumerate(prot):
        if a == "*":
            if len(cur) >= min_len:
                orfs.append((cur_start, i, "".join(cur)))
            cur_start = i + 1
            cur = []
        else:
            cur.append(a)
    if len(cur) >= min_len:
        orfs.append((cur_start, len(prot), "".join(cur)))
    return orfs


def all_frames(dna: str, min_len: int = 30) -> list[dict]:
    """Return list of ORFs across all 6 frames, sorted by aa length desc."""
    out = []
    dna_upper = dna.upper()
    rc = revcomp(dna_upper)
    for frame in (0, 1, 2):
        prot = translate(dna_upper, frame)
        for s_aa, e_aa, seq in longest_orfs(prot, min_len):
            nt_start = frame + s_aa * 3
            nt_end = frame + e_aa * 3
            out.append({
                "strand": "+", "frame": frame,
                "aa_len": len(seq), "nt_start": nt_start, "nt_end": nt_end,
                "seq": seq,
            })
    for frame in (0, 1, 2):
        prot = translate(rc, frame)
        for s_aa, e_aa, seq in longest_orfs(prot, min_len):
            nt_start_rc = frame + s_aa * 3
            nt_end_rc = frame + e_aa * 3
            L = len(dna_upper)
            nt_start = L - nt_end_rc
            nt_end = L - nt_start_rc
            out.append({
                "strand": "-", "frame": frame,
                "aa_len": len(seq), "nt_start": nt_start, "nt_end": nt_end,
                "seq": seq,
            })
    out.sort(key=lambda x: -x["aa_len"])
    return out


def frameshift_extended(dna: str, start_nt: int, strand: str, min_len: int = 50) -> list[dict]:
    """After finding a stop, try shifting frame ±1 and continuing — programmed
    frameshift simulation. Returns extended ORFs > min_len aa that beat the
    canonical version."""
    seq = dna.upper() if strand == "+" else revcomp(dna.upper())
    L = len(seq)
    results = []
    for slip in (-1, +1):
        # walk forward from start_nt, translate codon by codon,
        # after every stop, shift by slip and continue
        i = start_nt
        aa = []
        shifts = 0
        while i + 3 <= L and shifts < 2:
            c = seq[i:i+3]
            a = CODON.get(c, "X")
            if a == "*":
                # apply slip
                i = i + 3 + slip
                shifts += 1
                if 0 <= i and i + 3 <= L:
                    continue
                else:
                    break
            else:
                aa.append(a)
                i += 3
        if len(aa) >= min_len:
            results.append({
                "strand": strand, "slip": slip,
                "start_nt": start_nt, "aa_len": len(aa),
                "seq": "".join(aa),
            })
    return results


def main():
    src = Path("/global/scratch/users/kh36969/fna_ins_discovery/database/insertions_inserts.fna")
    targets = ["senterica.E000457", "senterica.E000611", "senterica.E000612",
                    "senterica.E002196", "senterica.E002376", "senterica.E002525"]
    seqs = {}
    cur = None; buf = []
    with src.open() as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if cur and cur in targets:
                    seqs[cur] = "".join(buf)
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if cur and cur in targets:
            seqs[cur] = "".join(buf)

    out_fa = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/"
                        "fna_ins_discovery/blast/cds08046_extended_orfs.faa")
    out_fa.parent.mkdir(exist_ok=True, parents=True)
    all_out = []

    for event, dna in seqs.items():
        # Drop leading polyN runs (senterica.E000457 has 300bp of N)
        clean = dna.strip("N")
        n_dropped = len(dna) - len(clean)
        print(f"\n## {event}  raw_len={len(dna)}  clean_len={len(clean)} (dropped {n_dropped} N)")
        if len(clean) < 100:
            print("  too short after N-strip; skip")
            continue

        # 6-frame ORFs
        orfs = all_frames(clean, min_len=30)
        if not orfs:
            print("  no 6-frame ORF >= 30 aa")
            continue
        print(f"  top 6-frame ORFs (aa_len, strand, frame, nt_start, nt_end):")
        for o in orfs[:8]:
            print(f"    {o['aa_len']:3d} aa  {o['strand']}  f{o['frame']}  "
                       f"nt[{o['nt_start']}..{o['nt_end']}]")

        # Track largest
        best = orfs[0]
        # Also try programmed frameshift extension of each ORF start
        fs = []
        for o in orfs[:5]:
            fs_r = frameshift_extended(clean, o['nt_start'], o['strand'], min_len=60)
            fs.extend(fs_r)
        fs.sort(key=lambda x: -x["aa_len"])
        if fs and fs[0]["aa_len"] > best["aa_len"]:
            print(f"  frameshift-extended ORF: {fs[0]['aa_len']} aa  "
                       f"strand={fs[0]['strand']} slip={fs[0]['slip']} "
                       f"(vs canonical {best['aa_len']} aa)")
            picked = {"seq": fs[0]["seq"], "aa_len": fs[0]["aa_len"],
                          "strand": fs[0]["strand"], "kind": f"fs{fs[0]['slip']:+d}"}
        else:
            print(f"  no frameshift-extended ORF beats canonical ({best['aa_len']} aa)")
            picked = {"seq": best["seq"], "aa_len": best["aa_len"],
                          "strand": best["strand"], "kind": "canonical"}

        for k, o in enumerate(orfs[:3]):
            all_out.append((f"{event}_top{k+1}_{o['strand']}f{o['frame']}",
                              o["seq"]))
        all_out.append((f"{event}_picked_{picked['kind']}", picked["seq"]))

    with out_fa.open("w") as f:
        for name, seq in all_out:
            f.write(f">{name}  len={len(seq)}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")
    print(f"\n# wrote {len(all_out)} extended-ORF sequences to {out_fa}")


if __name__ == "__main__":
    main()
