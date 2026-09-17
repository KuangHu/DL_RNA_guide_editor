"""Fetch 50 diverse bacterial reference genomes for v7 refactor's real-flank pool.

Selection covers major bacterial phyla (Proteobacteria, Firmicutes,
Actinobacteria, Bacteroidetes, Cyanobacteria, Spirochaetes, Chlamydiae,
Thermotogae, Aquificae, Deinococcus-Thermus, Tenericutes, etc.) and a
GC-content range from ~30% (Mycoplasma/Rickettsia) to ~70% (Streptomyces/
Deinococcus). Diversity is the design goal — flank draws will span the
whole distribution.

Runs once (idempotent — skips already-cached). Total: ~50 × ~4Mb = ~200Mb
in `/global/scratch/.../v7_refactor/genome_pool/`.

Usage:
    python -m scripts.fetch_v7_genome_pool
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

from Bio import Entrez, SeqIO

Entrez.email = "kh36969@berkeley.edu"

OUT_DIR = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/v7_refactor/genome_pool")

# (organism_label, refseq_accession)
GENOMES: list[tuple[str, str]] = [
    # Gammaproteobacteria
    ("Escherichia_coli_K12_MG1655",           "NC_000913.3"),
    ("Salmonella_enterica_LT2",               "NC_003197.2"),
    ("Yersinia_pestis_CO92",                  "NC_003143.1"),
    ("Klebsiella_pneumoniae_HS11286",         "NC_016845.1"),
    ("Pseudomonas_aeruginosa_PAO1",           "NC_002516.2"),
    ("Vibrio_cholerae_O1_N16961_chr1",        "NC_002505.1"),
    ("Xanthomonas_campestris_ATCC33913",      "NC_003902.1"),
    ("Legionella_pneumophila_Philadelphia1",  "NC_002942.5"),
    ("Francisella_tularensis_SchuS4",         "NC_006570.2"),
    ("Coxiella_burnetii_RSA493",              "NC_002971.4"),
    # Betaproteobacteria
    ("Neisseria_meningitidis_MC58",           "NC_003112.2"),
    ("Bordetella_pertussis_Tohama_I",         "NC_002929.2"),
    ("Ralstonia_solanacearum_GMI1000",        "NC_003295.1"),
    ("Nitrosomonas_europaea_ATCC19718",       "NC_004757.1"),
    ("Burkholderia_pseudomallei_K96243_chr1", "NC_006350.1"),
    # Alphaproteobacteria
    ("Rickettsia_prowazekii_MadridE",         "NC_000963.1"),
    ("Caulobacter_crescentus_CB15",           "NC_002696.2"),
    ("Sinorhizobium_meliloti_1021",           "NC_003047.1"),
    ("Brucella_melitensis_16M_chr1",          "NC_003317.1"),
    ("Zymomonas_mobilis_ZM4",                 "NC_006526.2"),
    # Epsilonproteobacteria
    ("Helicobacter_pylori_26695",             "NC_000915.1"),
    ("Campylobacter_jejuni_NCTC11168",        "NC_002163.1"),
    # Deltaproteobacteria
    ("Geobacter_sulfurreducens_PCA",          "NC_002939.5"),
    # Firmicutes (Bacilli)
    ("Bacillus_subtilis_168",                 "NC_000964.3"),
    ("Bacillus_cereus_ATCC14579",             "NC_004722.1"),
    ("Staphylococcus_aureus_NCTC8325",        "NC_007795.1"),
    ("Streptococcus_pneumoniae_TIGR4",        "NC_003028.3"),
    ("Enterococcus_faecalis_V583",            "NC_004668.1"),
    ("Listeria_monocytogenes_EGDe",           "NC_003210.1"),
    ("Lactobacillus_plantarum_WCFS1",         "NC_004567.2"),
    # Firmicutes (Clostridia)
    ("Clostridioides_difficile_630",          "NC_009089.1"),
    ("Clostridium_botulinum_A_ATCC3502",      "NC_009495.1"),
    # Tenericutes (Mycoplasma)
    ("Mycoplasma_pneumoniae_M129",            "NC_000912.1"),
    # Actinobacteria
    ("Mycobacterium_tuberculosis_H37Rv",      "NC_000962.3"),
    ("Corynebacterium_glutamicum_ATCC13032",  "NC_003450.3"),
    ("Streptomyces_coelicolor_A3_2",          "NC_003888.3"),
    ("Bifidobacterium_longum_NCC2705",        "NC_004307.2"),
    # Bacteroidetes
    ("Bacteroides_thetaiotaomicron_VPI5482",  "NC_004663.1"),
    ("Porphyromonas_gingivalis_W83",          "NC_002950.2"),
    # Fusobacteria
    ("Fusobacterium_nucleatum_ATCC25586",     "NC_003454.1"),
    # Spirochaetes
    ("Borrelia_burgdorferi_B31",              "NC_001318.1"),
    ("Treponema_pallidum_Nichols",            "NC_000919.1"),
    # Chlamydiae
    ("Chlamydia_trachomatis_D_UW3_CX",        "NC_000117.1"),
    # Cyanobacteria
    ("Synechocystis_sp_PCC6803",              "NC_000911.1"),
    ("Nostoc_sp_PCC7120",                     "NC_003272.1"),
    ("Prochlorococcus_marinus_MIT9313",       "NC_005071.1"),
    # Deinococcus-Thermus
    ("Deinococcus_radiodurans_R1_chr1",       "NC_001263.1"),
    ("Thermus_thermophilus_HB8",              "NC_006461.1"),
    # Aquificae
    ("Aquifex_aeolicus_VF5",                  "NC_000918.1"),
    # Verrucomicrobia
    ("Akkermansia_muciniphila_ATCC_BAA835",   "NC_010655.1"),
]
assert len(GENOMES) == 50, f"expected 50 genomes, got {len(GENOMES)}"


def fetch_one(label: str, acc: str) -> Path:
    out = OUT_DIR / f"{label}__{acc}.fna"
    if out.exists() and out.stat().st_size > 100_000:
        return out
    print(f"[fetch] {label}  ({acc}) ...", flush=True)
    handle = Entrez.efetch(db="nuccore", id=acc, rettype="fasta", retmode="text")
    rec = SeqIO.read(handle, "fasta")
    handle.close()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        SeqIO.write([rec], f, "fasta")
    print(f"[fetch]   wrote {len(rec.seq):,} bp → {out.name}", flush=True)
    return out


def main():
    print(f"[main] fetching {len(GENOMES)} bacterial genomes → {OUT_DIR}")
    fetched = []
    for i, (label, acc) in enumerate(GENOMES):
        try:
            out = fetch_one(label, acc)
            fetched.append((label, acc, out))
            # NCBI courtesy rate limit
            time.sleep(0.4)
        except Exception as e:
            print(f"[fetch] FAIL {label} ({acc}): {type(e).__name__}: {e}", flush=True)

    print(f"\n[main] fetched {len(fetched)}/{len(GENOMES)} genomes")
    # Report length + GC per genome
    print(f"\n{'label':<40s}  {'accession':<12s}  {'length':>10s}  {'gc%':>6s}")
    for label, acc, out in fetched:
        rec = next(SeqIO.parse(out, "fasta"))
        seq = str(rec.seq).upper()
        n = len(seq)
        gc = (seq.count("G") + seq.count("C")) / max(1, n) * 100
        print(f"  {label:<40s}  {acc:<12s}  {n:>10,d}  {gc:>5.1f}%")


if __name__ == "__main__":
    main()
