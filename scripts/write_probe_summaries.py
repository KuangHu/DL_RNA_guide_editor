"""Write probe_summary.md for each of the 5 deeply-probed candidates,
and append a Verdict section to their README.md.

Reads v84_probe.txt in each folder, extracts the key metrics, and produces
a compact summary matching what the user's been reading in chat.
"""
from __future__ import annotations
import re
from pathlib import Path

CAND_ROOT = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/"
                       "fna_ins_discovery/candidate")

# Per-CDS verdict text
VERDICTS = {
    "CDS01040": {
        "family": "rve_3 + TnsB_C + Tnp_IS2_N + HTH_Tnp_1 (IS3-family transposon)",
        "known_rna_guided": False,
        "rank_in_batch": 48,
        "verdict": ("Bicistronic/tricistronic IS3-family transposon architecture "
                          "(3 ORFs per event). rve/TnsB catalytic core is NOT known to "
                          "be RNA-guided. Cross-site coherence signal (v84 score 4.67, "
                          "peak_ratio 2.67, all 8 sites m_max ≥ 0.57) is compatible with "
                          "either genuine RNA guide OR DNA-DNA target immunity. "
                          "**Novel-RNA-guided rve-family candidate — needs discriminating "
                          "experiment to confirm.**"),
    },
    "CDS02752": {
        "family": "(no Pfam hit at E ≤ 1e-3)",
        "known_rna_guided": False,
        "rank_in_batch": 66,
        "verdict": ("**Purest novelty candidate.** Zero Pfam annotation across all "
                          "ORFs. 112aa dominant protein is *identical* across all 6 "
                          "efaecalis events and closely related in efaecium, "
                          "lmonocytogenes, and S. aureus — 4-species Firmicute "
                          "conservation. Model signal is REAL but SOFT (peak_ratio "
                          "1.23, m_min 0.36, K=6 sites). Recommend enlarging bag or "
                          "confirming with additional Firmicute genomes before "
                          "wet-lab prioritization."),
    },
    "CDS05957": {
        "family": "rve + rve_3 (retroviral integrase family)",
        "known_rna_guided": False,
        "rank_in_batch": 34,
        "verdict": ("**Cleanest novel-RNA-guided candidate.** rve-family (NOT known "
                          "RNA-guided) with 3-species distribution (66 events). Guide "
                          "`AACTGACAAGCCCT` is non-repetitive with no composition-null "
                          "risk. Sites 2, 6, 7 show strong negative flank_dev at "
                          "consistent positions — hallmark of guide-mediated coherence "
                          "rather than random target overlap. Peak_ratio 2.93, "
                          "moderate structure contribution. **Top pick for follow-up.**"),
    },
    "CDS00724": {
        "family": "HTH_38 + rve (retroviral integrase family)",
        "known_rna_guided": False,
        "rank_in_batch": 3,
        "verdict": ("Highest raw score of the 4 novel candidates (7.66) and sharpest "
                          "peak (peak_ratio 6.94), 3 species with 240 events. **BUT**: "
                          "guide `CAAAAAAAGAAGTA` is 9 consecutive A's — likely "
                          "composition-null artifact rather than genuine RNA guide. "
                          "Structure Δ=-5.74 is large but consistent with AT-tract "
                          "folding. **Requires dinuc-shuffle null control before "
                          "prioritization.**"),
    },
    "CDS03122": {
        "family": "MULE + Transposase_mut (Mutator-like Element)",
        "known_rna_guided": False,
        "rank_in_batch": 22,
        "verdict": ("**Rare-in-prokaryotes MULE family** with an extraordinary 747-event "
                          "footprint across 2 species. Guide `TAATTATTTTGCAG` AT-rich "
                          "but not homopolymeric; context "
                          "`...NNNNC[TAATTATTTTG]CAGGAGGACA...` shows a **GAGGACA "
                          "Shine-Dalgarno-like sequence just downstream** of the guide "
                          "— consistent with an actively transcribed ncRNA locus. "
                          "Peak well-localized (62% within ±5 bp). **Secondary pick.**"),
    },
}


def parse_probe(text: str) -> dict:
    """Extract key metrics from a v84_probe.txt output."""
    d = {}
    m = re.search(r"# TOP: pos=(\d+)\s+score=([-\d.]+)", text)
    if m:
        d["top_p"] = int(m.group(1))
        d["top_v"] = float(m.group(2))
    m = re.search(r"##\s*top-5 peaks.*?\n(  #1.*?)(?:\n\n|\n##)", text, re.S)
    if m:
        peaks = re.findall(r"#(\d+)\s+pos=\s*(\d+)\s+score=\+?([-\d.]+)", m.group(1))
        d["peaks"] = [(int(r), int(p), float(s)) for r, p, s in peaks]
    m = re.search(r"guide-candidate @ top_p \(L=14\):\s*'([^']+)'", text)
    if m:
        d["guide_L14"] = m.group(1)
    m = re.search(r"±20 bp context:\s*(\S+)", text)
    if m:
        d["context"] = m.group(1)
    m = re.search(r"raw records for \S+: (\d+)", text)
    if m:
        d["n_records"] = int(m.group(1))
    m = re.search(r"##\s*zero-ablation.*?\n((?:  zero .*\n)+)", text)
    if m:
        d["ablations"] = re.findall(r"zero\s+(\S.*?)\s+new_bag_max=\S+\s+Δ=([-+\d.]+)",
                                                m.group(1))
    m = re.search(r"##\s*localization.*?\n((?:  keep only.*\n)+)", text)
    if m:
        d["localization"] = re.findall(r"± ?(\d+) bp\).*?new_bag_max=([-+\d.]+)\s+Δ=([-+\d.]+)",
                                                   m.group(1))
    return d


def build_summary(cid: str, probe: dict, verdict: dict) -> str:
    lines = [f"# {cid} — deep interpretability probe summary\n"]
    lines.append(f"## Family + known-RNA-guided status")
    lines.append(f"- Pfam family: {verdict['family']}")
    known = "YES" if verdict["known_rna_guided"] else "**NO** (candidate novel RNA-guided family)"
    lines.append(f"- Known RNA-guided family? {known}")
    lines.append(f"- v84 top-100 rank (realness heuristic): {verdict['rank_in_batch']}")

    lines.append(f"\n## Model peak-position readout")
    lines.append(f"- Top position: nc[{probe.get('top_p', '?')}], score = **{probe.get('top_v', '?')}**")
    if probe.get("peaks"):
        lines.append(f"- Top 5 peaks:")
        for r, p, s in probe["peaks"]:
            lines.append(f"  - #{r}: nc[{p}] score={s:+.3f}")
    if probe.get("guide_L14"):
        lines.append(f"- Guide candidate at top_p (L=14): `{probe['guide_L14']}`")
    if probe.get("context"):
        lines.append(f"- ±20 bp context: `{probe['context']}`")

    if probe.get("ablations"):
        lines.append(f"\n## Zero-ablation (contribution of each channel group)")
        lines.append("| Channel group | Δ score |")
        lines.append("|---|---|")
        for name, delta in probe["ablations"]:
            lines.append(f"| {name} | {delta} |")

    if probe.get("localization"):
        lines.append(f"\n## Localization (score retained when only ±W bp around top_p is kept)")
        lines.append("| Window | new bag_max | Δ |")
        lines.append("|---|---|---|")
        for w, nb, d in probe["localization"]:
            lines.append(f"| ±{w} bp | {nb} | {d} |")

    lines.append(f"\n## Verdict\n\n{verdict['verdict']}")
    lines.append(f"\n---\n_Raw probe output: `v84_probe.txt`_")
    return "\n".join(lines) + "\n"


def append_to_readme(readme_path: Path, cid: str, summary_line: str, verdict: dict) -> None:
    txt = readme_path.read_text()
    marker = "\n## Deep interpretability probe\n"
    if marker in txt:
        return  # already appended
    section = [
        f"\n## Deep interpretability probe\n",
        f"See `probe_summary.md` and `v84_probe.txt` in this folder.\n",
        f"- Known RNA-guided family? {'YES' if verdict['known_rna_guided'] else '**NO** (candidate novel)'}",
        f"- v84 top-100 rank: {verdict['rank_in_batch']}",
        f"- **Verdict**: {verdict['verdict']}",
    ]
    readme_path.write_text(txt + "\n".join(section) + "\n")


def main():
    for cid, verdict in VERDICTS.items():
        folder = CAND_ROOT / cid
        probe_txt = folder / "v84_probe.txt"
        if not probe_txt.exists():
            print(f"[SKIP {cid}] no v84_probe.txt")
            continue
        probe = parse_probe(probe_txt.read_text())
        summary = build_summary(cid, probe, verdict)
        (folder / "probe_summary.md").write_text(summary)
        append_to_readme(folder / "README.md", cid,
                                summary.splitlines()[0], verdict)
        print(f"  {cid}: wrote probe_summary.md + updated README.md")


if __name__ == "__main__":
    main()
