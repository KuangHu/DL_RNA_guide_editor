# 05 — Top-100 annotation, filters, and final shortlist

**Date:** 2026-09-21
**Grounded in:** `/global/scratch/users/kh36969/fna_ins_discovery/v7real_scored_v1/`
(`top_100_candidates.tsv`, `top100_annotation.tsv`,
`top100_site_independence.tsv`, `top100_tsd_signal.tsv`) and the
scripts `scripts/annotate_top100_is110.py`,
`scripts/site_independence_check.py`, `scripts/batch_tsd_signal.py`.

Starting point: the top-100 bags by `bag_max_score` from the v7-real
Channel B scoring of the fna_ins_discovery corpus (see Note 4 for the
scoring run; the ranked list is `v7real_scored_v1/top_100_candidates.tsv`).
Goal: figure out which of those 100 are **actually candidate
RNA-guided-like mobile elements** vs artifacts.

Every filter dropped the count substantially.

---

## The three-filter funnel

```
                                              count   %surviving
100 top v7 candidates by bag_max_score        100     100.0
   ↓ Filter 1: site independence (max pair
     flank identity < 0.95 across the K sites)
                                                26      26.0
   ↓ Filter 2: any clustering of the model's
     per-site flank_argmax at the top nc peak
     (std ≤ 10)
                                                11      11.0
   ↓ Filter 3: TSD-anchored clustering
     (argmax std ≤ 5 AND ≥60% of sites in
      the junction window [60-L, 72])
                                                 1       1.0
```

Each of the three filters is documented below.

---

## Filter 1 — Site independence

Script: `scripts/site_independence_check.py`.

For each bag, pairwise Hamming identity of the full 120-bp flanks
across up to 30 site records (from the source bags jsonl, before the
scoring-time subsample to 8 sites). Verdict per bag:

| verdict | rule | count | %  |
|---|---|---:|---:|
| **REDUNDANT** | max pairwise flank identity ≥ 0.95 | 73 | 73% |
| MIXED | 0.75 ≤ max identity < 0.95 | 1 | 1% |
| **INDEPENDENT** | max identity < 0.75 | **26** | 26% |

Interpretation of REDUNDANT: the K "sites" of the bag are the **same
DNA fragment observed in multiple strain assemblies**, not K
independent insertion events. The fna_ins_discovery pipeline groups
sites by CDS cluster, and a compound transposon (or plain
insertion) at a fixed chromosomal locus that is conserved across
many strains gets emitted as many "sites" with identical flanks.
Cross-site coherence in this case is by construction — the model
sees the same base sequence K times.

Concrete: even the #1-ranked bag (TNP02309, HTH_38, score +7.94)
has 193 sites total with max pairwise flank identity **0.992** —
30 replicas of the same fragment.

---

## Filter 2 — Model-side clustering of the coherence signal

Script: `scripts/batch_tsd_signal.py`.

Applied to the 26 INDEPENDENT bags from Filter 1. For each bag, run
the frozen v7-real model, find the top-scoring nc position (arg
max of `pred × pos_mask`), then for each L ∈ {9, 10, 11, 12} pull
the per-site `flank_argmax` from the shard at that top nc position.

- If all K sites' argmax cluster at similar flank offsets (low std),
  the model has real cross-site coherence to work with — every site
  is finding its best match at the same flank position.
- If argmax is scattered across the 120-bp flank, the model's
  `max over positions × orient` operator is finding some spurious
  low-complexity match in each flank at a different position, and
  the signal is not real coherence.

Verdict per bag on the L that maximizes n_in_TSD (then min std):

| verdict | rule | count | %  |
|---|---|---:|---:|
| **TSD-CLUSTERED** | std ≤ 5 AND ≥60% of sites in `[60-L, 72]` (junction window) | 1 | 3.8% |
| **MOTIF-CLUSTERED** | std ≤ 10 (tight but not at junction) | 10 | 38.5% |
| **SCATTERED** | std > 10 | 15 | 57.7% |

Concrete: TNP02696 (IS1, rank 69, score +6.97) is SCATTERED — at
its top nc peak, the per-site argmax spans offsets 4..107 with std
29 (see the standalone TSD-signal check in log
`logs/tsd_check_26222627.out`). This is despite TNP02696 being a
bona fide IS1 element (777 bp inserts, PF03400 e=1.6e-44, 9-bp TSD
mode). IS1's TSD is a **length** conservation, not a **sequence**
conservation — the actual 9-bp TSD is whatever DNA each insertion
happened to land in, and it differs across independent insertions.
So the model can't detect IS1 TSDs as cross-site coherence — but
it still scored the bag high, based on spurious AT-rich matches.

---

## Filter 3 — TSD-anchored

The single strictest condition: for the coherence to look like a
canonical target-site duplication signature, the argmax must cluster
at flank offsets right around the junction position 60. That's how
a real bridge-RNA / IS110-style guide would look: every site's flank
has the same short guide-target sequence right at the insertion
point.

Only **1 of 26 INDEPENDENT bags** passed: TNP02581.

---

## Shortlist

### Tier 1 — TSD-CLUSTERED (1 bag)

| bag | rank | species | v7 score | Pfam family | Pfam top hit | n_sites (total→scored) | argmax at peak | in TSD window |
|---|---:|---|---:|---|---|---|---|---|
| **TNP02581** | 23 | kpneu | +7.53 | Other transposase | HTH_28 | 3 → 3 | offset 57, std 0.0 (L=9) | 3/3 |

Notes: sole hit passing all three filters. Small n (3 sites total),
single species, but the flank alignment is textbook — 3 divergent
flanks (pairwise identity 0.35) all with matching L-mer at flank
position 57, right at the junction. Pfam is HTH_28 (helix-turn-helix
DNA-binding, catch-all "Other transposase" in our rules). Worth an
individual sequence-level look before any biological claim: 3 sites
is a small dataset.

### Tier 2 — MOTIF-CLUSTERED with n ≥ 6 near junction (3 bags)

| bag | rank | species | v7 score | Pfam family | Pfam top hit | n_sites | argmax at peak | in TSD window |
|---|---:|---|---:|---|---|---|---|---|
| **TNP05881** | 36 | spneumoniae | +7.39 | DDE / IS4-family | DDE_Tnp_1 | 8 | offset 63, std 5.0 (L=9) | 6/8 |
| **TNP00607** | 58 | ecoli | +7.14 | Non-mobility Pfam | HTH_38 (+ IS110-blast) | 8 | offset 61, std 9.3 (L=12) | 6/8 |
| **TNP01750** | 42 | ecoli | +7.34 | Integrase / rve | rve_3 (+ IS110-blast) | 6 | offset 48, std 9.7 (L=12) | 4/6 |

These are the **strongest non-canonical candidates**: full n=6 or 8
sites, tight-ish clustering (std ≤ 10) at offsets near the junction
(63, 61, 48). Each carries a Pfam annotation that isn't IS110 but
is consistent with a mobility mechanism:

- **TNP05881 (DDE_Tnp_1 / IS4-family)** — IS4 elements have 8–13 bp
  TSDs. Offset 63 (right of junction) with std=5 is compatible with
  the model picking up a shared TSD-like sequence. Best n=8
  clustering case in the whole top-100.
- **TNP00607 (HTH_38 + IS110-blast hit)** — 8 sites, 169 total sites
  across 3 species. Mixed annotation (IS110 by DNA-BLAST, HTH_38 by
  Pfam-HMM — likely a case where BLAST is hitting the IS110 rep's
  flanking region rather than the transposase). Offset 61 is
  essentially at the junction.
- **TNP01750 (rve integrase + IS110-blast hit)** — 6 sites. rve is
  the retroviral integrase catalytic core, also found in Mu-family
  transposases and phage integrases. Offset 48 is left of junction
  but within the L=12 window that spans the junction.

### Tier 3 — MOTIF-CLUSTERED with n = 3-4 OR offset far from junction (7 bags, more uncertain)

| bag | rank | species | v7 score | Pfam family | Pfam top hit | n | argmax at peak |
|---|---:|---|---:|---|---|---|---|
| TNP03443 | 8 | kpneu | +7.70 | no-Pfam-hit | – | 3 | off 48, std 0.5 (L=9), far from junction |
| TNP03093 | 39 | kpneu | +7.36 | no-Pfam-hit | – | 3 | off 5, std 1.9 (L=9), far left |
| TNP03748 | 70 | lmonocytogenes | +6.96 | Non-mobility | – | 3 | off 34, std 0.5 (L=12), left of junction |
| TNP02439 | 76 | efaecium | +6.82 | Non-mobility | – | 4 | off 76, std 0.0 (L=9), right of junction |
| TNP01934 | 81 | ecoli | +6.73 | no-Pfam-hit | – | 3 | off 61, std 9.5 (L=9), at junction |
| TNP00799 | 91 | ecoli | +6.58 | Non-mobility | – | 4 | off 11, std 2.7 (L=9), far left |
| TNP05749 | 94 | ecoli | +6.56 | Non-mobility | – | 4 | off 65, std 6.0 (L=11), at junction |

Small-bag (n=3-4) clustering, or clustering at flank positions far
from the junction. Could be:
- Genuine shared upstream regulatory element (e.g., a common
  promoter neighboring the insertion site)
- A common repeat sequence adjacent to conserved genomic contexts
- Small-n noise (3 sites clustering perfectly is a low bar)

Interesting notes:
- **TNP03443, TNP03093, TNP01934** all hit "no Pfam" — no annotated
  protein family for their dominant ORF. Uncharacterized proteins
  with clustered flank matches at fixed positions.
- **TNP00799 offset=11** (far from junction): could be a shared
  upstream signal.

---

## What we don't have

- **No obvious novel RNA-guided element.** Zero Pfam hits to PF01385
  (TnpB) or PF07282 (Cas12f-like) across the entire top-100 — no
  bridge-RNA-like TnpB carriers, no Cas12f-like elements.
- **No genuinely novel IS110 discovery.** 31 bags had IS110-family
  annotation (29 by DNA-BLAST + 2 by Pfam DEDD_Tnp_IS110), but the
  vast majority failed the site-independence filter — they're
  same-locus-multi-strain artifacts of known IS110 members already
  in the reference set.
- **Nothing that beats the Durrant IS621 baseline** for RNA-guided
  activity. The candidates we do have (Tier 1+2) are non-IS110
  mobility elements the model is detecting via genuine cross-site
  motif conservation — interesting but not novel RNA-guided.

---

## Broader finding (the real headline of this exercise)

**The fna_ins_discovery bag-construction pipeline produces bags with
non-independent sites 73% of the time.** Grouping sites by CDS
cluster without a within-bag flank-diversity constraint means any
conserved insertion locus present across multiple strains ends up
as a "K-site bag" that trivially satisfies cross-site coherence —
because the K "sites" are literally the same DNA fragment.

For downstream v7-real scoring to be meaningful on this corpus, the
bag builder needs one additional constraint at construction time:

> **Reject site pairs with flank Hamming identity ≥ 0.95 within a
>  bag** (either drop redundant sites, or split the bag into a
>  per-strain-diverse subset). This is the constraint the negtop10
>  curation enforced via TSD-corroboration + strain diversity, and
>  it's why negtop10 could deliver AUROC 0.973–0.995 for the
>  specificity gate.

Applying this constraint retrospectively to the current top-100 is
what Filter 1 above does — dropping 73% of the ranking. Applying
it at bag build time would surface a very different top-100, biased
toward genuinely multi-locus elements.

---

## Files in `v7real_scored_v1/`

| file | description |
|---|---|
| `top100_annotation.tsv` | per-bag BLAST + Pfam annotation |
| `top100_site_independence.tsv` | per-bag independence verdict + pairwise flank stats |
| `top100_tsd_signal.tsv` | per-bag TSD-signal verdict on the 26 INDEPENDENT bags |
| `top100_blast.tsv`, `top100_hmmscan.tbl` | raw BLAST + HMMER outputs |
| `top100_query_dna.fna`, `top100_query_protein.faa` | annotation queries |
| `annotation_summary.txt` | aggregated regime counts |

## Follow-up not yet run

1. **flank_dev vs m_max ablation** on the Tier 1 + Tier 2 bags
   (zero out ch 9-12 or ch 0-3 at the top peak, rescore) — would
   formally confirm the model is USING the clustered flank_dev
   signal, not just that the signal happens to be present in the
   input tensor.
2. **Per-bag sequence-level look** at TNP02581 (Tier 1) and TNP05881
   (Tier 2's best) — dump the actual nc L-mer + each site's flank
   L-mer alignment, see if it looks like a real biological motif.
3. **Rebuild the bag corpus with the site-independence constraint at
   construction time**, then re-score. Expected: a completely
   different top-100 with far fewer artifacts. Requires re-running
   the fna_ins_discovery pipeline with a within-bag flank-diversity
   filter.
