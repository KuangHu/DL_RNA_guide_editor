# 02 — Strong negative data (twin construction from the positive pipeline)

**Date:** 2026-09-20
**Grounded in:** `scripts/generator_v5/bag_v7_real.py` — the same
generator that makes v7-real positives, invoked with a different
`negative_mode`.

The "strong" in strong-negative here is **matched-confounder**: the
negative is manufactured from the positive by breaking exactly one
rule — cross-site coherence in nc — and preserving every other axis.
The classifier can only tell them apart by using the coherence signal
itself; every shortcut (nc length, gc, flank distribution, orient mix,
n_sites, planted_m spread, structural fold, etc.) is neutralized by
construction.

This is the **training-time** strong negative, invoked from the same
`run_generator.py --v7-real` entry as the positive corpus. It's
different from the **deployment-time** validation negative (real DDE-
family IS elements from the `negtop10` catalog), which is documented
separately.

---

## The design principle: break exactly one rule

A v7-real positive bag holds K real bacterial flanks whose target
windows were **all rewritten toward the same `bag_guide`**, and an
`nc_planted` region containing **that same `bag_guide`** at one
position. Cross-site coherence at that nc position is the signal the
model is trained to learn.

To manufacture a strong negative, keep everything about the
construction identical **except** the "all sites converge on the same
nc guide" rule. Break that one axis and see if the classifier still
scores the bag high.

Everything shared with positive:
- 120 bp real bacterial flanks from the 50-genome pool (per site,
  independently drawn — same distribution)
- AT-content filter (≤ 0.70), N-mask filter
- Target center offset U[−40, +40] from junction
- Target L U{9..14}, planted_m U{8..11}
- Two synthetic nc regions U[100, 250] each, GC-weighted random ACGT
- Region shuffle + `active_noncoding_index` gold
- Concat with `MAX_L−1 = 11` N spacer at load
- n_sites U{3..8}, orient sampling, gc, guide length

What changes: only the guide identity across sites and/or in nc.

---

## Rule-breaks (the four modes at the same code call site)

`VALID_V7_REAL_NEGATIVE_MODES = ("none", "twin", "partial", "scattered")`
(line 76 of `bag_v7_real.py`). `is_positive = (negative_mode == "none")`.
All negatives are emitted with `is_positive = False`.

### `none` — positive (reference)

- All K sites rewritten toward the SHARED `bag_guide`.
- `nc_planted` contains ONE copy of `bag_guide` at one position.
- Coherence: **K sites align at the same nc position**. Full signal.

### `twin` — the primary strong negative

**Rule broken:** the bag no longer has a shared guide.

- Each of the K sites draws its OWN independent guide `G_i` and its
  flank target is rewritten toward `G_i`.
- `nc_planted` contains ONE **unrelated** guide `g_unrelated` — not
  matching any of the K per-site guides.
- Per-site m against any position in nc collapses to background: for
  random ACGT, expected Hamming distance to any per-site guide is
  ~3L/4 ≈ 7-10 out of 9-14. No explicit rejection sampling needed to
  guarantee non-match.

Coherence: **zero**. Sites' target sequences don't align across
flanks, and no site's guide is at any nc position. Every axis
matches positive; only cross-site coherence is missing.

This is why the model has to learn coherence rather than any
per-site shortcut. If the classifier used "does any site look like
it has a good guide match to the nc" as a signal, twin bags would
score just as high as positives — same flank sample, same nc, same
planted_m per site (each site DOES hit m ~ U{8..11} matches to its
own guide, just not to a shared guide).

### `partial` — coherence at reduced K

**Rule broken:** only some sites converge on the shared guide.

- `n_planted = U{1 .. K−1}` sites are rewritten toward `bag_guide`;
  the rest keep raw real flanks (no rewrite).
- `nc_planted` contains ONE copy of `bag_guide` at one position (same
  as positive).
- Coherence: `n_planted` sites at the shared nc position; the un-
  planted sites contribute background noise.

`is_positive = False`. This is a **hard** negative — the coherence
signal exists, just not at full K. Trains the model to be sensitive
to signal amplitude, not just presence. Also tests the model's
noise floor: at `n_planted = 1`, a single planted site out of K
should still leave the bag negative.

### `scattered` — coherence diluted across positions

**Rule broken:** the "one nc guide" rule.

- `N_SCATTERED_GUIDES = 3` shared candidates `{G_1, G_2, G_3}` drawn
  once per bag.
- Each site independently picks one of the three candidates and its
  flank is rewritten toward that pick.
- `nc_planted` contains **all three** candidates, spaced by at least
  `MAX_L = 12` bp (no L-window can straddle two).

Coherence: **≤ K/3** sites at any single nc position. The signal
exists in a diluted form — the model has to reject scattered
coherence and only fire on concentrated coherence (all K at one
position).

Why not simply skip scattered? Because a lazy attention model could
score high whenever any nc position has some cross-site m elevation,
regardless of dilution. Scattered forces the model to concentrate
attention.

---

## What each negative mode falsifies

| mode | shortcut it rules out |
|---|---|
| `twin` | "any bag with real bacterial flanks + random nc + K planted-m sites is positive" — same distribution across every axis, so any classifier that uses non-coherence signal would score twin ≈ positive |
| `partial` | "presence of any coherence in the bag is enough" — partial has real coherence, just at reduced amplitude |
| `scattered` | "high per-position m at any nc position is enough" — scattered has high per-position m at three positions, but each is only ≤ K/3 supported |

Positive vs any of these three passing is the operational statement
that the model learned coherence properly.

---

## How the negatives get generated at runtime

Same entry point as positives — `run_generator.py --v7-real
--negative-mode {twin|partial|scattered} --n-bags ...`. Each mode
writes its own JSONL. The training corpus is a stack:

| corpus | mode | n_bags | share |
|---|---|---|---|
| `v7real_pos50k.jsonl` | none | 50,000 | positives |
| `v7real_twin50k.jsonl` | twin | 50,000 | strong negatives |
| `v7real_partial50k.jsonl` | partial | 50,000 | partial-coherence |
| `v7real_scattered50k.jsonl` | scattered | 50,000 | scattered-coherence |

(Names are the convention in `checkpoints/channel_b/v7real_main`
training config.) Total training corpus: 200 K bags, 50/50 positive
vs any-negative when the loader mixes them, with the three negative
sub-modes contributing equally.

The training loss is per-position ordinal (per §3 of the spec —
`_build_target` in `data.py`), so a twin's y is a length-`nc_len_eff`
vector of zeros (nothing planted at any nc position); a partial's y
has count `n_planted` at one position; a scattered's y has count ~K/3
at each of 3 positions; a positive's y has count K at one position.
The model has to predict the WHOLE per-position count vector, not
just the bag scalar — which is what makes the strong-negative design
tight.

---

## Emitted labels distinguishing the modes

Every negative record has:

```json
"is_positive": false,
"negative_mode": "twin" | "partial" | "scattered",
```

For downstream stratified analysis, the emit path also records
per-site metadata that lets a diagnostic script reconstruct exactly
what happened:

- `is_planted` (per site) — true unless partial-untouched
- `per_site_guide` — the site's own guide (differs from bag_guide in
  twin/scattered)
- `per_site_nc_planted_pos` — the site's guide position in nc, or
  `None` if the site is not planted or the site's guide isn't in nc
  (twin case)
- `nc_planted_positions` — where any of the shared guide(s) sit in nc

The stratification meta in the loader's `BagInputs.meta` carries
`negative_mode` so ablation reports can slice per mode without
re-parsing JSONL.

---

## Why "strong negative" here means what it means

- **Same-distribution.** Every generator axis except the coherence
  rule is identical to positive. A classifier that scores twin low
  and positive high can only be doing so via cross-site coherence.
- **Falsifiability at construction time.** Twin's expected background
  match against any nc position is derivable from `L` and gc alone.
  We don't have to trust that the negative is negative — it is by
  construction.
- **Same code path.** Positive and negative are two calls to the same
  `build_bag_v7_real` function. No skew from separate pipelines,
  separate flank pools, or separate nc distributions.
- **Twin generalizes to real data.** The frozen deployment-time gate
  (real bacterial DDE-family IS elements, `negtop10`, AUROC 0.973-
  0.995 vs DurrantWT) is what confirms training on twin transfers to
  real non-RNA-guided bacterial data. See the note on the deployment-
  side validation set for that story.

---

## Related but distinct: the deployment-time negative

`real_data/negative_top10/` is a curated set of 63,048 real bacterial
DDE-family IS insertions (10 families, TSD-corroborated + HMM-screened
for RNA-guided carriers). It's a REAL-DATA validation of the frozen
classifier's specificity — the answer to "does the twin-trained
classifier also reject real non-RNA-guided IS elements?"

That set was NOT used to build the classifier. It is the deployment-
side test the classifier had to pass, and the AUROC 0.973-0.995 result
(vs DurrantWT real positive) is what retired the specificity concern.

If a later note covers that set, it will document the real-data
curation criteria (TSD-mode agreement, PF01385/PF07282 HMM screen,
IS200/605 wholesale exclusion) — distinct from the synth generation
described here.
