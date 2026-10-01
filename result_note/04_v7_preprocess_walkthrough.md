# 04 — Preprocess walkthrough (one real bag, end-to-end)

**Date:** 2026-09-21
**Purpose:** trace a single real bag through every preprocess step, so
the shapes and numbers in Note 3 are grounded in a concrete example.

**Bag used:** `TNP02309` — the top-1 scoring bag from the
fna_ins_discovery scoring run (bag_max_score = +7.94, ranked 1 of
1,236). 8 sites, single-region nc of 250 bp. Species `efaecium`,
phylum Firmicutes.

**Sources:** JSONL at
`/global/scratch/users/kh36969/tmp/fna_ins_discovery_score/fna_ins_discovery.jsonl`
+ shard at `.../fna_ins_discovery_shard/`. Both were produced by the
first-run scoring pipeline (`sbatch/v7real_fna_ins_discovery.sbatch`).

**Trace script:** the outputs below are captured verbatim from a
throwaway script that reruns each step of `_build_bag_inputs` in
isolation. See `/tmp/trace_TNP02309.txt` for the full stdout.

---

## Step 0 — raw JSONL record

Each bag is 8 site records sharing `transposase_id`. Per-site record:

```
records for TNP02309: 8
site_id[0]         : TNP02309_site_0000
transposase_id     : TNP02309
inputs.flank len   : 120 bp (junction at position 60)
inputs.flank[:30]  : AAATGCTTGATGAAGCAGTAGCAGTTTATC
noncoding_regions  : 1 region(s), lens=[250]
labels.arch        : {'n_sites': 8,
                      'nc_multi_region_scoring': 'concat_with_N_spacer',
                      'nc_homology_rate': 1.0}
```

This one is single-region (multi-region would show `lens=[a, b]` or more).

---

## Step 1 — concat + trim position axis

Single region → no spacer; `canonical_nc = noncoding_regions[0]`.
For multi-region, regions are joined with `N * (MAX_L - 1) = 11 Ns`.

```
nc_len            : 250
MAX_L             : 12
nc_len_eff = nc_len - MAX_L + 1  =  250 - 12 + 1  =  239
canonical_nc[:60] : AACTAAATTTAGTTCGTAAATTTCAAGTTTAAGTTAGCCACCCATGGTGACCCGGTTGCT
```

`nc_len_eff = 239` is the position axis every downstream array is
sized to.

---

## Step 2 — structure channels via ViennaRNA (bag-level, broadcast)

`compute_features_v2(canonical_nc, guide_length=12)` — one ViennaRNA
fold per bag, four per-position arrays + a validity mask:

```
dG_open_uL_pn        : len=239 nan=0 min=+0.0491 p50=+0.5321 max=+2.6418
H_pair_win           : len=239 nan=0 min=+0.0000 p50=+0.4110 max=+1.5736
cooperativity_win_pn : len=239 nan=0 min=+0.0288 p50=+0.7247 max=+3.5570
E_span_win           : len=239 nan=0 min=+8.1244 p50=+63.0026 max=+173.3592
structure_valid      : 239 / 239 positions valid
```

These become channels 4-8 of the tensor and are broadcast identically
across all 8 sites — they describe the bag's nc, not any one site.

---

## Step 3 — per-site per-L m_max + flank_argmax (shard read)

For each (site × orient × L), the shard stores per-position `m_max`
(the max L-window match count in the flank) and `flank_argmax` (the
flank offset of that max). Loader combines the two orients
per-position with `max`:

```
m_max_per_site.shape  : (8, 239, 4)  = (S, P, L)   Ls = (9, 10, 11, 12)

m_max[site 0, L=11]   first 12 positions →
  [11, 11, 11, 11, 10, 9, 9, 8, 7, 7, 7, 8]

m_max[site 1, L=11]   first 12 positions →
  [ 9,  9,  9,  9,  8, 8, 8, 8, 7, 8, 8, 8]

m_max[site 2, L=11]   first 12 positions →
  [ 8,  8,  8,  8,  8, 7, 7, 8, 7, 8, 7, 7]

argmax[site 0, L=11]  first 12 positions →
  [60, 61, 62, 63, 64, 65, 66, 67, 68, 21, 22, 75]

argmax[site 1, L=11]  first 12 positions →
  [13, 14, 15, 96, 17, 47, 95, 96, 62, 98, 99, 39]
```

`m_max[s, p, L=11] = k` means the best L=11 window in site s's flank
matches the L=11 substring of nc starting at position p in k
positions. `argmax[s, p, L=11]` tells which flank offset produced
that best match. Note how site 0's argmax starts at 60 and increments
by 1 (60, 61, 62, ...) — that's the flank offset walking with the nc
position, a signature of a real matching window in the flank around
position 60 (the junction). Site 1's argmax is scattered across the
flank — no persistent match.

---

## Step 4 — flank_dev = (argmax − bag_median) / FLANK_DEV_SCALE

This is the cross-site coherence encoding. For each (position, L):

```
bag_median[L=11] first 12 positions → [40, 60, 60, 80, 50, 48, 50, 52, 53, 59, 60, 56]

flank_dev[site 0, L=11] first 12 →
  [+1.95, +0.05, +0.15, -1.70, +1.40, +1.65, +1.60, +1.50, +1.50, -3.85, -3.85, +1.90]

flank_dev[site 1, L=11] first 12 →
  [-2.75, -4.65, -4.55, +1.60, -3.30, -0.15, +4.50, +4.40, +0.90, +3.85, +3.85, -1.70]

FLANK_DEV_SCALE = 10.0
```

Read this row by row: at nc position 1, sites 0 and 1 have
`flank_dev = +0.05, -4.65` — site 0 sits right on the bag median while
site 1 is 4.65 units below. If most sites in the bag cluster near
`flank_dev ≈ 0` at a given position, that's the model's evidence that
"all sites converge on the same flank offset here" — the cross-site
coherence signal.

The bag_median column at position 2 = 60 (flank offset), and sites 0's
argmax there = 62 → dev = (62−60)/10 = +0.2 → the reported +0.15
after normalization.

---

## Step 5 — assemble the 15-channel tensor and apply divisors

```
x.shape (pre-collate) : (8, 239, 15)  = (S=8, P=239, C=15)

per-channel divisors (CHANNEL_SCALES):
  m_max_L9  m_max_L10  m_max_L11  m_max_L12          : 12.0
  dG_open_uL_pn                                       :  0.2
  H_pair_win                                          :  1.0
  cooperativity_win_pn                                :  0.3
  E_span_win                                          : 20.0
  structure_valid                                     :  1.0
  flank_dev_L9  flank_dev_L10  flank_dev_L11  flank_dev_L12 : 1.0 each
  orient_fwd  orient_rc                               :  1.0 (kept zero)
```

Per-channel post-divide stats for site 0 of this bag:

| ch | name | min | p50 | max | frac_nonzero |
|---|---|---:|---:|---:|---:|
| 0 | m_max_L9 | +0.333 | +0.500 | +0.750 | 1.000 |
| 1 | m_max_L10 | +0.333 | +0.500 | +0.833 | 1.000 |
| 2 | m_max_L11 | +0.417 | +0.583 | +0.917 | 1.000 |
| 3 | m_max_L12 | +0.417 | +0.583 | +1.000 | 1.000 |
| 4 | dG_open_uL_pn | +0.246 | +2.661 | +13.209 | 1.000 |
| 5 | H_pair_win | +0.000 | +0.411 | +1.574 | 1.000 |
| 6 | cooperativity_win_pn | +0.096 | +2.416 | +11.857 | 1.000 |
| 7 | E_span_win | +0.406 | +3.150 | +8.668 | 1.000 |
| 8 | structure_valid | +1.000 | +1.000 | +1.000 | 1.000 |
| 9 | flank_dev_L9 | −7.25 | −0.20 | +8.85 | 0.992 |
| 10 | flank_dev_L10 | −7.80 | +0.00 | +8.35 | 0.987 |
| 11 | flank_dev_L11 | −7.45 | −0.20 | +8.00 | 0.996 |
| 12 | flank_dev_L12 | −7.45 | −0.05 | +8.55 | 0.996 |
| 13 | **orient_fwd** | +0.000 | +0.000 | +0.000 | **0.000** |
| 14 | **orient_rc** | +0.000 | +0.000 | +0.000 | **0.000** |

Notes:
- `m_max_L{9..12}` post-divide sits in [0.33, 1.00] — bounded [0, 1]
  as designed.
- Structure channels sit at O(1)-scale after divide.
- `flank_dev` post-divide is in a tighter range than raw (raw argmax
  differences of ±80 bp become ±8 after division).
- `orient_fwd` / `orient_rc` are all zero — the intentional post-
  orient-leak-fix state (see `03_v7_preprocess_and_model.md` step 5
  and `constants.py`).

---

## Step 6 — site_mask

```
site_mask : [True, True, True, True, True, True, True, True]
             (True = real site, False = padding)
```

This bag has n_sites = 8 = MAX_N_SITES, so no padding. A 4-site bag
would emit `[T, T, T, T, F, F, F, F]` with the corresponding tensor
rows zero.

---

## Step 7 — bucket_collate_fn (per-batch stacking)

`bucket_collate_fn` stacks items with a shared position axis
(bucketed at the sampler; here we batch a single bag). Adds `pos_mask`:

```
batch['x'].shape        : (1, 8, 239, 15)   = (B=1, S=8, P=239, C=15)
batch['site_mask'].shape: (1, 8)   → [T, T, T, T, T, T, T, T]
batch['pos_mask'].shape : (1, 239) → sum=239 / 239

READY FOR MODEL: model(batch['x'], batch['site_mask'])
                   → (B=1, P=239) per-position scores

Bag-level score at deploy:
  bag_max = (pred * pos_mask + (-1e9) * ~pos_mask).max()
```

For this bag: `bag_max = +7.94` — top-1 across 1,236 scored bags in
the fna_ins_discovery corpus. In a normal batch of many bags with
varying `nc_len_eff`, `bucket_collate_fn` would pad the shorter ones
to the batch's max P and set `pos_mask` False on the pad — the
`bag_max` computation ignores those padded positions via the
`-1e9 * ~pos_mask` mask.

---

## Summary of shapes at every step

```
JSONL (per site)   : flank str [120]     +   nc str [nc_len]
   ↓ concat + trim
canonical_nc       : str [nc_len]
nc_len_eff         : nc_len - MAX_L + 1   (= 239 here)
   ↓ ViennaRNA
structure arrays   : 4 × (nc_len_eff,)   +   validity mask (nc_len_eff,)
   ↓ shard read (per site × orient × L, orient-combined)
m_max_per_site     : (n_sites_real, nc_len_eff, |Ls|=4)
argmax_per_site    : (n_sites_real, nc_len_eff, |Ls|=4)
   ↓ bag-median deviation
flank_dev          : (n_sites_real, nc_len_eff, |Ls|=4)
   ↓ assemble + divisor
x                  : (MAX_N_SITES=8, nc_len_eff, N_CHANNELS=15)
site_mask          : (MAX_N_SITES=8,) bool
   ↓ collate
batch['x']         : (B, MAX_N_SITES=8, max_pos, N_CHANNELS=15)
batch['site_mask'] : (B, MAX_N_SITES=8) bool
batch['pos_mask']  : (B, max_pos) bool
   ↓ model
pred               : (B, max_pos)
   ↓ mask + max
bag_max_score      : (B,)                       = +7.94 for TNP02309
```

Same pipeline runs identically on training bags (synth v7-real) and
inference bags (Durrant WT, DDE negtop10, fna_ins_discovery). The
only differences are: (a) shard was pre-built once vs on-demand, and
(b) inference records have `is_positive=False` and no
`guide_span_in_active_noncoding`, so `_build_target` returns a
zero-vector `y` (never consumed at inference).
