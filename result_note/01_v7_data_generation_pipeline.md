# 01 — Data generation pipeline (v7-real training generator)

**Date:** 2026-09-20
**Grounded in:** `scripts/generator_v5/bag_v7_real.py`,
`scripts/generator_v5/real_flank_pool.py`,
`scripts/generator_v5/bag_v2.py`, `scripts/generator_v5/run_generator.py`.

This is the pipeline that manufactures Channel B's synthetic training
bags. The user's short description — "get 120 bp, then get motif, then
fake RNA structure, then non-coding region noise" — is the correct
mental model. The details below make it precise.

**Reversed flow.** Unlike the earlier v6r2 generator (guide first, then
find flanks that match), v7-real starts with a real bacterial flank and
edits it toward a chosen guide. That lets the flank distribution be
real bacterial sequence — the deployment regime — rather than a
synthetic distribution the model could learn to identify.

---

## Locked design (2026-09-13 user directive)

| axis | value | note |
|---|---|---|
| flank source | 50-genome pool via `RealFlankPool` | real bacterial sequence, junction at position 60 |
| flank length | 120 bp (60 + 60) | matches Durrant real-flank convention |
| target center offset | U[−40, +40] from junction | target = the flank region to be rewritten |
| target L (motif length) | U{9..14} | wide, deliberately NOT tuned to Durrant's L=11 |
| planted_m | U{8..11} (bounded above by L) | controlled Hamming match count target ↔ guide |
| nc region length | U[100, 250] per region | covers IS621's 193/107 with margin |
| n_sites | U{3..8} | matches MAX_N_SITES=8 in the model |
| junction motif | RETIRED | weights collapsed to {0: 1.0} in `bag_v2.py` |
| multi-region policy | `concat_with_N_spacer` at load | spacer = `MAX_L − 1` = 11 Ns |

`NC_LEN_MIN` was raised from 80 → 100 in the 2026-09-13 revision to
eliminate a ~10 bp systematic lift in `scattered` mode
(`finding_gc_difficulty_residual`).

---

## Stage-by-stage pipeline (one bag)

Entry: `build_bag_v7_real(bag_id, rng, real_flank_pool, n_sites,
negative_mode, gc, at_max) → V7RealBagRecord`.

### Step 1 — sample the bag-shared guide (the motif)

```
bag_guide_L = U{9..14}
bag_guide   = gc-weighted random ACGT string of length bag_guide_L
```

Under positive / partial modes, all sites in the bag share this
`bag_guide` — the cross-site coherence signal the model is trained to
detect. Twin and scattered modes replace this with per-site guides
(see negative modes below).

### Step 2 — sample K real 120 bp flanks from the pool

`real_flank_pool.sample_flank_120(rng, at_max=0.70)` — one 120 bp
window per site, drawn uniformly across the 50-genome pool
(`/global/scratch/users/kh36969/DL_novel_guide_editor/v7_refactor/genome_pool`).

Filters at sample time:
- **AT-content ≤ 0.70** (rejects extreme AT windows that would look
  synthetic under downstream stats).
- **No N in window** (assembly gaps rejected).

Junction convention: position 60 in the returned 120-string is the
boundary — matches Durrant WT / deploy convention.

### Step 3 — pick a target region + planted_m per site

For each site independently:

```
center_offset  = U[-40, +40] relative to junction (position 60)
target_start   = clip(center_offset + junction − bag_guide_L/2,
                      0, 120 − bag_guide_L)
planted_m      = U{8, ..., min(11, bag_guide_L)}
```

The **target region** is `flank[target_start : target_start + L]`.
This is the substring of the real flank that will be rewritten in the
next step to hit a controlled Hamming-match count against the guide.

### Step 4 — rewrite the flank at the target region

`mutate_target_to_match(rng, target_seq, guide, target_m)` performs
the minimum-edit rewrite to reach exactly `planted_m` matches:

- If the natural match count `n_natural < planted_m`, flip
  `planted_m − n_natural` mismatched positions to guide-matching bases
  (chosen uniformly at random from the mismatched positions).
- If `n_natural > planted_m`, flip `n_natural − planted_m` matched
  positions to a random non-guide base.

Result: the site's flank now matches the guide at exactly
`planted_m` positions inside the target window, with the rest of the
flank untouched. This is the cross-site coherence signal — same guide
"visible" at consistent positions across the K sites of the bag.

Sites that are **not planted** (partial mode's un-planted sites) keep
the original flank verbatim — no rewrite.

### Step 5 — synthesize two nc regions (the "fake RNA structure" + noise)

```
nc_planted_len = U[100, 250]
nc_noise_len   = U[100, 250]
nc_planted_base = gc-weighted random ACGT of length nc_planted_len
nc_noise        = gc-weighted random ACGT of length nc_noise_len
```

Both nc regions are synthetic. `sample_ncrna` from `bag_v2.py` draws
GC-weighted random ACGT — the fold structure emerges later at load
time via `compute_features_v2` (ViennaRNA), which computes real
BPP-based structural channels on whatever sequence the loader is
handed. There is no "structural sampling" here; the structure is
whatever the synthesized random sequence happens to fold into.

`nc_planted` will get the guide implanted (Step 6); `nc_noise` stays
as pure random background — this is the "non-coding region noise"
part of the user's description.

### Step 6 — plant the guide(s) in `nc_planted`

`plant_guide_in_nc(rng, nc_planted_base, guide, edge_avoid=MAX_L)`:

- Pick a random position `pos ∈ [MAX_L, len(nc) − MAX_L − L]`.
- Overwrite `nc[pos : pos + L]` with the guide bases.
- nc length is preserved.

Edge avoidance = `MAX_L = 12` bases on both ends. Rationale: the
loader's concat_with_N_spacer path puts an 11-N spacer between
regions; keeping the guide ≥ MAX_L from each nc end guarantees no
length-L search window can straddle the spacer and pick up a false
match involving guide bases + spacer Ns.

For **scattered** mode, `plant_multiple_guides_in_nc` places
`N_SCATTERED_GUIDES = 3` guides in `nc_planted`, spaced by at least
`MAX_L = 12` bp so no length-L window can straddle two consecutive
guides.

### Step 7 — shuffle regions, mark `active_noncoding_index`

```
order = [0, 1]; rng.shuffle(order)
noncoding_regions = [nc_planted, nc_noise] if order[0]==0
                    else [nc_noise, nc_planted]
active_noncoding_index = order.index(0)   # 0 or 1
```

The bag emits **two** nc regions. Their order is shuffled per bag,
and `labels.active_noncoding_index` records which slot holds the
planted region — gold, consumed by the target builder to place `y`
in the loader's concatenated coordinate frame.

---

## Negative modes

| mode | how the flank is written | what's in `nc_planted` | cross-site coherence in nc |
|---|---|---|---|
| `none` (= positive) | all K sites rewritten toward the **shared** `bag_guide` | one copy of `bag_guide` at random position | **K sites hit the same nc position** — the RNA-guided signature |
| `twin` | each site rewritten toward its **own independent** guide | one **unrelated** guide (not any site's guide) | zero — no nc position matches any site's target |
| `partial` | `n_planted ∈ {1..K−1}` sites rewritten toward `bag_guide`; rest untouched | one copy of `bag_guide` | only the planted sites hit the shared nc position (partial coherence) |
| `scattered` | each site rewritten toward one of 3 shared candidates (picked uniformly per site) | all 3 candidate guides at distinct positions | partial — each nc position drawn by ~K/3 sites |

Twin is the **strong negative** at training time — same flank
distribution and same nc distribution as the positive, but the target
sequences don't align across sites in the flank and don't match the
guide in nc. It's the twin construction that makes the model learn
"cross-site coherence" as the signal rather than "any bag with a
recognizable motif."

(This is different from the **deployment-time** strong negative in
`02_v7_strong_negative_data.md`, which is real bacterial DDE-family IS
insertions. Twin here defends synth training; DDE there validates real
inference.)

---

## Emitted JSONL structure

`v7_real_to_jsonl_records(bag) → list[dict]` writes one record per site.
All records in a bag share `transposase_id = bag_id`. Per-site record:

```json
{
  "site_id":        "<bag_id>_site_i",
  "transposase_id": "<bag_id>",
  "ncrna_id":       "<bag_id>_ncrna",
  "inputs": {
    "flank": "<120 bp — rewritten if planted, else raw>",
    "noncoding_regions": ["<region_0>", "<region_1>"]  // shuffled order
  },
  "labels": {
    "is_positive": bool,                        // True iff negative_mode == "none"
    "is_planted":  bool,                        // per-site; true unless un-planted (partial)
    "guide_length": int,                        // = bag.bag_guide_L
    "target_position_in_flank": int,            // per-site target_start
    "guide_span_in_active_noncoding": [start, end] | null,
                                                // gold guide position — in CONCAT coord
                                                // (accounts for active_noncoding_index offset
                                                // + MAX_L−1 spacer if active_index==1)
    "active_noncoding_index": 0 | 1,            // which region holds the guide
    "num_noncoding_regions": 2,
    "ncrna_length": int,                        // len of active nc
    "arch": {
      "n_sites": int,                           // K
      "nc_multi_region_scoring": "concat_with_N_spacer",
      "nc_homology_rate": 1.0,
      "flank_offset_mode": str
    },
    // TRAIN_ONLY fields (target-only or bookkeeping):
    "planted_start", "m_at_planted", "n_mismatches",
    "planted_A_end", "planted_B_end", "bag_target_m",
    "perfect_guide_dna", "epsilon_align", ...
  }
}
```

The concat-coord conversion for `guide_span_in_active_noncoding` is
non-trivial: if `active_noncoding_index == 1`, the loader concatenates
`[region_0, spacer, region_1]`, so the true guide position in the
loader's coordinate frame is `len(region_0) + (MAX_L−1) +
planted_pos_in_region_1`. The v7-real emit path in
`v7_real_to_jsonl_records` computes this shift correctly — the
2026-09-15 "9 h wasted A40 training" incident happened because an
earlier version emitted `planted_pos` (local coord) instead, so
`_build_target` read zeros everywhere and the model learned to predict
zero.

---

## Runner

`scripts/generator_v5/run_generator.py --v7-real ...` orchestrates:

- Worker init loads `RealFlankPool.load_default()` (50 genomes) per
  process. No `is_sites` pool, no rate table (v7-real path skips both).
- One bag per worker call via `build_bag_v7_real` →
  `v7_real_to_jsonl_records`.
- Streams JSONL to disk; holds only per-bag summary stats in memory
  (~50 MB for 50 K bags).
- Emits an acceptance stats JSON at the end (mode counts, mean
  planted_m, nc_len distributions, etc.).

Typical corpus: 50 K bags per mode (pos50k, twin, partial, scattered);
mode selection is one CLI switch per launch.

---

## Guarantees the pipeline maintains

1. **Real flank distribution.** The 120 bp flank base is drawn from
   real bacteria; only the target window is rewritten. AT-content
   filter and N-mask filter keep it in the plausible deployment regime.
2. **Controlled coherence gap.** planted_m is uniform in {8..11},
   giving a wide match-count distribution across bags. The generator
   does not degenerate to a single planted_m the model could memorize.
3. **Twin defends the classifier.** Twin bags share every axis with
   positives (flanks, nc lengths, gc, K, orient) except the presence
   of cross-site coherence — so any signal the model uses to
   distinguish must be coherence, not a shortcut.
4. **No cross-boundary matches.** Edge avoidance = MAX_L and multi-
   guide gap = MAX_L guarantee that no length-L search window can
   straddle either the spacer or two planted guides. Combined with
   the loader's concat_with_N_spacer with spacer_len = MAX_L − 1, the
   model can never see a false match involving spacer bases.
5. **Target-position coord frame is emitted correctly.** The two-
   region shuffle + concat requires a coordinate shift that the emit
   path computes explicitly. Bugs in this shift silently zero the
   training target (the 2026-09-15 lesson).

---

## What "generate data" does NOT include

- **Structure design.** The nc is random ACGT; the RNA structure is
  whatever ViennaRNA folds it into. There is no attempt to sample nc
  by structural criteria at generation time. If the classifier turns
  out to need a structural prior in the negative, that would be a
  design change to this pipeline, not a knob on it.
- **Real-data corpora.** The v7-real generator produces synth training
  bags only. Real bags (Durrant WT, DDE negtop10, fna_ins_discovery)
  are ingested separately by adapters and scored by the frozen model.
- **Cache / shard build.** Once JSONL exists, the offline shard build
  (`scripts/build_v7_shard.py`) and the loader
  (`model/channel_b/data.py`) are the preprocess side —
  see `03_v7_preprocess_and_model.md`.

---

## Consumers so far

- **v7-real Channel B training** — the frozen `v7real_main/best.pt`
  was trained on 50 K bags/mode across `{none, twin, partial,
  scattered}` = 200 K bags produced by this pipeline.
- **Held-out-m scope tests** — same pipeline with a filter on
  `median(planted_m) == 8` for the held-out gate
  (`finding-v7real-gates`).
- **Ablations & diagnostics** — every gate in `FROZEN.md` v7-real
  section walks through data made here.
