"""Channel B — Stage 5 model.

Three principles locked before writing this file. Each was paid for in
past regressions.

1. Cross-site aggregation BEFORE per-site pooling. Every V5A-3 variant
   that pooled per site first hit a chance-level ceiling because the
   real guide is not per-site argmax (D5b). The site-symmetric attention
   pool must operate on the raw (m_max[L'=9..12], structure channels,
   flank_argmax[L']) tensor per nc position — the site axis collapses
   only after cross-site cross-position interaction.

2. Permutation-equivariance. The site axis is a set of size n_sites,
   not a sequence. Any layer that reads sites in a fixed order is a
   leak. Enforce with a Set-Transformer or masked-multihead attention
   over sites. Pre-registered gate C3: shuffling site order per bag
   must give AUROC = 0.500 exactly (n=5000 → SE 0.007, so |Δ| ≤ 0.014).

3. Inputs must be deploy-computable. NO gc_target, twin_p_true,
   planted_m, mismatch_positions. GC comes from the nc sequence at
   inference, not from labels.

   site_to_canonical_map is the PairwiseAligner output (same function
   at training and deploy). The mutation-model oracle map is preserved
   as labels.oracle_map for stratification by measured ε_align, but
   it MUST NOT enter the model's input tensor. Rationale: conjunction
   models amplify per-site alignment error as (1-ε)^n (Measurement A:
   0.6 nt shift → 30% coverage drop at n=5), so magnitude arguments
   like "0.4% is under window resolution" do not justify a training-
   time oracle input. Same function at both times, always.
"""
