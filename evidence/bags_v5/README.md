# bags_v5 metadata

Per-bag summary of the v5 insertion-discovery corpus (`fna_ins_discovery`,
built 2026-10-04). The matching full per-site records live under

```
/global/scratch/users/kh36969/fna_ins_discovery/bags_v5/insertions_sites.jsonl
```

That file is 2.4 GB (1.88 M site records). It is **not** on github — host it
on Hugging Face Hub or an object store if you need an off-cluster copy.

## insertions_bags.tsv

6139 bags, five columns:

| column | meaning |
|---|---|
| `bag_id` | `cds_cluster_id` (same as `transposase_id` in the scorer input) |
| `n_sites` | total per-site records for this bag in `insertions_sites.jsonl` |
| `n_observed` | sites whose empty allele is directly observed in a carrier |
| `n_reconstructed` | sites whose empty allele was reconstructed (TSD / alignment) |
| `n_species` | distinct species the carriers span |

`n_sites` ≠ the per-bag count in the scorer-input `adapted.jsonl`. The
scorer sees a stratified / downsampled subset of `insertions_sites.jsonl`;
the Step-1 audit (`evidence/_audit/`) reports those downsampled totals.
