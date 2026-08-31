#!/bin/bash
# V4.2 parameter sweep — 5000 sites × several (rank_cap, tail_prob) combos
set -e
SCRATCH=/global/scratch/users/kh36969/tmp/v41_cal
CFG=/global/scratch/users/kh36969/DL_novel_guide_editor/configs/transposases
GEN=/global/home/users/kh36969/tools/DL_RNA_guide_edotor_positive_generator
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python

cd "$GEN"
for RANK_CAP in 100 200; do
  for TAIL in 0.10 0.20 0.30; do
    if [ "$RANK_CAP" = "200" ] && [ "$TAIL" = "0.30" ]; then continue; fi
    OUT=$SCRATCH/assembled_v42_rc${RANK_CAP}_tp${TAIL}.jsonl
    echo "=== v42  rank_cap=$RANK_CAP  tail_prob=$TAIL  =>  $OUT ==="
    $PY -u -m scripts.assemble_sites \
      --ncrnas $SCRATCH/ncrnas_5k.jsonl \
      --sites  $SCRATCH/sites_5k.jsonl \
      --config-dir $CFG \
      --out $OUT \
      --seed 0 \
      --mismatch-strategy durrant_calibrated_rankaware \
      --rank-cap $RANK_CAP --tail-prob $TAIL --max-retries 8 2>&1 | tail -3
  done
done
