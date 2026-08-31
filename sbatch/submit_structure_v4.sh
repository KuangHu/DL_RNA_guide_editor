#!/bin/bash
# Build the RNAplfold structure cache for the V4 splits, then merge the shards.
#
#   bash sbatch/submit_structure_v4.sh
#
# cf1 hands out whole 64-core nodes (OverSubscribe=EXCLUSIVE), so we pack
# PROCS shards onto each node instead of one shard per job. RNAplfold runs at
# ~3.6 NC/s per core on these sequences, so train_v4 (~1.19 M NC rows) needs
# ~330 K core-seconds: 4 nodes x 48 procs finishes it in about half an hour.
# The merge for each split is chained on its array with afterok.
set -euo pipefail

CODEDIR=/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
OUTBASE=/global/scratch/users/kh36969/DL_novel_guide_editor/structure
PYTHON=/global/home/users/kh36969/.conda/envs/opfi/bin/python
PROCS=48
cd "$CODEDIR"

submit () {                      # $1 split name, $2 node count
    local split=$1 nodes=$2
    local shards=$(( nodes * PROCS )) last=$(( $2 - 1 ))
    local jid
    jid=$(sbatch --parsable \
        --export=SPLIT="$split",NUM_SHARDS="$shards",PROCS="$PROCS" \
        --array=0-${last} \
        sbatch/precompute_structure_node.sbatch)
    echo "  $split: array $jid — $nodes node(s) x $PROCS procs = $shards shards"
    sbatch --parsable --dependency=afterok:"$jid" \
        --job-name="merge_$split" --partition=cf1 --account=pc_rubinlab \
        --qos=cf_normal --nodes=1 --cpus-per-task=1 --mem=64G --time=3:00:00 \
        --output="$CODEDIR/logs/merge_${split}_%j.out" \
        --error="$CODEDIR/logs/merge_${split}_%j.err" \
        --wrap "cd $CODEDIR && $PYTHON -m scripts.merge_structure_shards \
                  --in-glob '$OUTBASE/${split}_u16_shard*.index.json' \
                  --out '$OUTBASE/${split}_u16' --delete-shards" \
        | sed "s/^/  $split: merge /"
}

submit train_v4         4
submit val_v4           1
submit test_v4          1
submit test_v4_control  1
