#!/bin/bash
# Per-nt RNAplfold structure precompute — one Slurm array task per NODE, with
# PROCS single-threaded shards running concurrently inside it.
#
# Why not one array task per shard (precompute_structure_array.sbatch):
# cf1 is OverSubscribe=EXCLUSIVE, so a job asking for 1 CPU is still handed a
# whole 64-core node. One shard per job therefore idles 63 cores and caps
# concurrency at the node count. Measured throughput is ~3.6 NC/s per core
# (compute-bound; node-local vs Lustre TMPDIR makes no difference), so the
# only way to finish train_v4's ~1.19 M NC rows in reasonable time is to fill
# the nodes we are already being charged for.
#
# Submit:
#   sbatch --export=SPLIT=train_v4,NUM_SHARDS=192,PROCS=48 --array=0-3 \
#          sbatch/precompute_structure_node.sbatch
#
# Array task k runs shards [k*PROCS, (k+1)*PROCS).  NUM_SHARDS must equal
# (number of array tasks) * PROCS.

#SBATCH --job-name=precomp_node
#SBATCH --partition=cf1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=cf_normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=6:00:00
#SBATCH --output=logs/precompnode_%x_%A_%a.out
#SBATCH --error=logs/precompnode_%x_%A_%a.err

set -euo pipefail
export PATH=/global/home/users/kh36969/.conda/envs/vrna/bin:$PATH
PYTHON=/global/home/users/kh36969/.conda/envs/opfi/bin/python

CODEDIR=/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
DATADIR=/global/scratch/users/kh36969/DL_novel_guide_editor
OUTBASE=${DATADIR}/structure
mkdir -p "$OUTBASE" "$CODEDIR/logs"

: "${SPLIT:?SPLIT env var required, e.g. train_v4}"
: "${NUM_SHARDS:?NUM_SHARDS env var required}"
: "${PROCS:?PROCS env var required (shards run concurrently on this node)}"

SPLIT_PATH=${DATADIR}/splits/${SPLIT}.jsonl
[[ -f "$SPLIT_PATH" ]] || { echo "split not found: $SPLIT_PATH" >&2; exit 1; }

TASK=${SLURM_ARRAY_TASK_ID:-0}
FIRST=$(( TASK * PROCS ))
LAST=$(( FIRST + PROCS - 1 ))

echo "[$(date -Iseconds)] node task $TASK -> shards $FIRST..$LAST of $NUM_SHARDS"
echo "  host:      $(hostname)   cores: $(nproc)"
echo "  split:     $SPLIT_PATH"
echo "  RNAplfold: $(which RNAplfold)"

cd "$CODEDIR"
pids=()
for (( s=FIRST; s<=LAST; s++ )); do
    (( s < NUM_SHARDS )) || break
    OUT_PATH=$(printf '%s/%s_u16_shard%03d' "$OUTBASE" "$SPLIT" "$s")
    # preprocess.structure runs RNAplfold with cwd set to its own private
    # TemporaryDirectory per batch, so concurrent shards cannot collide over
    # the <id>_lunp / <id>_dp.ps files it drops in the working directory.
    "$PYTHON" -m scripts.precompute_structure \
        --split "$SPLIT_PATH" \
        --out "$OUT_PATH" \
        --u-max 16 --W 120 --L 60 \
        --batch-size 200 \
        --shard-index "$s" \
        --shard-count "$NUM_SHARDS" \
        --progress-every 20000 \
        > "logs/shard_${SPLIT}_$(printf '%03d' "$s").log" 2>&1 &
    pids+=($!)
done

echo "  launched ${#pids[@]} shard processes"
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[$(date -Iseconds)] node task $TASK done (rc=$rc)"
exit $rc
