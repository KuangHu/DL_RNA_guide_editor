#!/bin/bash
#SBATCH --job-name=stage3_l3_rerun
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0:30:00
#SBATCH --output=logs/stage3_l3_rerun_%j.out
#SBATCH --error=logs/stage3_l3_rerun_%j.err
set -uo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python
BATCH=/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/large_batch
MTBASE=/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/channel_a
SCRATCH=/global/scratch/users/kh36969/tmp/claude-50246/-global-home-users-kh36969-tools-DL-RNA-guide-edotor-classifer/0ccc8f3b-56a6-4130-9f44-1c40a872fe3f/scratchpad

echo "=== L3 off (rerun after per-n_sites monotonicity fix) ==="
$PY -m scripts.generator_v5.generator_data_checker \
    --jsonl $BATCH/positives_v5_v6_stage2_pos50k.jsonl \
    --shard-dir $MTBASE/mt_v6_stage2_pos50k \
    --layer 3 --flank-coherence off --workers 32 \
    --report-out $SCRATCH/stage3_l3_pos50k_off_v2.json 2>&1 | tail -12

echo "=== L3 jitter (rerun) ==="
$PY -m scripts.generator_v5.generator_data_checker \
    --jsonl $BATCH/positives_v5_v6_stage2_pos50k.jsonl \
    --shard-dir $MTBASE/mt_v6_stage2_pos50k \
    --layer 3 --flank-coherence jitter --workers 32 \
    --report-out $SCRATCH/stage3_l3_pos50k_jitter_v2.json 2>&1 | tail -12

echo "=== DONE ==="
