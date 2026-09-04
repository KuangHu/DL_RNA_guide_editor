#!/bin/bash
#SBATCH --job-name=stage3_layers
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=3:00:00
#SBATCH --output=logs/stage3_layers_%j.out
#SBATCH --error=logs/stage3_layers_%j.err
set -uo pipefail
# NOTE: no -e; the checker exits nonzero on any FAIL and we want ALL layers to run.
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python
BATCH=/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/large_batch
MTBASE=/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/channel_a
SCRATCH=/global/scratch/users/kh36969/tmp/claude-50246/-global-home-users-kh36969-tools-DL-RNA-guide-edotor-classifer/0ccc8f3b-56a6-4130-9f44-1c40a872fe3f/scratchpad

echo "=== L1 + L2 on pos50k ==="
$PY -m scripts.generator_v5.generator_data_checker \
    --jsonl $BATCH/positives_v5_v6_stage2_pos50k.jsonl \
    --layer 1,2 \
    --report-out $SCRATCH/stage3_l12_pos50k.json 2>&1 | tail -200

echo "=== L2 on partial40k (flank_planted_vs_unplanted probe fires here) ==="
$PY -m scripts.generator_v5.generator_data_checker \
    --jsonl $BATCH/positives_v5_v6_stage2_partial40k.jsonl \
    --layer 2 \
    --report-out $SCRATCH/stage3_l2_partial40k.json 2>&1 | tail -80

echo "=== L2 on scat10k (flank + negative-mode probe fires) ==="
$PY -m scripts.generator_v5.generator_data_checker \
    --jsonl $BATCH/positives_v5_v6_stage2_scat10k.jsonl \
    --layer 2 \
    --report-out $SCRATCH/stage3_l2_scat10k.json 2>&1 | tail -80

echo "=== L2 on ctrl10k (control corpus) ==="
$PY -m scripts.generator_v5.generator_data_checker \
    --jsonl $BATCH/positives_v5_v6_stage2_ctrl10k.jsonl \
    --layer 2 \
    --report-out $SCRATCH/stage3_l2_ctrl10k.json 2>&1 | tail -80

echo "=== L3 on pos50k, flank_coherence=off ==="
$PY -m scripts.generator_v5.generator_data_checker \
    --jsonl $BATCH/positives_v5_v6_stage2_pos50k.jsonl \
    --shard-dir $MTBASE/mt_v6_stage2_pos50k \
    --layer 3 \
    --flank-coherence off \
    --workers 32 \
    --report-out $SCRATCH/stage3_l3_pos50k_off.json 2>&1 | tail -300

echo "=== L3 on pos50k, flank_coherence=jitter ==="
$PY -m scripts.generator_v5.generator_data_checker \
    --jsonl $BATCH/positives_v5_v6_stage2_pos50k.jsonl \
    --shard-dir $MTBASE/mt_v6_stage2_pos50k \
    --layer 3 \
    --flank-coherence jitter \
    --workers 32 \
    --report-out $SCRATCH/stage3_l3_pos50k_jitter.json 2>&1 | tail -300

echo "=== ALL LAYERS DONE ==="
