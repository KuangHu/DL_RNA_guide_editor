#!/bin/bash
#SBATCH --job-name=v8_warm2
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=4:00:00
#SBATCH --output=logs/v8_warm2_%j.out
#SBATCH --exclude=n0005.es1,n0066.es1
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

# V8 pre-warm v2: writes 20-channel tensors (with flank_bg_identity)
# to the schema-tagged cache subdir. Old 19-channel cache under
# schema=<old_hash> is invisible to this pre-warm (different SCHEMA_KEY)
# and untouched. Config is 7 corpora (repeat_flank added).
#
# CPU-only, 52 parallel DataLoader workers → ~30-60 min per mode.

CACHE_ROOT=/global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v8
WORKERS=52

echo "[v8_warm2] start at $(date)"
echo "[v8_warm2] using schema-tagged cache subdir; old 19-ch cache is dormant."
$PY -u -m scripts.prewarm_channel_b_cache \
    --config config/channel_b_v8_corpora.json \
    --cache-root "$CACHE_ROOT" \
    --workers "$WORKERS"

echo "[v8_warm2] cache dir summary:"
for D in "$CACHE_ROOT"/*/; do
    NAME=$(basename "$D")
    # count schema subdirs and their contents
    for SD in "$D"schema=*/; do
        [ -d "$SD" ] || continue
        N=$(find "$SD" -name "*.pt" 2>/dev/null | wc -l)
        printf "  %-32s / %-24s  n_pt=%s\n" "$NAME" "$(basename $SD)" "$N"
    done
done

echo "[v8_warm2] DONE at $(date)"
