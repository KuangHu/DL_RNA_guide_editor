#!/bin/bash
#SBATCH --job-name=v7real_full_gen
#SBATCH --partition=lr4
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/v7real_full_gen_%j.out
# CPU partition (2026-09-13): v7-real generation is pure-Python bag
# construction (no fold — that happens at loader time). Earlier A40
# reservation was wasteful; entire 160k gen finished in ~40s. Kept 8
# CPUs so the 8-worker mp.Pool is saturated; walltime tightened to 1h.
set -euo pipefail    # -e added 2026-09-13 (same rationale as v7real_train_main.sbatch)
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

OUT_DIR=/global/scratch/users/kh36969/DL_novel_guide_editor/v7real_full
mkdir -p "$OUT_DIR" logs

# ---- v7-real 5-corpus generation (mirrors v7_full_gen sizes so v7↔v7-real
#      training-hyperparam comparison is controlled).
# name          | mode       | n_bags | seed
# --------------+------------+--------+------
# pos50k        | none       | 50000  | 9000
# twin50k       | twin       | 50000  | 9100
# partial40k    | partial    | 40000  | 9200
# ctrl10k       | none       | 10000  | 9300
# scat10k       | scattered  | 10000  | 9400
#
# Notes:
#   * --v7-real, not --v7 — routes to build_bag_v7_real (real 60+60
#     bacterial flanks + multi-region synthetic nc + reversed flow).
#   * --negative-mode is REQUIRED under --v7-real; ctrl uses "none"
#     with a distinct seed to give a second independent positive draw.
#   * No --gc-target / --n-sites / --nc-homology-rate overrides:
#     v7-real samples each internally (bag-level gc U[0.30..0.70] via
#     bag_v7_real defaults; n_sites U{3..8}; nc_homology_rate is
#     synthetic-nc-only under v7-real so this axis is uniform 1.0 by
#     construction — the axis's v6r2 role of "how similar is the nc to
#     the guide" is subsumed by planted_m).

run_one() {
    local name=$1 mode=$2 n=$3 seed=$4
    local jsonl="$OUT_DIR/v7real_${name}.jsonl"
    local stats="$OUT_DIR/v7real_${name}_stats.json"
    local manif="$OUT_DIR/v7real_${name}_manifest.json"
    echo ""
    echo "[v7real-full-gen] ================================================"
    echo "[v7real-full-gen] $name  mode=$mode  n_bags=$n  seed=$seed"
    echo "[v7real-full-gen] ================================================"
    $PY -u -m scripts.generator_v5.run_generator \
        --n-bags "$n" --seed "$seed" --v7-real --negative-mode "$mode" \
        --out "$jsonl" \
        --stats-out "$stats" \
        --manifest-out "$manif" \
        --workers 8 --progress-every 2000
}

run_one pos50k     none      50000 9000
run_one twin50k    twin      50000 9100
run_one partial40k partial   40000 9200
run_one ctrl10k    none      10000 9300
run_one scat10k    scattered 10000 9400

echo ""
echo "[v7real-full-gen] === wc on all outputs ==="
wc -l "$OUT_DIR"/v7real_*.jsonl

echo "[v7real-full-gen] DONE"
