# Transfer to Lawrencium (LBL)

Move classifier + generator + dataset + Claude session state from
`igi.biotite` to `lrc-login.lbl.gov`.

| Endpoint | Value |
|---|---|
| Source host | `kuangh@igi.biotite` |
| Destination host | `kh36969@lrc-login.lbl.gov` |
| Destination home | `/global/home/users/kh36969` |
| Suggested data root | `/global/scratch/users/kh36969` (check purge policy) |

Two things that differ from a generic transfer:
- **Username changes** (`kuangh` → `kh36969`) → the Claude project-state
  directory has a path-derived name that changes on destination.
- **`/groups/rubin/...` won't exist on Lawrencium** and can't be created at
  root on a shared HPC → hardcoded paths get patched with one `sed` pass.

---

## 1. Inventory

| # | Source path | Size | Destination path |
|---|---|---|---|
| 1 | `/home/kuangh/tools/DL_RNA_guide_edotor_classifer/` | 13 MB | `/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/` |
| 2 | `/home/kuangh/tools/DL_RNA_guide_edotor_positive_generator/` | 2.3 MB | `/global/home/users/kh36969/tools/DL_RNA_guide_edotor_positive_generator/` |
| 3 | `/groups/rubin/projects/kuang/out/DL_novel_guide_editor/` | 76 GB | `/global/scratch/users/kh36969/DL_novel_guide_editor/` |
| 4 | `/home/kuangh/.claude/projects/-home-kuangh-tools-DL-RNA-guide-edotor-classifer/` | 5.9 MB | `/global/home/users/kh36969/.claude/projects/-global-home-users-kh36969-tools-DL-RNA-guide-edotor-classifer/` |
| 5 | `/home/kuangh/.claude/projects/-home-kuangh-tools-DL-RNA-guide-edotor-positive-generator/` (optional) | small | `-global-home-users-kh36969-tools-DL-RNA-guide-edotor-positive-generator/` |

Claude project-state directory name is derived from the workdir absolute
path (`/` → `-`). The rsync writes to the renamed directory directly; the
`.jsonl` transcript inside is UUID-named and needs no rename.

---

## 2. Transfer commands (from Lawrencium, ~5 rsyncs)

```bash
SRC=kuangh@igi.biotite

mkdir -p ~/tools ~/.claude/projects /global/scratch/users/kh36969

# code (fast)
rsync -avP $SRC:/home/kuangh/tools/DL_RNA_guide_edotor_classifer/          ~/tools/DL_RNA_guide_edotor_classifer/
rsync -avP $SRC:/home/kuangh/tools/DL_RNA_guide_edotor_positive_generator/ ~/tools/DL_RNA_guide_edotor_positive_generator/

# Claude session state (renamed on the fly to match new workdir path)
rsync -avP $SRC:/home/kuangh/.claude/projects/-home-kuangh-tools-DL-RNA-guide-edotor-classifer/ \
                 ~/.claude/projects/-global-home-users-kh36969-tools-DL-RNA-guide-edotor-classifer/

# optional: generator's Claude state
rsync -avP $SRC:/home/kuangh/.claude/projects/-home-kuangh-tools-DL-RNA-guide-edotor-positive-generator/ \
                 ~/.claude/projects/-global-home-users-kh36969-tools-DL-RNA-guide-edotor-positive-generator/

# 76 GB dataset (slow; overnight)
rsync -avP $SRC:/groups/rubin/projects/kuang/out/DL_novel_guide_editor/ \
                /global/scratch/users/kh36969/DL_novel_guide_editor/
```

`-avP` = preserve everything + resumable on interruption.

---

## 3. Patch hardcoded paths (single sed pass)

```bash
cd ~/tools/DL_RNA_guide_edotor_classifer

# dataset path
grep -rl '/groups/rubin/projects/kuang/out/DL_novel_guide_editor' scripts/ tests/ \
  | xargs sed -i 's|/groups/rubin/projects/kuang/out/DL_novel_guide_editor|/global/scratch/users/kh36969/DL_novel_guide_editor|g'

# sys.path prefix in scripts/
grep -rl "sys.path.insert(0, '/home/kuangh/tools/DL_RNA_guide_edotor_classifer')" scripts/ \
  | xargs sed -i "s|/home/kuangh/tools/DL_RNA_guide_edotor_classifer|/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer|g"

# GTDB path (only if you'll run the generator)
cd ~/tools/DL_RNA_guide_edotor_positive_generator
GTDB=/global/scratch/users/kh36969/GTDB
grep -rl '/groups/rubin/projects/kuang/db/Cerebras_db/sequence/GTDB' scripts/ configs/ tests/ 2>/dev/null \
  | xargs -r sed -i "s|/groups/rubin/projects/kuang/db/Cerebras_db/sequence/GTDB|$GTDB|g"
```

---

## 4. Re-setup notes for destination Claude Code

Once the rsyncs complete on Lawrencium, open Claude Code in
`~/tools/DL_RNA_guide_edotor_classifer/` and hand it this checklist. It
should figure out the exact commands based on what's actually available on
Lawrencium.

### 4.1 Environment probe
- [ ] `nvidia-smi` on a GPU node — note the GPU model + CUDA version.
- [ ] Available conda: is there a shared module (`module avail python`) or
      should we install miniconda locally under `$HOME`?
- [ ] `module avail viennarna` — only needed if regenerating structure
      cache. Not needed for training/inference on the 76 GB precomputed
      dataset we transferred.
- [ ] Read `~/.claude/projects/…/memory/MEMORY.md` for prior feedback
      (login-node discipline, etc.).

### 4.2 Recreate `opfi` conda env
Match the source-cluster env (Python 3.11, PyTorch 2.x on CUDA matching
the destination GPU). Interpreter path used by the sbatch scripts is
hardcoded — after creating the env, patch:
```bash
cd ~/tools/DL_RNA_guide_edotor_classifer
grep -rl '/home/kuangh/miniconda3/envs/opfi' scripts/ \
  | xargs -r sed -i 's|/home/kuangh/miniconda3/envs/opfi|<destination env prefix>|g'
```
`<destination env prefix>` is whatever `which python` reports inside the
recreated `opfi` env, minus `/bin/python`.

### 4.3 Slurm partition + QOS
Source scripts use `--partition=gpu_h200 --qos=standard`. Ask sysadmin or
check `sinfo -o "%P %a %G"` for GPU partitions. Once known:
```bash
cd ~/tools/DL_RNA_guide_edotor_classifer
grep -rl 'gpu_h200' scripts/ \
  | xargs -r sed -i 's|--partition=gpu_h200 --qos=standard|--partition=<DEST_PARTITION> --qos=<DEST_QOS>|g'
```

### 4.4 GPU memory adjustment
Batch size in `training/train_v1.py` is tuned for H200 (~141 GB). Smaller
GPUs (V100 16/32 GB, A40 48 GB) may need `--batch-size` reduced. Test with
a short dry-run before committing to a full training epoch. Not relevant
if you're only running inference on `checkpoints/v1_on_v3/best.pt`.

### 4.5 Verification target
Byte-clean transfer means running the packaged eval reproduces:
```
AUROC=0.9567    AUPRC=0.9274
HARD_AUROC=0.9309    HARD_AUPRC=0.9293
```
from the epoch-17 checkpoint on `test_v3.jsonl`. Reference eval script:
`scripts/eval_v1_on_v3_test.py`. Reproduce within ±0.001 → transfer is
clean.

### 4.6 Files-that-reference-absolute-paths (reference list)
Classifier (`/groups/rubin/projects/kuang/out/DL_novel_guide_editor`):
- `tests/test_alignment.py`, `test_candidates.py`, `test_dataset.py`,
  `test_site.py`, `test_structure.py`, `test_model_v1.py`
- `scripts/eval_v1_on_test.py`, `eval_v1_on_v3_test.py`,
  `diagnose_strength_ordering.py`, `diagnose_level3_vs_strong.py`
Generator (`/groups/rubin/projects/kuang/db/Cerebras_db/sequence/GTDB`):
- `configs/default.yaml`, `scripts/generate_flanking.py`, `tests/test_genome.py`

All are handled by the single sed pass in §3.
