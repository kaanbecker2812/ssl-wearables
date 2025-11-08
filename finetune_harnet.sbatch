#!/bin/bash
#SBATCH -J finetune_harnet
#SBATCH -p lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH -t 10:00:00
#SBATCH -o /dss/dsshome1/06/ge38qav/Pretrained_Models/slurm/%x_%j.out
#SBATCH -e /dss/dsshome1/06/ge38qav/Pretrained_Models/slurm/%x_%j.err

# ----- Container settings (LRZ) -----
#SBATCH --export=ALL,ENROOT_CACHE_PATH=/tmp/ge38qav/enroot-cache,ENROOT_DATA_PATH=/tmp/ge38qav/enroot-data,ENROOT_RUNTIME_PATH=/tmp/ge38qav/enroot-runtime
#SBATCH --container-image="nvcr.io#nvidia/pytorch:23.07-py3"
#SBATCH --container-mounts="/dss/dsshome1/06/ge38qav/Pretrained_Models/ssl-wearables:/workspace,/dss/dsshome1/06/ge38qav/Pretrained_Models/ssl-wearables/data/downstream:/data,/dss/dsshome1/06/ge38qav:/hosthome"

set -euo pipefail
echo "Node: $(hostname) | GPU(s): ${CUDA_VISIBLE_DEVICES:-N/A} | Start: $(date)"

# ---------------- paths inside the container ----------------
CODE=/workspace                       # your codebase (mounted)
DATA=/data                            # your datasets (mounted)
HOMEHOST=/hosthome                    # your LRZ $HOME (mounted)

# Create a fresh per-job output dir in your LRZ home
OUTDIR="${HOMEHOST}/Pretrained_Models/experiments/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${OUTDIR}"/{models,logs,reports}
echo "OUTDIR: ${OUTDIR}"

# Optional: put user-site binaries (pip --user) on PATH
export PATH="/root/.local/bin:$PATH"

# Make pip cache persistent across jobs
export PIP_CACHE_DIR="${HOMEHOST}/.cache/pip"
mkdir -p "${PIP_CACHE_DIR}"

export HYDRA_FULL_ERROR=1

echo "Python & CUDA in container:"
python -V || true
nvcc --version 2>/dev/null || true
nvidia-smi || true

# ---------------- install requirements into the container ----------------
echo "Installing requirements from: ${REQS_FILE}"
python -m pip install --upgrade pip wheel setuptools --user
if [ -f "${REQS_FILE}" ]; then
  python -m pip install --user -r "${REQS_FILE}"
else
  echo "WARNING: ${REQS_FILE} not found. Skipping 'pip install -r'."
fi

# ---------------- run the downstream evaluation ----------------

echo "Launching downstream_task_evaluation.py ..."
python downstream_task_evaluation.py \
  root="${CODE}" \
  model_path="${OUTDIR}/models" \
  logging_path="${OUTDIR}/logs" \
  report_path="${OUTDIR}/reports" \
  evaluation.flip_net_path="${CODE}/model_check_point/mtl_5_best.mdl" \
  data.data_root="${DATA}/EASE_30Hz_0.4w_0.2s" \
  evaluation.num_epochs=50 \

echo "Done: $(date)"
