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

# Optional: put user-site binaries (pip --user) on PATH (both container + host)
export PATH="/root/.local/bin:${HOMEHOST}/.local/bin:$PATH"

# Use fast local scratch for tmp/cache if SLURM provides it
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export PIP_CACHE_DIR="${TMPDIR}/pip-cache"
mkdir -p "${PIP_CACHE_DIR}"

# Keep pip installs inside the job directory instead of $HOME
PYUSERBASE="${OUTDIR}/.pyuser"
mkdir -p "${PYUSERBASE}"
export PYTHONUSERBASE="${PYUSERBASE}"
export PATH="${PYUSERBASE}/bin:${PATH}"
PYVER="$(python -c 'import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")')"
export PYTHONPATH="${PYUSERBASE}/lib/python${PYVER}/site-packages:${PYTHONPATH:-}"

# Limit BLAS threads to the CPUs you asked for
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${OUTDIR}/.mpl"

# ---------------- repo-specific knobs (EDIT as needed) ----------------
REQS_FILE="${CODE}/req.txt"         # where requirements live

python -V || true
nvcc --version 2>/dev/null || true
nvidia-smi || true

# ---------------- install requirements ----------------
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

echo "Done: $(date)"
