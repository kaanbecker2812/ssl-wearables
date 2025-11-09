#!/bin/bash
#SBATCH -J ease_dss
#SBATCH -p lrz-dgx-1-p100x8          # adjust to your GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                     # 64G is usually enough; raise if needed
#SBATCH -t 10:00:00
#SBATCH -o /dss/dsshome1/06/ge38qav/Pretrained_Models/slurm/%x_%j.out
#SBATCH -e /dss/dsshome1/06/ge38qav/Pretrained_Models/slurm/%x_%j.err

set -euo pipefail
echo "Node: $(hostname) | GPU(s): ${CUDA_VISIBLE_DEVICES:-N/A} | Start: $(date)"

# ---------- Paths (edit these to your layout) ----------
CODE="$HOME/Pretrained_Models/ssl-wearables"           # cloned repo
VENV_DIR="$HOME/.venvs/harnet"                              # venv location in HOME
REQS_FILE="$CODE/req.txt"
OUTDIR="$HOME/Pretrained_Models/experiments/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"  # output dir for models/logs/reports

DATA="$HOME/Pretrained_Models/ssl-wearables/data/downstream"                                      # folder with your X.npy/Y.npy/pid.npy

mkdir -p "${OUTDIR}"/{models,logs,reports}

# Keep BLAS threads in check
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export HYDRA_FULL_ERROR=1
export PATH="$HOME/.local/bin:$PATH"   # user-site scripts (pip, virtualenv, etc.)

# --- Activate venv ---
source "$VENV_DIR/bin/activate"
python -V || { echo 'Python in venv not working'; exit 1; }
nvidia-smi || true

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
