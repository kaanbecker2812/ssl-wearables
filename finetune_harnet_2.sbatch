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

DATA_ROOT="$HOME/Pretrained_Models/ssl-wearables/data/downstream"                                      # folder with your X.npy/Y.npy/pid.npy

mkdir -p "${OUTDIR}"/{models,logs,reports}
mkdir -p "$HOME/.venvs"

# Keep BLAS threads in check
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export HYDRA_FULL_ERROR=1

# ---------- Create/activate venv ----------
PYBIN="${PYBIN:-python3}"   # use system python3
$PYBIN -V
if [ ! -d "$VENV_DIR" ]; then
  echo "[venv] Creating at $VENV_DIR"
  $PYBIN -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -V
which python

python -m pip install --upgrade pip wheel setuptools --no-cache-dir

# ---------- Install requirements (no cache to save HOME quota) ----------
if [ -f "$REQS_FILE" ]; then
  echo "[pip] Installing repo requirements from $REQS_FILE"
  python -m pip install --no-cache-dir -r "$REQS_FILE"
else
  echo "[warn] $REQS_FILE not found; continuing."
fi

nvidia-smi || true
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'GPU:', torch.cuda.is_available())"

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
