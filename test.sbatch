#!/bin/bash
#SBATCH -J pyxis_test
#SBATCH -p lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --export=ALL,ENROOT_CACHE_PATH=/tmp/ge38qav/enroot-cache,ENROOT_DATA_PATH=/tmp/ge38qav/enroot-data,ENROOT_RUNTIME_PATH=/tmp/ge38qav/enroot-runtime
#SBATCH --container-image="nvcr.io#nvidia/pytorch:23.07-py3"

nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"