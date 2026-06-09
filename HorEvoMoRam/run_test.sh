#!/bin/bash
#SBATCH --job-name=test_horevomoram
#SBATCH --output=HorEvoMoRam/logs/test_job_%j.out
#SBATCH --error=HorEvoMoRam/logs/test_job_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --export=NONE

set -eo pipefail

# --- Setup ---
APP_DIR="/user/sshafi/u20715/neurogenesis"
cd "${APP_DIR}"
mkdir -p HorEvoMoRam/logs

# 1. Initialize environment for the compute node
source /etc/profile || true
module purge
module use /sw/modules/gwdg/rev/25.04/rome_mofed_cuda80_rocky8
module load gcc/13.2.0 python/3.11.9

# 2. Point to the existing pip packages
SITE_PKGS="${APP_DIR}/.venv/lib/python3.11/site-packages"
export PYTHONPATH="${SITE_PKGS}"

# 3. Tell HuggingFace to never try to reach the internet
export HF_HUB_OFFLINE=1

# 4. Make CUDA visible
NVIDIA_LIBS=""
for d in "${SITE_PKGS}"/nvidia/*/lib; do
    [ -d "$d" ] && NVIDIA_LIBS="${NVIDIA_LIBS:+$NVIDIA_LIBS:}$d"
done
export LD_LIBRARY_PATH="/usr/lib64:/usr/local/cuda/lib64:${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"

echo "============================================"
echo "Job ID:        ${SLURM_JOB_ID:-N/A}"
echo "Node:          ${SLURM_NODELIST:-N/A}"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time:    $(date)"
echo "Working dir:   ${APP_DIR}"
echo "Experiment:    Test HorEvoMoRAM Continual Learning"
echo "============================================"

# --- Run the test script ---
python3 -u HorEvoMoRam/test_hor_evo_moram.py

echo ""
echo "============================================"
echo "End time:      $(date)"
echo "============================================"
