#!/bin/bash

#SBATCH --job-name=jax_gpu_test
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=jax_gpu_test-%j.out
#SBATCH --error=jax_gpu_test-%j.err

set -e

# ------------------------------------------------------------
# User configuration
# ------------------------------------------------------------

IMAGE=/path/to/jax-hpc-apptainer_1.0.0.sif
PROJECT_DIR=/path/to/jax-hpc-apptainer

# ------------------------------------------------------------
# Environment information
# ------------------------------------------------------------

echo "===== SLURM ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Time: $(date)"

echo
echo "===== NVIDIA GPU ====="
nvidia-smi

echo
echo "===== JAX GPU test ====="

# ------------------------------------------------------------
# Run inside Apptainer
# ------------------------------------------------------------

apptainer exec \
    --nv \
    --bind ${PROJECT_DIR}:/workspace \
    --pwd /workspace \
    ${IMAGE} \
    python examples/jax_gpu_test.py
