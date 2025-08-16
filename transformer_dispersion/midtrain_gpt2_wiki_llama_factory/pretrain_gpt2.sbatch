#!/bin/bash

#SBATCH --job-name=pretrain-1gpu
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1           # 1 GPU for that process
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_out/pretrain-1gpu-%j.out
#SBATCH --error=slurm_out/pretrain-1gpu-%j.err

echo "Setting up environment variables and paths..."
# --- User-Specific Paths ---
# Please set your scratch directory before running
if [ -z "${CONTAINER_PATH}" ]; then
  echo "ERROR: Please set the CONTAINER_PATH environment variable."
  exit 1
fi

if [ -z "${OVERLAY_PATH}" ]; then
  echo "ERROR: Please set the OVERLAY_PATH environment variable."
  exit 1
fi

if [ -z "${WORK_DIR}" ]; then
  echo "ERROR: Please set the WORK_DIR environment variable."
  exit 1
fi

YAML_PATH=${WORK_DIR}/my/gpt2_wiki_pretrain.yaml

# if [ -z "${YAML_PATH}" ]; then
#   echo "ERROR: Please set the YAML_PATH environment variable."
#   exit 1
# fi

if [ -z "${TRITON_CACHE_DIR}" ]; then
  echo "ERROR: Please set the TRITON_CACHE_DIR environment variable."
  exit 1
fi

if [ -z "${HOST_CA_CERT_PATH}" ]; then
  echo "ERROR: Please set the HOST_CA_CERT_PATH environment variable."
  exit 1
fi

if [ -z "${CONTAINER_CA_CERT_PATH}" ]; then
  echo "ERROR: Please set the CONTAINER_CA_CERT_PATH environment variable."
  exit 1
fi

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "Starting Apptainer container and executing script..."

srun --chdir=${WORK_DIR} apptainer exec \
    --nv \
    --bind ${HOST_CA_CERT_PATH}:${CONTAINER_CA_CERT_PATH} \
    --overlay ${OVERLAY_PATH}\
    ${CONTAINER_PATH} \
    bash -c "FORCE_TORCHRUN=1 llamafactory-cli train ${YAML_PATH}"

echo "Job finished successfully."
