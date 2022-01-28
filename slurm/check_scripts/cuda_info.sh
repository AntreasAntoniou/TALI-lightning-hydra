#!/bin/bash
#SBATCH --job-name=nvidia-smi
#SBATCH --output=nvidia-smi.out
#SBATCH --error=nvidia-smi.err

source $HOME/.bashrc
export MOUNT_DIR="/mnt/disk/tali/"
export EXPERIMENTS_DIR="/mnt/disk/tali/experiments/"
export DATASET_DIR="/mnt/disk/tali/dataset/"

python $CODE_DIR/slurm/check_scripts/cuda_info.py