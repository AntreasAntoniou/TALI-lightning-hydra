#!/bin/bash
#SBATCH --job-name=nvidia-smi
#SBATCH --output=nvidia-smi.out
#SBATCH --error=nvidia-smi.err

source $HOME/.bashrc
source $CODE_DIR/setup_scripts/setup_base_experiment_disk.sh
source $CODE_DIR/setup_scripts/setup_tali_dataset_disk.sh