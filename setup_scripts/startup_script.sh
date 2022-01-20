#!/bin/bash
CONDA_DIR=$HOME/conda/
ource $CONDA_DIR/bin/activate
conda activate tali

cd $CODE_DIR
bash $CODE_DIR/setup_scripts/setup_base_experiment_disk.sh
bash $CODE_DIR/setup_scripts/setup_tali_dataset_disk.sh
wandb agent
