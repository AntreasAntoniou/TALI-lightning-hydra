#!/bin/bash
source /root/.bashrc
source /root/conda/bin/activate
conda activate tali

cd $CODE_DIR
git pull
bash $CODE_DIR/setup_scripts/setup_base_experiment_disk.sh
bash $CODE_DIR/setup_scripts/setup_tali_dataset_disk.sh
wandb agent evolvingfungus/TALI-gcp-sweep-1/5ls9k9ch


