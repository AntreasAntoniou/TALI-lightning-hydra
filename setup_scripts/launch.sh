#!/bin/bash
export HOME=/root/
source $HOME/.bashrc
source $HOME/conda/bin/activate
conda activate tali

cd $CODE_DIR
git pull

bash $CODE_DIR/setup_scripts/setup_base_experiment_disk.sh
bash $CODE_DIR/setup_scripts/setup_wandb_credentials.sh


cd $CODE_DIR
wandb agent evolvingfungus/TALI-gcp-sweep-1/$WANDB_SWEEP_ID