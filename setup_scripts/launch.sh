#!/bin/bash
export HOME=/root/
source $HOME/.bashrc
source $HOME/conda/bin/activate
conda activate tali

cd $CODE_DIR
git pull
pip install -r $CODE_DIR/requirements.txt


bash $CODE_DIR/setup_scripts/setup_base_experiment_disk.sh
bash $CODE_DIR/setup_scripts/setup_wandb_credentials.sh


cd $CODE_DIR

for i in {0..9}
do
  echo "Starting WANDB Agent ID: $i"
  screen -dmS startup_script_session bash -c 'wandb agent evolvingfungus/TALI-gcp-sweep-1/$WANDB_SWEEP_ID; exec bash'
  sleep 10
done
