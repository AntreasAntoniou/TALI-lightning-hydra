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

fuser -k /dev/nvidia*; \
python $CODE_DIR/run.py \
hydra.verbose=True \
trainer=default \
resume=True \
batch_size=64 \
wandb_project_name=TALI-gcp-sweep-1 \
trainer.gpus=-1 \
trainer.auto_scale_batch_size=False \
datamodule.config.rescan_paths=True \
datamodule.prefetch_factor=3 \
datamodule.num_workers=96 \
model=base_modus_prime_resnet50 \
datamodule.config.training_set_size_identifier=milli \
datamodule.config.modality_config.image=True \
datamodule.config.modality_config.text=True \
datamodule.config.modality_config.audio=True \
datamodule.config.modality_config.video=True

#for i in {0..9}
#do
#  echo "Starting WANDB Agent ID: $i"
#  screen -dmS startup_script_session bash -c 'wandb agent evolvingfungus/TALI-gcp-sweep-1/$WANDB_SWEEP_ID; exec bash'
#  sleep 10
#done
