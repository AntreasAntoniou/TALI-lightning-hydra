#!/bin/bash
export HOME=/root/
source $HOME/.bashrc
source $HOME/conda/bin/activate
conda activate tali

cd $CODE_DIR
git pull
pip install -r $CODE_DIR/requirements.txt

source $CODE_DIR/setup_scripts/setup_base_experiment_disk.sh
source $CODE_DIR/setup_scripts/setup_wandb_credentials.sh

cd $CODE_DIR

fuser -k /dev/nvidia*; \
python $CODE_DIR/run.py \
hydra.verbose=True \
trainer=default \
resume=True \
batch_size=32 \
trainer.gpus=16 \
trainer.auto_scale_batch_size=True \
datamodule.config.rescan_paths=True \
datamodule.prefetch_factor=3 \
datamodule.num_workers=96 \
model=centi_modus_prime_resnet50 \
datamodule.config.dataset_size_identifier=base \
datamodule.config.modality_config.image=True \
datamodule.config.modality_config.text=True \
datamodule.config.modality_config.audio=False \
datamodule.config.modality_config.video=False 

