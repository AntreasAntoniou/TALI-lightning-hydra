#!/bin/bash
echo "Running admin defined startup script"
export MY_WANDB_API_KEY="NOT-AVAILABLE"
export CODE_DIR=/root/target_codebase
git clone https://github.com/AntreasAntoniou/TALI-lightning-hydra.git $CODE_DIR

echo "Starting WANDB Agent"
screen -dmS startup_script_session bash -c '$CODE_DIR/setup_scripts/launch.sh; exec bash'
