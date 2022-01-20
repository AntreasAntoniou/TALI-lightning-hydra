#!/bin/bash
source $HOME/.bashrc
cd $CODE_DIR
git pull
bash $CODE_DIR/setup_scripts/startup_script.sh