#!/bin/bash
echo "ALL HAIL THE SCRIPT"
source /home/evolvingfungus/.bashrc
cd /home/evolvingfungus/TALI-lightning-hydra/
git pull
bash /home/evolvingfungus/TALI-lightning-hydra/setup_scripts/startup_script.sh
echo "STARTUP SCRIPT DONE, hopefully"