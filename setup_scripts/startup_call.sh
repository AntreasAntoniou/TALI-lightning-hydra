#!/bin/bash
echo "ALL HAIL THE SCRIPT"
source /root/.bashrc
cd /root/TALI-lightning-hydra/
git pull
bash /root/TALI-lightning-hydra/setup_scripts/startup_script.sh
echo "STARTUP SCRIPT DONE, hopefully"

#sudo -u evolvingfungus bash -c 'bash ~/startup_script.sh'