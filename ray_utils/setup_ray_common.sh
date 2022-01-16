#!/bin/bash
  # Setup disk mounts to file-store
#sudo mkdir -p /mnt/disk/filestore
#sudo mount 10.157.245.2:/taligate_filestore /mnt/disk/filestore/

  # Install Ray
yes | pip install ray
export TOKENIZERS_PARALLELISM=false
yes | pip install --upgrade "wandb[launch]"
yes | pip install cryptography
yes | pip install google-api-python-client
yes | pip install ray[tune] --upgrade
yes | pip install git+https://github.com/ray-project/ray_lightning#ray_lightning
yes | pip install hydra-core --upgrade
yes | pip install hydra --upgrade
yes | pip install -e /home/evolvingfungus/current_research_forge/TALI/
yes | pip install -e /home/evolvingfungus/current_research_forge/GATE/