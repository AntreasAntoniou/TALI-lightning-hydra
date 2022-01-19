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

gcloud compute instances create gpu-small-node-1 --project=tali-multi-modal --zone=us-central1-f --machine-type=a2-highgpu-1g --network-interface=network-tier=PREMIUM,subnet=default --no-restart-on-failure --maintenance-policy=TERMINATE --preemptible --service-account=tali-multi-modal@tali-multi-modal.iam.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform --accelerator=count=1,type=nvidia-tesla-a100 --create-disk=auto-delete=yes,boot=yes,device-name=gpu-small-node,image=projects/ml-images/global/images/c0-deeplearning-common-cu113-v20211219-debian-10,mode=rw,size=350,type=projects/tali-multi-modal/zones/us-central1-a/diskTypes/pd-standard --create-disk=auto-delete=yes,device-name=persistent-disk-1,image=projects/tali-multi-modal/global/images/tali-dataset-v2-6-us-central1-full-compact,mode=rw,size=3500,type=projects/tali-multi-modal/zones/us-central1-a/diskTypes/pd-ssd --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --reservation-affinity=any