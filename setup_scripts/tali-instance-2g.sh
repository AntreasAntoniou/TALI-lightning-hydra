#!/bin/bash
gcloud compute instances create tali-base-0 \
--project=tali-multi-modal \
--zone=us-central1-f \
--machine-type=a2-highgpu-2g \
--network-interface=network-tier=PREMIUM,subnet=default \
--no-restart-on-failure \
--maintenance-policy=TERMINATE \
--preemptible \
--service-account=829809103946-compute@developer.gserviceaccount.com \
--scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
--accelerator=count=2,type=nvidia-tesla-a100 \
--create-disk=auto-delete=yes,boot=yes,device-name=tali-base-0,image=projects/tali-multi-modal/global/images/tali-base-gpu-image-v-2-0,mode=rw,size=200,type=projects/tali-multi-modal/zones/us-central1-f/diskTypes/pd-standard \
--create-disk=auto-delete=yes,device-name=dataset-disk-0,image=projects/tali-multi-modal/global/images/tali-dataset-v2-6-us-central1-full-compact,mode=rw,name=dataset-disk-0,size=3500,type=projects/tali-multi-modal/zones/us-central1-f/diskTypes/pd-ssd --no-shielded-secure-boot \
--shielded-vtpm \
--shielded-integrity-monitoring \
--reservation-affinity=any