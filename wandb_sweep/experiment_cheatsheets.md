```bash
gcloud beta compute instances create gpu-instance-small-0 \
--project=tali-multi-modal \
--zone=us-central1-f \
--machine-type=a2-highgpu-8g \
--network-interface=network-tier=PREMIUM,subnet=default \
--no-restart-on-failure \
--maintenance-policy=TERMINATE \
--preemptible \
--service-account=tali-multi-modal@tali-multi-modal.iam.gserviceaccount.com \
--scopes=https://www.googleapis.com/auth/cloud-platform \
--accelerator=count=8,type=nvidia-tesla-a100 \
--create-disk=auto-delete=yes,boot=yes,device-name=gpu-instance-large-0,image=projects/tali-multi-modal/global/images/tali-ubuntu-cuda110-pytorch-v-1-3,mode=rw,size=150,type=projects/tali-multi-modal/zones/us-central1-f/diskTypes/pd-standard \
--create-disk=auto-delete=yes,device-name=tali-dataset-disk,image=projects/tali-multi-modal/global/images/tali-v-3-5-high-npy-error-rate,mode=rw,name=disk-6,size=10000,type=projects/tali-multi-modal/zones/us-central1-f/diskTypes/pd-ssd \
--no-shielded-secure-boot \
--shielded-vtpm \
--shielded-integrity-monitoring \
--reservation-affinity=any \
--provisioning-model=SPOT
```

```bash
gcloud beta compute instances create gpu-instance-large-0 \
--project=tali-multi-modal \
--zone=us-central1-f \
--machine-type=a2-highgpu-8g \
--network-interface=network-tier=PREMIUM,subnet=default \
--no-restart-on-failure \
--maintenance-policy=TERMINATE \
--preemptible \
--service-account=tali-multi-modal@tali-multi-modal.iam.gserviceaccount.com \
--scopes=https://www.googleapis.com/auth/cloud-platform \
--accelerator=count=8,type=nvidia-tesla-a100 \
--create-disk=auto-delete=yes,boot=yes,device-name=gpu-instance-large-0,image=projects/tali-multi-modal/global/images/tali-ubuntu-cuda110-pytorch-v-1-3,mode=rw,size=150,type=projects/tali-multi-modal/zones/us-central1-f/diskTypes/pd-standard \
--create-disk=auto-delete=yes,device-name=tali-dataset-disk,image=projects/tali-multi-modal/global/images/tali-v-3-5-high-npy-error-rate,mode=rw,name=disk-6,size=10000,type=projects/tali-multi-modal/zones/us-central1-f/diskTypes/pd-ssd \
--no-shielded-secure-boot \
--shielded-vtpm \
--shielded-integrity-monitoring \
--reservation-affinity=any \
--provisioning-model=SPOT
```


Currently running on gpu-instance-1
```bash
python run.py hydra.verbose=False \
resume=True \
batch_size=800 \
datamodule.num_workers=24 \
trainer.gpus=2 \
model=milli_modus_prime_resnet50 \
datamodule=tali \
datamodule.config.modality_config.image=True \
datamodule.config.modality_config.text=True \
datamodule.config.modality_config.audio=False \
datamodule.config.modality_config.video=False \
datamodule.config.rescan_paths=False \
datamodule.prefetch_factor=2 \
datamodule.config.dataset_size_identifier=base
```

Currently running on gpu-instance-2
```bash
python run.py hydra.verbose=False \
resume=True \
batch_size=32 \
datamodule.num_workers=24 \
trainer.gpus=2 \
model=milli_modus_prime_resnet50 \
datamodule=tali \
datamodule.config.modality_config.image=True \
datamodule.config.modality_config.text=True \
datamodule.config.modality_config.audio=False \
datamodule.config.modality_config.video=True \
datamodule.config.rescan_paths=False \
datamodule.prefetch_factor=2 \
datamodule.config.dataset_size_identifier=base
```