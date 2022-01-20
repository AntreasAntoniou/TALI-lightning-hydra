#!/usr/bin/env python

"""
Builds a convolutional neural network on the fashion mnist data set.

Designed to show wandb integration with pytorch.
"""

import os
import wandb
import torch

hyperparameter_defaults = dict(
    use_image_modality=True,
    use_video_modality=False,
    use_audio_modality=False,
    use_text_modality=True,
    datamodule_name="base-tali",
    model_name="base_modus_prime_resnet50",
)

wandb.init(config=hyperparameter_defaults, project="TALI-gcp-sweep")
config = wandb.config


def main():
    template_command = (
        f"python $CODE_DIR/run.py hydra.verbose=True trainer=default "
        f"resume=True batch_size={torch.cuda.device_count() * 2}"
        f" wandb_project_name=TALI-gcp-sweep-0 "
        f"trainer.gpus=-1 "
        f"trainer.auto_scale_batch_size=True "
        f"model={config.model_name} datamodule={config.datamodule_name} "
        f"datamodule.config.modality_config.image={config.use_image_modality} "
        f"datamodule.config.modality_config.text={config.use_text_modality} "
        f"datamodule.config.modality_config.audio={config.use_audio_modality} "
        f"datamodule.config.modality_config.video={config.use_video_modality}\n\n"
    )

    os.system(template_command)


if __name__ == "__main__":
    main()
