# python run.py hydra.verbose=True trainer=default \
# resume=True batch_size=48 trainer.gpus=1 model=base_modus_prime_resnet50
#
# #######################################################################################
#
# datamodule.config.modality_config.image=True
# datamodule.config.modality_config.text=False
# datamodule.config.modality_config.video=False
# datamodule.config.modality_config.audio=True
import numpy as np

def main():
    # batch_size and gpus should be set by model
    configs = {}

    dataset_names = ["base", "milli"]  # , "milli/tali", hecta/tali"]
    system_configs = [
        dict(model_name="centi_modus_prime_resnet50", batch_size=64, num_gpus=-1),
        dict(model_name="base_modus_prime_resnet50", batch_size=64, num_gpus=-1),
        dict(model_name="centi_modus_prime_vi-transformer16", batch_size=64, num_gpus=-1),
        dict(model_name="base_modus_prime_vi-transformer16", batch_size=64, num_gpus=-1),
    ]
    exp_list = []
    for use_image_modality in [True]:
        for use_audio_modality in [False, True]:
            for use_video_modality in [False, True]:
                for use_text_modality in [False, True]:
                    for system_config in system_configs:
                        for dataset_name in dataset_names:
                            if any(
                                [
                                    use_text_modality,
                                    use_audio_modality,
                                    use_video_modality,
                                ]
                            ):
                                batch_size = system_config["batch_size"]
                                num_gpus = system_config["num_gpus"]
                                model_name = system_config["model_name"]

                                if model_name in [
                                    "base_modus_prime_resnet50",
                                    "base_modus_prime_vi-transformer16",
                                ]:
                                    score = np.sum(np.array([use_text_modality,
                                                             use_audio_modality]).astype(
                                        np.int32))

                                    num_gpus = 8 if use_video_modality else 2 * score

                                elif model_name in [
                                    "centi_modus_prime_resnet50",
                                    "centi_modus_prime_vi-transformer16",
                                ]:
                                    score = np.sum(np.array([use_text_modality,
                                                             use_audio_modality]).astype(
                                        np.int32))

                                    num_gpus = 2 if use_video_modality else 1
                                else:
                                    raise NotImplementedError(
                                        f"Given config does not fall into "
                                        f"the expected patterns "
                                        f"dataset_name: {dataset_name} "
                                        f"system_config: {system_config} "
                                        f"use_audio_modality: {use_audio_modality} "
                                        f"use_image_modality: {use_image_modality} "
                                        f"use_video_modality: {use_video_modality} "
                                        f"use_text_modality: {use_text_modality}")

                                template_command = (
                                    f"fuser -k /dev/nvidia*; \\\n"
                                    f"python $CODE_DIR/run.py \\\n"
                                    f"hydra.verbose=True \\\n"
                                    f"trainer=default \\\n"
                                    f"resume=True \\\n"
                                    f"batch_size={batch_size} \\\n"
                                    f"wandb_project_name=TALI-gcp-sweep-1 \\\n"
                                    f"trainer.gpus={num_gpus} \\\n"
                                    f"trainer.auto_scale_batch_size=False \\\n"
                                    f"datamodule.config.rescan_paths=True \\\n"
                                    f"datamodule.prefetch_factor=3 \\\n"
                                    f"datamodule.num_workers={int(num_gpus * 12)} \\\n"
                                    f"model={model_name} \\\n"
                                    f"datamodule.config.training_set_size_identifier={dataset_name} \\\n"
                                    f"datamodule.config.modality_config.image={use_image_modality} \\\n"
                                    f"datamodule.config.modality_config.text={use_text_modality} \\\n"
                                    f"datamodule.config.modality_config.audio={use_audio_modality} \\\n"
                                    f"datamodule.config.modality_config.video={use_video_modality}\n\n"
                                )
                                exp_list.append(template_command)

    with open("bash/experiment_commands.sh", mode="w+") as file_writer:
        file_writer.writelines(exp_list)


if __name__ == "__main__":
    main()
