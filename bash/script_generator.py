# python run.py hydra.verbose=True trainer=default \
# resume=True batch_size=48 trainer.gpus=1 model=base_modus_prime_resnet50
#
# #######################################################################################
#
# datamodule.config.modality_config.image=True
# datamodule.config.modality_config.text=False
# datamodule.config.modality_config.video=False
# datamodule.config.modality_config.audio=True


def main():
    # batch_size and gpus should be set by model
    configs = {}

    dataset_options = ["base-tali", "hecta-tali"]  # , "milli/tali", hecta/tali"]
    system_options = [
        dict(model_name="milli_modus_prime_resnet50", batch_size=16, num_gpus=1),
        dict(model_name="centi_modus_prime_resnet50", batch_size=8, num_gpus=1),
        dict(model_name="deci_modus_prime_resnet50", batch_size=10, num_gpus=1),
        # dict(model_name="base-deci-hybrid_modus_prime_resnet50", batch_size=4, num_gpus=1),
        dict(model_name="base_modus_prime_resnet50", batch_size=10, num_gpus=1),
        dict(
            model_name="milli_modus_prime_vi-transformer16", batch_size=16, num_gpus=1
        ),
        dict(model_name="centi_modus_prime_vi-transformer16", batch_size=8, num_gpus=1),
        dict(model_name="deci_modus_prime_vi-transformer16", batch_size=4, num_gpus=1),
        dict(model_name="base_modus_prime_vi-transformer16", batch_size=2, num_gpus=1),
    ]
    exp_list = []
    for use_image_modality in [True]:
        for use_audio_modality in [False, True]:
            for use_video_modality in [False, True]:
                for use_text_modality in [False, True]:
                    for datamodule_name in dataset_options:
                        for model_options in system_options:
                            if any(
                                [
                                    use_text_modality,
                                    use_audio_modality,
                                    use_video_modality,
                                ]
                            ):
                                batch_size = model_options["batch_size"]
                                num_gpus = model_options["num_gpus"]
                                model_name = model_options["model_name"]
                                template_command = (
                                    f"python run.py hydra.verbose=True trainer=default "
                                    f"resume=True batch_size={batch_size} "
                                    f"trainer.gpus={num_gpus} "
                                    f"trainer.auto_scale_batch_size=True "
                                    f"model={model_name} datamodule={datamodule_name} "
                                    f"datamodule.config.modality_config.image={use_image_modality} "
                                    f"datamodule.config.modality_config.text={use_text_modality} "
                                    f"datamodule.config.modality_config.audio={use_audio_modality} "
                                    f"datamodule.config.modality_config.video={use_video_modality}\n\n"
                                )
                                exp_list.append(template_command)

    with open("bash/experiment_commands.sh", mode="w+") as file_writer:
        file_writer.writelines(exp_list)


if __name__ == "__main__":
    main()
