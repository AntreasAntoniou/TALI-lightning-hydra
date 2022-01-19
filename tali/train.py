import os
import pathlib
import shutil
from typing import Dict, List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.tuner.tuning import Tuner

from base import utils

log = utils.get_logger(__name__)

from tali.utils.storage import (
    google_storage_rsync_gs_to_local,
    google_storage_rsync_local_to_gs,
    pretty_print_dict,
)


def train_eval(config: DictConfig) -> List[Dict[str, float]]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    checkpoint_path = None

    if config.resume:

        log.info("Continue from existing checkpoint")

        if not pathlib.Path(f"{config.exp_dir}").exists():
            os.makedirs(f"{config.exp_dir}", exist_ok=True)

        google_storage_rsync_gs_to_local(
            bucket_name=config.callbacks.gs_file_monitor.bucket_name,
            experiments_root_dir=config.callbacks.gs_file_monitor.experiments_root_dir,
            experiment_name=config.callbacks.gs_file_monitor.experiment_name,
            exclude_list=config.callbacks.gs_file_monitor.exclude_list,
            options_list=config.callbacks.gs_file_monitor.options_list,
        )

        checkpoint_path = f"{config.exp_dir}/checkpoints/last.ckpt"

        log.info(checkpoint_path)

        if not pathlib.Path(checkpoint_path).exists():
            checkpoint_path = None

    else:

        log.info("Starting from scratch")
        # shutil.rmtree(config.exp_dir)
        if not pathlib.Path(f"{config.exp_dir}").exists():
            os.makedirs(f"{config.exp_dir}", exist_ok=True)

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule, _recursive_=False
    )
    datamodule.setup(stage="fit")

    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model, _recursive_=False)

    model.forward(iter(datamodule.train_dataloader()).__next__())

    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial",
        _recursive_=True,
    )

    if config.trainer.auto_scale_batch_size:
        tuner = Tuner(trainer)
        new_batch_size = tuner.scale_batch_size(model, datamodule=datamodule,
                                                mode='binsearch')
        datamodule.batch_size = new_batch_size
        config.datamodule.batch_size = new_batch_size

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )


    if config.mode.fit:
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)

    # # Get metric score for hyperparameter optimization
    # optimized_metric = config.get("optimized_metric")
    # if optimized_metric and optimized_metric not in trainer.callback_metrics:
    #     raise Exception(
    #         "Metric for hyperparameter optimization not found! "
    #         "Make sure the `optimized_metric` in `hparams_search` config is correct!"
    #     )
    # score = trainer.callback_metrics.get(optimized_metric)

    if config.mode.test and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=trainer.checkpoint_callback.best_model_path,
        )

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    # return score


def multi_train_eval(config: DictConfig):
    import ray

    ray.init(address="auto")
    dataset_options = ["base-tali"]  # , "milli/tali", "hecta/tali"]
    system_options = [
        # "milli_modus_prime_resnet50",
        # "centi_modus_prime_resnet50",
        # "deci_modus_prime_resnet50",
        "base-deci-hybrid_modus_prime_resnet50",
        "base_modus_prime_resnet50",
        # "milli_modus_prime_vi-transformer16",
        # "centi_modus_prime_vi-transformer16",
        # "deci_modus_prime_vi-transformer16",
        "base_modus_prime_vi-transformer16",
    ]

    @ray.remote(
        num_cpus=config.num_workers * config.num_cpus_per_worker,
        num_gpus=config.num_workers * config.num_gpus_per_worker,
    )
    def remote_train_eval(config: DictConfig) -> List[Dict[str, float]]:
        return train_eval(config)

    configs = {}
    for use_image_modality in [True]:
        for use_audio_modality in [False, True]:
            for use_video_modality in [False, True]:
                for use_text_modality in [False, True]:
                    for dataset_option in dataset_options:
                        for system_option in system_options:
                            if any(
                                [
                                    use_text_modality,
                                    use_audio_modality,
                                    use_video_modality,
                                ]
                            ):
                                overrides = [
                                    f"datamodule={dataset_option}",
                                    f"model={system_option}",
                                    f"datamodule.config."
                                    f"modality_config.image={use_image_modality}",
                                    f"datamodule.config."
                                    f"modality_config.audio={use_audio_modality}",
                                    f"datamodule.config."
                                    f"modality_config.text={use_text_modality}",
                                    f"datamodule.config."
                                    f"modality_config.video={use_video_modality}",
                                    f"datamodule.config."
                                    f"modality_config.name="
                                    f"video-{use_video_modality}"
                                    f"-audio-{use_audio_modality}"
                                    f"-text-{use_text_modality}"
                                    f"-image-{use_image_modality}".lower(),
                                ]
                                run_cfg = hydra.compose(
                                    config_name="ray_config", overrides=overrides
                                )

                                configs[run_cfg.name] = DictConfig(run_cfg)
    log.info(
        f"Number of trials -------- {len(configs)}, "
        f"{[name for name in configs.keys()]}"
    )

    experiment_objects = []
    for key, value in configs.items():
        log.info(f"Running {key}")
        # log.debug(f'{pretty_print_dict(dict(value))}')

        object_experiment = remote_train_eval.remote(config=value)
        # object_experiment
        experiment_objects.append(object_experiment)

    results = ray.get(experiment_objects)

    log.info(f"{results}")
