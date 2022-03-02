import os
import pathlib
from typing import Dict, List, Optional

import hydra
import torch.cuda
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.tuner.tuning import Tuner
from wandb.util import generate_id

from tali.base import utils
from tali.trainer import CustomTrainer

log = utils.get_logger(__name__)

from tali.utils.storage import (
    google_storage_rsync_gs_to_local,
)

torch.cuda.is_available()


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

        if not pathlib.Path(f"{config.current_experiment_dir}").exists():
            os.makedirs(f"{config.current_experiment_dir}", exist_ok=True)

        if "gs_file_monitor" in config.callbacks:
            google_storage_rsync_gs_to_local(
                bucket_name=config.callbacks.gs_file_monitor.bucket_name,
                experiments_root_dir=config.callbacks.gs_file_monitor.experiments_root_dir,
                experiment_name=config.callbacks.gs_file_monitor.experiment_name,
                exclude_list=config.callbacks.gs_file_monitor.exclude_list,
                options_list=config.callbacks.gs_file_monitor.options_list,
            )

        checkpoint_path = f"{config.current_experiment_dir}/checkpoints/last.ckpt"

        log.info(checkpoint_path)

        if not pathlib.Path(checkpoint_path).exists():
            checkpoint_path = None

    else:

        log.info("Starting from scratch")
        # shutil.rmtree(config.current_experiment_dir)
        if not pathlib.Path(f"{config.current_experiment_dir}").exists():
            os.makedirs(f"{config.current_experiment_dir}", exist_ok=True)

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule, _recursive_=False
    )
    datamodule.setup(stage="fit")

    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model, _recursive_=False)

    dummy_data_dict = iter(datamodule.train_dataloader()).__next__()

    # str_data_descr_dict = {
    #     key: value.shape if isinstance(value, torch.Tensor) else value
    #     for key, value in dummy_data_dict.items()
    # }
    # log.info(f"Data description: {str_data_descr_dict}")
    _ = model.forward(dummy_data_dict)

    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_RUN_ID"] = generate_id()

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: CustomTrainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial",
    )

    if config.trainer.auto_scale_batch_size:
        tuner = Tuner(trainer)
        new_batch_size = tuner.scale_batch_size(
            model,
            datamodule=datamodule,
            mode="power",
            init_val=2 * torch.cuda.device_count(),
        )
        datamodule.batch_size = new_batch_size
        config.datamodule.batch_size = new_batch_size

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    # utils.log_hyperparameters(
    #     config=config,
    #     model=model,
    #     datamodule=datamodule,
    #     trainer=trainer,
    #     callbacks=callbacks,
    #     logger=logger,
    # )

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
        datamodule.setup(stage="test")
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
