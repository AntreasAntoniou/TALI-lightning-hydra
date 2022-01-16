import logging
from typing import Any, Dict, List, Optional, Union

import hydra.utils
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from tali.config_repository import (
    AutoAveragerConfig,
    AutoCLIPResNetConfig,
    AutoCLIPTextTransformerConfig,
    AutoCLIPVisionTransformerConfig,
    AutoConv1DTransformersConfig,
    AutoVideoTransformersConfig,
    GoogleStorageConfig,
)
from tali.utils.metric_tracking import compute_accuracy
from tali.utils.storage import google_storage_rsync_local_to_gs


def contrastive_logits_labels(logits):
    labels = torch.arange(len(logits), device=logits.device)
    return logits, labels


class CrossModalMatchingNetwork(LightningModule):
    def __init__(
        self,
        embedding_output_features: int,
        modality_embeddings: nn.ModuleDict,
        logit_scale: float = 1.0,
        sub_batch_size_dict: Optional[Dict[str, int]] = None,
    ):
        super(CrossModalMatchingNetwork, self).__init__()

        self.embed_dim = embedding_output_features
        self.modality_embeddings = modality_embeddings
        self.sub_batch_size_dict = sub_batch_size_dict

        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(logit_scale), requires_grad=True
        )
        self.is_built = False

    def build(
        self,
        batch_shape,
    ):

        logging.info(f"{batch_shape}")

        for modality_key, modality_shape in batch_shape.items():
            if modality_shape is not None:
                modality_shape = list(modality_shape)
                modality_shape[0] = 1
                if (
                    modality_key == "video"
                    and "image" in batch_shape
                    and batch_shape["image"] is not None
                ):
                    self.modality_embeddings["video"].connect_image_embedding(
                        self.modality_embeddings["image"]
                    )

                self.modality_embeddings[modality_key].build(modality_shape)
                self._check_modality_embedding_shape(modality_shape, modality_key)

        logging.info(
            f"built {self.__class__.__name__} with output shape {self.embed_dim}",
        )

        self.is_built = True

    def _check_modality_embedding_shape(self, input_shape, embedding_name):
        input_dummy = torch.zeros(size=input_shape)

        embeddings, _ = self.modality_embeddings[embedding_name].forward(input_dummy)

        assert embeddings.shape[1] == self.embed_dim

    def _get_normalized_features(self, inputs, embedding_name):

        if inputs is not None:
            inputs, _ = self.modality_embeddings[embedding_name](inputs)
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
            return inputs
        else:
            return None

    def _compute_cross_modal_cosine_similarities(self, embedding_dict):
        logit_scale = self.logit_scale.exp()
        logit_dict = {}

        for source_key, source_value in embedding_dict.items():
            for target_key, target_value in embedding_dict.items():
                if (
                    source_key != target_key
                    and source_value is not None
                    and target_value is not None
                ):
                    if f"{target_key}_to_{source_key}_similarity" in logit_dict:
                        logit_dict[
                            f"{source_key}_to_{target_key}_similarity"
                        ] = logit_dict[f"{target_key}_to_{source_key}_similarity"].t()
                    else:
                        logit_dict[f"{source_key}_to_{target_key}_similarity"] = (
                            torch.matmul(source_value, target_value.t()) * logit_scale
                        )

        return logit_dict

    def forward(self, batch):

        if not self.is_built:
            self.build(
                {
                    key: value.shape if isinstance(value, torch.Tensor) else None
                    for key, value in batch.items()
                }
            )

        if self.sub_batch_size_dict is not None:
            embedding_dict = {}
            for embedding_name, inputs in batch.items():
                if embedding_name in self.sub_batch_size_dict:
                    sub_batch_size = self.sub_batch_size_dict[embedding_name]
                    sub_batches = inputs.view(
                        (
                            sub_batch_size,
                            -1,
                        )
                        + inputs.shape[1:]
                    )
                    current_embedding_features = []
                    for sub_batch in sub_batches:
                        sub_batch_features_for_current_embedding = (
                            self._get_normalized_features(sub_batch, embedding_name)
                        )
                        current_embedding_features.append(
                            sub_batch_features_for_current_embedding
                        )
                    embedding_dict[embedding_name] = torch.cat(
                        current_embedding_features, dim=0
                    )
                else:
                    embedding_dict[embedding_name] = self._get_normalized_features(
                        inputs, embedding_name
                    )
        else:
            embedding_dict = {
                embedding_name: self._get_normalized_features(inputs, embedding_name)
                for embedding_name, inputs in batch.items()
            }

        return (
            embedding_dict,
            self._compute_cross_modal_cosine_similarities(embedding_dict),
        )


class ModusPrime(LightningModule):
    def __init__(
        self,
        name: str,
        google_storage_config: GoogleStorageConfig,
        image_embedding_config: Union[
            AutoCLIPVisionTransformerConfig, AutoCLIPResNetConfig
        ],
        audio_embedding_config: Union[AutoConv1DTransformersConfig],
        text_embedding_config: Union[AutoCLIPTextTransformerConfig],
        video_embedding_config: Union[AutoAveragerConfig, AutoVideoTransformersConfig],
        optimizer_config: None,
        lr_scheduler_config: None,
        sub_batch_size_dict: Optional[Dict[str, int]] = None,
        embedding_output_features: int = 512,
        logit_scale: float = 1.0,
    ):
        super(ModusPrime, self).__init__()
        self.google_storage_config = google_storage_config
        self.system = CrossModalMatchingNetwork(
            embedding_output_features=embedding_output_features,
            modality_embeddings=nn.ModuleDict(
                dict(
                    image=hydra.utils.instantiate(image_embedding_config),
                    audio=hydra.utils.instantiate(audio_embedding_config),
                    text=hydra.utils.instantiate(text_embedding_config),
                    video=hydra.utils.instantiate(video_embedding_config),
                )
            ),
            logit_scale=logit_scale,
            sub_batch_size_dict=sub_batch_size_dict,
        )
        self.sub_batch_size_dict = sub_batch_size_dict
        self.is_built = False
        self.metrics_to_track = {
            "cross_entropy": lambda x, y: F.cross_entropy(input=x, target=y),
            "accuracy": lambda x, y: compute_accuracy(x, y),
        }
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.save_hyperparameters(logger=False)

    def build(self, batch):
        _, _ = self.system.forward(batch=batch)

        self.is_built = True
        self.save_hyperparameters(logger=False)

    def forward(self, batch):

        if not self.system.is_built:
            self.system.build(
                batch_shape={
                    key: value.shape if isinstance(value, torch.Tensor) else None
                    for key, value in batch.items()
                }
            )

        outputs, cosine_similarities = self.system.forward(
            batch=batch,
        )

        logits, _ = contrastive_logits_labels(
            torch.stack(list(cosine_similarities.values()))
        )

        return logits

    def collect_metrics(self, logits, targets, phase_name):
        metrics_computed = {}
        for metric_key, metric_function in self.metrics_to_track.items():
            for measurement_key, measurement_value, target_value in zip(
                logits.keys(), logits.values(), targets
            ):
                # logging.info(f"{measurement_value'].shape} {target_value.shape}")
                metrics_computed[
                    f"{phase_name}/{metric_key}_{measurement_key}"
                ] = metric_function(measurement_value.detach(), target_value.detach())
                self.log(
                    name=f"{phase_name}/{metric_key}_{measurement_key}",
                    value=metrics_computed[
                        f"{phase_name}/" f"{metric_key}_{measurement_key}"
                    ],
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    on_epoch=False,
                )

            metrics_computed[f"{phase_name}/overall_{metric_key}"] = metric_function(
                torch.stack(list(logits.values())).detach(), targets.detach()
            )

            self.log(
                name=f"{phase_name}/overall_{metric_key}",
                value=metrics_computed[f"{phase_name}/overall_{metric_key}"],
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=False,
            )
            return metrics_computed

    def training_step(self, batch, batch_idx):
        # logging.info(f'{[(key, value.shape) for key, value in batch.items()]}')
        outputs, cosine_similarities = self.system.forward(
            batch,
        )

        targets = torch.stack(
            [
                contrastive_logits_labels(modality_similarities)[1]
                for modality_similarities in cosine_similarities.values()
            ],
            dim=0,
        )

        logits = torch.stack(list(cosine_similarities.values()), dim=0)

        metrics_computed = self.collect_metrics(
            logits=cosine_similarities, targets=targets, phase_name="training"
        )
        metrics_computed["loss"] = F.cross_entropy(input=logits, target=targets)

        return metrics_computed

    def training_epoch_end(self, outputs: List[Any]):

        for metric_key in outputs[0].keys():
            self.log(
                name=f"{metric_key}-mean",
                value=torch.stack([x[metric_key] for x in outputs]).mean(),
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
            )

            self.log(
                name=f"{metric_key}-std",
                value=torch.stack([x[metric_key] for x in outputs]).std(dim=0),
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
            )

    def validation_step(self, batch, batch_idx):
        # logging.info(f'{[(key, value.shape) for key, value in batch.items()]}')

        outputs, cosine_similarities = self.system.forward(batch)

        targets = torch.stack(
            [
                contrastive_logits_labels(modality_similarities)[1]
                for modality_similarities in cosine_similarities.values()
            ],
            dim=0,
        )

        return self.collect_metrics(
            logits=cosine_similarities, targets=targets, phase_name="validation"
        )

    def validation_epoch_end(self, outputs: List[Any]):

        for metric_key in outputs[0].keys():
            self.log(
                name=f"{metric_key}-mean",
                value=torch.stack([x[metric_key] for x in outputs]).mean(),
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
            )

            self.log(
                name=f"{metric_key}-std",
                value=torch.stack([x[metric_key] for x in outputs]).std(dim=0),
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
            )

    def test_step(self, batch, batch_idx):

        outputs, cosine_similarities = self.system.forward(batch)

        targets = torch.stack(
            [
                contrastive_logits_labels(modality_similarities)[1]
                for modality_similarities in cosine_similarities.values()
            ],
            dim=0,
        )

        return self.collect_metrics(
            logits=cosine_similarities, targets=targets, phase_name="test"
        )

    def testing_epoch_end(self, outputs: List[Any]):

        for metric_key in outputs[0].keys():
            self.log(
                name=f"{metric_key}-mean",
                value=torch.stack([x[metric_key] for x in outputs]).mean(),
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
            )

            self.log(
                name=f"{metric_key}-std",
                value=torch.stack([x[metric_key] for x in outputs]).std(dim=0),
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
            )

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            config=self.optimizer_config, params=self.parameters()
        )

        lr_scheduler = hydra.utils.instantiate(
            config=self.lr_scheduler_config, optimizer=optimizer
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
