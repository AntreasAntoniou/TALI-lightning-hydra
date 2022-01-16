from __future__ import print_function

import logging as log
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from clip.model import LayerNorm, ModifiedResNet, Transformer
from einops import rearrange, repeat
from gate.architectures import SqueezeExciteConv1dBNLeakyReLU
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPVisionConfig,
    CLIPVisionModel,
)

from tali.config_repository import (
    AutoAveragerConfig,
    AutoCLIPResNetConfig,
    AutoCLIPTextTransformerConfig,
    AutoCLIPVisionTransformerConfig,
    AutoConv1DTransformersConfig,
    AutoVideoTransformersConfig,
)
from tali.models.auto_builder.base import (
    AvgPoolFlexibleDimension,
    BaseLinearOutputModel,
)
from tali.models.auto_builder.densenet import DenseNetEmbedding


class FCCNetwork(nn.Module):
    def __init__(
        self,
        num_hidden_features,
        embedding_output_features,
        num_hidden_layers,
        activation_fn=F.leaky_relu,
    ):
        super(FCCNetwork, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.is_layer_built = False
        self.num_hidden_features = num_hidden_features
        self.embedding_output_features = embedding_output_features
        self.num_hidden_layers = num_hidden_layers
        self.activation_fn = activation_fn

    def build(self, input_shape):
        out = torch.zeros(input_shape)

        for i in range(self.num_hidden_layers):
            self.layer_dict[f"fcc_layer_{i}"] = nn.Linear(
                in_features=out.shape[1],
                out_features=self.num_hidden_features,
                bias=True,
            )
            out = self.activation_fn(self.layer_dict[f"fcc_layer_{i}"].forward(out))

        self.layer_dict["fcc_layer_output"] = nn.Linear(
            in_features=out.shape[1],
            out_features=self.embedding_output_features,
            bias=True,
        )

        out = self.layer_dict["fcc_layer_output"].forward(out)

        self.is_layer_built = True

        log.info(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

    def forward(self, x):
        if not self.is_layer_built:
            self.build(input_shape=x.shape)
            self.to(x.device)

        out = x

        self.to(out.device)

        for i in range(self.num_hidden_layers):
            out = self.activation_fn(self.layer_dict[f"fcc_layer_{i}"].forward(out))

        out = self.layer_dict["fcc_layer_output"].forward(out)

        return out


class ChooseSpecificTimeStepFromVector(nn.Module):
    def __init__(self, time_step_to_choose):
        super(ChooseSpecificTimeStepFromVector, self).__init__()
        self.time_step_to_choose = time_step_to_choose

    def forward(self, x):

        return x, x[:, self.time_step_to_choose, :]


class Conv2DTransformer(nn.Module):
    def __init__(
        self,
        grid_patch_size: int,
        transformer_num_filters: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        transformer_dim_feedforward: int,
        stem_conv_bias: False,
    ):
        super(Conv2DTransformer, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.grid_patch_size = grid_patch_size
        self.transformer_num_filters = transformer_num_filters
        self.transformer_num_layers = transformer_num_layers
        self.transformer_num_heads = transformer_num_heads
        self.transformer_dim_feedforward = transformer_dim_feedforward
        self.stem_conv_bias = stem_conv_bias

        self.is_built = False

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape)

        ratio = (dummy_x.shape[2]) / self.grid_patch_size

        if not ratio.is_integer():
            ceiling = int(np.ceil(ratio))
            new_h = ceiling * self.grid_patch_size
            new_w = ceiling * self.grid_patch_size
            dummy_x = F.interpolate(
                dummy_x,
                size=(new_h, new_w),
            )

        out = dummy_x

        b, c, h, w = out.shape

        out = rearrange(
            out,
            "b f (h h1) (w w1) -> (b h w) (h1 w1 f)",
            h1=self.grid_patch_size,
            w1=self.grid_patch_size,
        )
        log.info(f"{out.shape}")
        num_patches = out.shape[0] / dummy_x.shape[0]

        self.layer_dict["stem_linear"] = nn.Linear(
            in_features=out.shape[1],
            out_features=int(self.transformer_num_filters / 2),
            bias=False,
        )

        out = self.layer_dict["stem_linear"].forward(out)
        # b, c, h, w
        log.info(f"{out.shape}")

        self.layer_dict["stem_layer_normalization"] = nn.LayerNorm(out.shape[1])

        out = self.layer_dict["stem_layer_normalization"].forward(out)
        log.info(f"{out.shape}")

        out = rearrange(out, "(b s) (f) -> b s f", s=int(num_patches))
        log.info(f"{out.shape}")

        self.enumerate_patches_idx = (
            torch.arange(start=0, end=num_patches) / num_patches
        )

        position_inputs = repeat(
            self.enumerate_patches_idx, "p -> b p", b=dummy_x.shape[0]
        )
        log.info(f"{position_inputs.shape}")

        position_inputs = rearrange(position_inputs, "b (p d) -> (b p) d", d=1)
        log.info(f"{position_inputs.shape}")

        self.layer_dict["positional_embedding_generator_network"] = FCCNetwork(
            num_hidden_features=64,
            num_hidden_layers=2,
            embedding_output_features=out.shape[2],
        )

        positional_embeddings = self.layer_dict[
            "positional_embedding_generator_network"
        ].forward(position_inputs)
        log.info(f"{positional_embeddings.shape}")

        positional_embeddings = rearrange(
            positional_embeddings,
            "(b p) d -> b p d",
            b=dummy_x.shape[0],
            d=out.shape[2],
        )
        log.info(f"{positional_embeddings.shape} {out.shape}")
        out = torch.cat([out, positional_embeddings], dim=2)

        self.layer_dict["transformer_encoder_layer"] = nn.TransformerEncoderLayer(
            d_model=self.transformer_num_filters,
            dim_feedforward=self.transformer_dim_feedforward,
            nhead=self.transformer_num_heads,
            activation=F.gelu,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.layer_dict["transformer_encoder"] = nn.TransformerEncoder(
            encoder_layer=self.layer_dict["transformer_encoder_layer"],
            num_layers=self.transformer_num_layers,
        )

        out = self.layer_dict["transformer_encoder"].forward(out)

        self.is_built = True
        log.info(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape} {positional_embeddings.shape}"
        )

    def forward(self, x):
        if not self.is_built:
            self.build(input_shape=x.shape)

        self.to(x.device)

        ratio = (x.shape[2]) / self.grid_patch_size

        if not ratio.is_integer():
            ceiling = int(np.ceil(ratio))
            new_h = ceiling * self.grid_patch_size
            new_w = ceiling * self.grid_patch_size
            x = F.interpolate(
                x,
                size=(new_h, new_w),
            )

        out = x

        b, c, h, w = out.shape

        out = rearrange(
            out,
            "b f (h h1) (w w1) -> (b h w) (h1 w1 f)",
            h1=self.grid_patch_size,
            w1=self.grid_patch_size,
        )

        num_patches = out.shape[0] / x.shape[0]

        out = self.layer_dict["stem_linear"].forward(out)
        # b, c, h, w

        out = self.layer_dict["stem_layer_normalization"].forward(out)

        out = rearrange(out, "(b s) (f) -> b s f", s=int(num_patches))

        position_inputs = repeat(
            self.enumerate_patches_idx, "p -> b p", b=x.shape[0]
        ).to(x.device)

        position_inputs = rearrange(position_inputs, "b (p d) -> (b p) d", d=1)

        positional_embeddings = self.layer_dict[
            "positional_embedding_generator_network"
        ].forward(position_inputs)

        positional_embeddings = rearrange(
            positional_embeddings,
            "(b p) d -> b p d",
            b=x.shape[0],
            d=out.shape[2],
        )
        out = torch.cat([out, positional_embeddings], dim=2)

        out = self.layer_dict["transformer_encoder"].forward(out)

        return out


class Conv1DTransformer(nn.Module):
    def __init__(
        self,
        grid_patch_size: int,
        densenet_num_filters: int,
        densenet_num_stages: int,
        densenet_num_blocks: int,
        densenet_dilated: bool,
        densenet_adaptive_pool_output_features: int,
        transformer_num_filters: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        transformer_dim_feedforward: int,
        stem_conv_bias: False,
    ):
        super(Conv1DTransformer, self).__init__()
        self.grid_patch_size = grid_patch_size
        self.densenet_num_filters = densenet_num_filters
        self.densenet_num_stages = densenet_num_stages
        self.densenet_num_blocks = densenet_num_blocks
        self.densenet_dilated = densenet_dilated
        self.densenet_adaptive_pool_output_features = (
            densenet_adaptive_pool_output_features
        )
        self.transformer_num_filters = transformer_num_filters
        self.transformer_num_layers = transformer_num_layers
        self.transformer_num_heads = transformer_num_heads
        self.transformer_dim_feedforward = transformer_dim_feedforward
        self.stem_conv_bias = stem_conv_bias
        self.positional_embeddings = nn.Parameter(
            data=torch.randn((int(self.grid_patch_size), self.transformer_num_filters)),
            requires_grad=True,
        )
        self.is_built = False

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape)

        out = dummy_x

        self.layer_dict = nn.ModuleDict()

        self.layer_dict["conv1d_densenet"] = DenseNetEmbedding(
            num_filters=self.densenet_num_filters,
            num_stages=self.densenet_num_stages,
            num_blocks=self.densenet_num_blocks,
            dilated=self.densenet_dilated,
            processing_block_type=SqueezeExciteConv1dBNLeakyReLU,
        )

        out = self.layer_dict["conv1d_densenet"].forward(out)

        log.info(f"conv1d_densenet output shape {out.shape}")

        self.layer_dict["conv_feature_pooling"] = nn.Conv1d(
            in_channels=out.shape[1],
            out_channels=self.transformer_num_filters,
            kernel_size=1,
        )

        out = self.layer_dict["conv_feature_pooling"].forward(out)
        log.info(f"conv_feature_pooling output shape {out.shape}")

        self.layer_dict["conv_spatial_pool"] = nn.AdaptiveAvgPool1d(
            output_size=self.grid_patch_size
        )

        out = self.layer_dict["conv_spatial_pool"].forward(out)
        log.info(f"conv_spatial_pool output shape {out.shape}")

        out = rearrange(out, "b f h -> (b h) f")

        self.layer_dict["stem_layer_normalization"] = nn.LayerNorm(out.shape[1])

        out = self.layer_dict["stem_layer_normalization"].forward(out)
        log.info(f"stem_layer_normalization output shape {out.shape}")

        out = rearrange(out, "(b s) (f) -> b s f", b=dummy_x.shape[0])
        log.info(f"rearrange output shape {out.shape}")

        self.positional_embeddings = self.positional_embeddings.to(out.device)

        positional_embeddings = repeat(
            self.positional_embeddings, "p f -> b p f", b=dummy_x.shape[0]
        )

        out = out + positional_embeddings
        log.info(f"out + positional_embeddings output shape {out.shape}")

        self.layer_dict["transformer_encoder_layer"] = nn.TransformerEncoderLayer(
            d_model=self.transformer_num_filters,
            dim_feedforward=self.transformer_dim_feedforward,
            nhead=self.transformer_num_heads,
            activation=F.gelu,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.layer_dict["transformer_encoder"] = nn.TransformerEncoder(
            encoder_layer=self.layer_dict["transformer_encoder_layer"],
            num_layers=self.transformer_num_layers,
        )

        out = self.layer_dict["transformer_encoder"].forward(out)
        log.info(f"transformer_encoder output shape {out.shape}")

        self.is_built = True
        log.info(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

    def forward(self, x):
        x = x.type(torch.float32)
        if not self.is_built:
            self.build(input_shape=x.shape)

        self.to(x.device)

        out = x

        out = self.layer_dict["conv1d_densenet"].forward(out)

        out = self.layer_dict["conv_feature_pooling"].forward(out)

        out = self.layer_dict["conv_spatial_pool"].forward(out)

        out = rearrange(out, "b f h -> (b h) f")

        out = self.layer_dict["stem_layer_normalization"].forward(out)

        out = rearrange(out, "(b s) (f) -> b s f", b=x.shape[0])

        self.positional_embeddings = self.positional_embeddings.to(out.device)

        positional_embeddings = repeat(
            self.positional_embeddings, "p f -> b p f", b=x.shape[0]
        )

        out = out + positional_embeddings

        out = self.layer_dict["transformer_encoder"].forward(out)

        return out


class TexTransformer(nn.Module):
    def __init__(
        self,
        transformer_num_filters: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        transformer_dim_feedforward: int,
        vocab_size: int,
        context_length: int,
    ):
        super(TexTransformer, self).__init__()
        self.transformer_num_filters = transformer_num_filters
        self.transformer_num_layers = transformer_num_layers
        self.transformer_num_heads = transformer_num_heads
        self.transformer_dim_feedforward = transformer_dim_feedforward
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.layer_dict = nn.ModuleDict()
        self.layer_params = nn.ParameterDict()
        self.is_built = False

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape).long()

        out = dummy_x

        self.layer_dict["token_embedding"] = nn.Embedding(
            self.vocab_size, self.transformer_num_filters
        )

        out = self.layer_dict["token_embedding"].forward(out)

        # b, l, c

        self.enumerate_patches_idx = (
            torch.arange(start=0, end=out.shape[1]) / out.shape[1]
        )

        position_inputs = repeat(
            self.enumerate_patches_idx, "p -> b p", b=dummy_x.shape[0]
        )

        position_inputs = rearrange(position_inputs, "b (p d) -> (b p) d", d=1)

        self.layer_dict["positional_embedding_generator_network"] = FCCNetwork(
            num_hidden_features=64,
            num_hidden_layers=2,
            embedding_output_features=out.shape[2],
        )

        positional_embeddings = self.layer_dict[
            "positional_embedding_generator_network"
        ].forward(position_inputs)

        positional_embeddings = rearrange(
            positional_embeddings,
            "(b p) d -> b p d",
            b=dummy_x.shape[0],
            d=out.shape[2],
        )
        out = torch.cat([out, positional_embeddings], dim=2)

        self.layer_dict["transformer_encoder_layer"] = nn.TransformerEncoderLayer(
            d_model=self.transformer_num_filters * 2,
            dim_feedforward=self.transformer_dim_feedforward,
            nhead=self.transformer_num_heads,
            batch_first=True,
        )
        self.layer_dict["transformer_encoder"] = nn.TransformerEncoder(
            encoder_layer=self.layer_dict["transformer_encoder_layer"],
            num_layers=self.transformer_num_layers,
        )

        out = self.layer_dict["transformer_encoder"].forward(out)

        self.is_built = True

        log.info(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

    def forward(self, x):
        if not self.is_built:
            self.build(input_shape=x.shape)

        self.to(x.device)

        out = x.long()

        out = self.layer_dict["token_embedding"].forward(out)

        position_inputs = repeat(
            self.enumerate_patches_idx, "p -> b p", b=x.shape[0]
        ).to(x.device)

        position_inputs = rearrange(position_inputs, "b (p d) -> (b p) d", d=1)

        positional_embeddings = F.leaky_relu(
            self.layer_dict["positional_embedding_generator_network"].forward(
                position_inputs
            )
        )

        positional_embeddings = rearrange(
            positional_embeddings,
            "(b p) d -> b p d",
            b=x.shape[0],
            d=out.shape[2],
        )
        out = torch.cat([out, positional_embeddings], dim=2)

        out = self.layer_dict["transformer_encoder"].forward(out)

        return out


class VideoTransformer(nn.Module):
    def __init__(
        self,
        transformer_num_filters: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        transformer_dim_feedforward: int,
        image_embedding: nn.Identity(),
    ):
        super(VideoTransformer, self).__init__()
        self.transformer_num_filters = transformer_num_filters
        self.transformer_num_layers = transformer_num_layers
        self.transformer_num_heads = transformer_num_heads
        self.transformer_dim_feedforward = transformer_dim_feedforward

        self.image_embedding = image_embedding
        self.layer_dict = nn.ModuleDict()
        self.layer_params = nn.ParameterDict()
        self.is_built = False

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape)

        out = dummy_x.view(-1, dummy_x.shape[-3], dummy_x.shape[-2], dummy_x.shape[-1])

        _, out = self.image_embedding(out)

        if self.image_embedding is not None:
            _, out = self.image_embedding(out)

        out = out.view(dummy_x.shape[0], dummy_x.shape[1], -1)

        self.enumerate_patches_idx = (
            torch.arange(start=0, end=out.shape[1]) / out.shape[1]
        )

        self.positional_embeddings = torch.empty(size=(out.shape[1], out.shape[2])).to(
            out.device
        )

        nn.init.normal(self.positional_embeddings)

        out = out + repeat(self.positional_embeddings, "p f -> b p f", b=out.shape[0])

        self.layer_dict["transformer_encoder_layer"] = nn.TransformerEncoderLayer(
            d_model=self.transformer_num_filters,
            dim_feedforward=self.transformer_dim_feedforward,
            nhead=self.transformer_num_heads,
            batch_first=True,
        )
        self.layer_dict["transformer_encoder"] = nn.TransformerEncoder(
            encoder_layer=self.layer_dict["transformer_encoder_layer"],
            num_layers=self.transformer_num_layers,
        )

        out = self.layer_dict["transformer_encoder"].forward(out)

        self.is_built = True

        log.info(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

    def forward(self, x):

        if not self.is_built:
            self.build(input_shape=x.shape)

        self.to(x.device)

        out = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])

        if self.image_embedding is not None:
            _, out = self.image_embedding(out)

        out = out.view(x.shape[0], x.shape[1], -1)

        self.positional_embeddings = self.positional_embeddings.to(out.device)

        out = out + repeat(self.positional_embeddings, "p f -> b p f", b=out.shape[0])

        out = self.layer_dict["transformer_encoder"].forward(out)

        return out

    def connect_image_embedding(self, image_embedding):
        self.image_embedding = image_embedding


class Averager(nn.Module):
    def __init__(self, image_embedding: nn.Identity()):
        super(Averager, self).__init__()

        self.image_embedding = image_embedding
        self.is_built = False

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape)

        out = dummy_x.view(-1, dummy_x.shape[-3], dummy_x.shape[-2], dummy_x.shape[-1])

        if self.image_embedding is not None:
            _, out = self.image_embedding(out)

        out = out.view(dummy_x.shape[0], dummy_x.shape[1], -1)

        self.is_built = True

        log.info(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

    def forward(self, x):

        if not self.is_built:
            self.build(input_shape=x.shape)

        self.to(x.device)

        out = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])

        if self.image_embedding is not None:
            _, out = self.image_embedding(out)

        out = out.view(x.shape[0], x.shape[1], -1)

        return out

    def connect_image_embedding(self, image_embedding):
        self.image_embedding = image_embedding


class VisionTransformer(nn.Module):
    def __init__(
        self, patch_size: int, width: int, layers: int, heads: int, output_dim: int
    ):
        super().__init__()
        self.output_dim = output_dim
        self.width = width
        self.patch_size = patch_size
        self.layers = layers
        self.heads = heads
        self.is_built = False

    def build(self, input_shape):
        self.input_resolution = input_shape[-1]
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(self.width))
        self.positional_embedding = nn.Parameter(
            scale
            * torch.randn(
                (self.input_resolution // self.patch_size) ** 2 + 1, self.width
            )
        )
        self.ln_pre = LayerNorm(self.width)

        self.transformer = Transformer(self.width, self.layers, self.heads)

        self.is_built = True

    def forward(self, x: torch.Tensor):

        if not self.is_built:
            self.build(x.shape)

        self.to(x.device)

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        return x


class AutoConv1DTransformers(BaseLinearOutputModel):
    def __init__(self, config: AutoConv1DTransformersConfig):
        feature_embedding_modules = [
            nn.InstanceNorm1d,
            Conv1DTransformer,
            AvgPoolFlexibleDimension,
        ]
        feature_embeddings_args = [
            dict(num_features=2, affine=True, track_running_stats=True),
            dict(
                grid_patch_size=config.grid_patch_size,
                densenet_num_filters=config.densenet_num_filters,
                densenet_num_stages=config.densenet_num_stages,
                densenet_num_blocks=config.densenet_num_blocks,
                densenet_adaptive_pool_output_features=config.densenet_adaptive_pool_output_features,
                densenet_dilated=config.densenet_dilated,
                transformer_num_filters=config.transformer_num_filters,
                transformer_num_layers=config.transformer_num_layers,
                transformer_num_heads=config.transformer_num_heads,
                transformer_dim_feedforward=config.transformer_dim_feedforward,
                stem_conv_bias=True,
            ),
            dict(dim=1),
        ]
        super(AutoConv1DTransformers, self).__init__(
            embedding_output_features=config.embedding_output_features,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )


class AutoVideoTransformers(BaseLinearOutputModel):
    def __init__(
        self,
        config: AutoVideoTransformersConfig,
        image_embedding: Optional[nn.Module] = None,
    ):
        self.config = config
        feature_embedding_modules = [
            VideoTransformer,
            AvgPoolFlexibleDimension,
        ]

        feature_embeddings_args = [
            dict(
                image_embedding=image_embedding,
                transformer_num_filters=config.transformer_num_filters,
                transformer_num_layers=config.transformer_num_layers,
                transformer_num_heads=config.transformer_num_heads,
                transformer_dim_feedforward=config.transformer_dim_feedforward,
            ),
            dict(dim=1),
        ]
        super(AutoVideoTransformers, self).__init__(
            embedding_output_features=config.embedding_output_features,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )

    def connect_image_embedding(self, image_embedding: nn.Module):
        self.__init__(config=self.config, image_embedding=image_embedding)


class AutoAverager(BaseLinearOutputModel):
    def __init__(
        self, config: AutoAveragerConfig, image_embedding: Optional[nn.Module] = None
    ):
        self.config = config
        feature_embedding_modules = [
            Averager,
            AvgPoolFlexibleDimension,
        ]

        feature_embeddings_args = [
            dict(image_embedding=image_embedding),
            dict(dim=config.dim),
        ]
        super(AutoAverager, self).__init__(
            embedding_output_features=config.embedding_output_features,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )

    def connect_image_embedding(self, image_embedding: nn.Module):
        self.__init__(config=self.config, image_embedding=image_embedding)


class AutoCLIPVisionTransformer(BaseLinearOutputModel):
    def __init__(self, config: AutoCLIPVisionTransformerConfig):
        feature_embedding_modules = [
            nn.InstanceNorm2d,
            CLIPVisionModel,
        ]

        feature_embeddings_args = [
            dict(num_features=3, affine=True, track_running_stats=True),
            dict(
                config=CLIPVisionConfig(
                    hidden_size=config.transformer_num_filters,
                    intermediate_size=config.transformer_dim_feedforward,
                    num_hidden_layers=config.transformer_num_layers,
                    num_attention_heads=config.transformer_num_heads,
                    image_size=config.image_resolution,
                    patch_size=config.grid_patch_size,
                    hidden_act="quick_gelu",
                    layer_norm_eps=0.00001,
                    dropout=0.0,
                    attention_dropout=0.0,
                    initializer_range=0.02,
                    initializer_factor=1.0,
                )
            ),
        ]
        super(AutoCLIPVisionTransformer, self).__init__(
            embedding_output_features=config.embedding_output_features,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )


class AutoCLIPResNet(BaseLinearOutputModel):
    def __init__(self, config: AutoCLIPResNetConfig):
        vision_heads = config.vision_width * 32 // 64

        feature_embedding_modules = [
            nn.InstanceNorm2d,
            ModifiedResNet,
        ]

        feature_embeddings_args = [
            dict(num_features=3, affine=True, track_running_stats=True),
            dict(
                layers=config.vision_layers,
                output_dim=config.embedding_output_features,
                heads=vision_heads,
                input_resolution=config.image_resolution,
                width=config.vision_width,
            ),
        ]
        super(AutoCLIPResNet, self).__init__(
            embedding_output_features=config.embedding_output_features,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )


class AutoCLIPTextTransformer(BaseLinearOutputModel):
    def __init__(self, config: AutoCLIPTextTransformerConfig):
        text_config = CLIPTextConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.transformer_num_filters,
            intermediate_size=config.transformer_dim_feedforward,
            num_hidden_layers=config.transformer_num_layers,
            num_attention_heads=config.transformer_num_heads,
            max_position_embeddings=config.context_length,
            hidden_act="quick_gelu",
            layer_norm_eps=0.00001,
            dropout=0.0,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=1.0,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
        )
        feature_embedding_modules = [CLIPTextModel]

        feature_embeddings_args = [dict(config=text_config)]

        super(AutoCLIPTextTransformer, self).__init__(
            embedding_output_features=config.embedding_output_features,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
            input_type="long",
        )
