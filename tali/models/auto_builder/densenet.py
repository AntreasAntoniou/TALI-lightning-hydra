from __future__ import print_function

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gate.model_blocks.auto_builder_modules.auto_builder_conv_blocks import (
    Conv2dBNLeakyReLU,
    ClassificationModel,
    SqueezeExciteConv2dBNLeakyReLU,
    SqueezeExciteConv1dBNLeakyReLU,
    Conv1dBNLeakyReLU,
)


class DenseBlock(nn.Module):
    def __init__(
        self,
        num_filters,
        dilation_factor,
        kernel_size,
        stride,
        downsample_output_size=None,
        processing_block_type=Conv2dBNLeakyReLU,
    ):
        super(DenseBlock, self).__init__()
        self.num_filters = num_filters
        self.dilation_factor = dilation_factor
        self.downsample_output_size = downsample_output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.processing_block_type = processing_block_type
        self.is_built = False

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape)

        self.conv_bn_relu_in = self.processing_block_type(
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.dilation_factor,
            dilation=self.dilation_factor,
            bias=False,
        )

        out = self.conv_bn_relu_in.forward(dummy_x)

        if self.downsample_output_size is not None:
            if len(out.shape) == 4:
                out = F.adaptive_avg_pool2d(
                    output_size=self.downsample_output_size, input=out
                )
                dummy_x = F.adaptive_avg_pool2d(
                    output_size=self.downsample_output_size, input=dummy_x
                )
            elif len(out.shape) == 3:
                out = F.adaptive_avg_pool1d(
                    output_size=self.downsample_output_size, input=out
                )
                dummy_x = F.adaptive_avg_pool1d(
                    output_size=self.downsample_output_size, input=dummy_x
                )
            else:
                return ValueError("Dimensions of features are neither 3 nor 4")

        out = torch.cat([dummy_x, out], dim=1)

        self.is_built = True
        logging.debug(
            f"Built module {self.__class__.__name__} with input -> output sizes "
            f"{dummy_x.shape} {out.shape}"
        )

    def forward(self, x):

        if not self.is_built:
            self.build(x.shape)

        out = self.conv_bn_relu_in.forward(x)
        if self.downsample_output_size is not None:
            # logging.debug(f'Downsampling output size: {out.shape}'
            #              f'{self.downsample_output_size}')
            if len(out.shape) == 4:
                out = F.avg_pool2d(kernel_size=2, input=out)
                x = F.avg_pool2d(kernel_size=2, input=x)
            elif len(out.shape) == 3:
                # logging.debug(out.shape)
                out = F.avg_pool1d(kernel_size=2, input=out)
                x = F.avg_pool1d(kernel_size=2, input=x)
            else:
                return ValueError("Dimensions of features are neither 3 nor 4")

        out = torch.cat([x, out], dim=1)

        return out


class DenseNetEmbedding(nn.Module):
    def __init__(
        self,
        num_filters,
        num_stages,
        num_blocks,
        processing_block_type,
        dilated=False,
    ):
        super(DenseNetEmbedding, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.is_built = False
        self.num_filters = num_filters
        self.num_stages = num_stages
        self.num_blocks = num_blocks
        self.processing_block_type = processing_block_type
        self.dilated = dilated

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape)
        out = dummy_x

        dilation_factor = 1

        self.layer_dict["stem_conv"] = DenseBlock(
            num_filters=self.num_filters * 4,
            dilation_factor=1,
            stride=1,
            kernel_size=3,
            downsample_output_size=int(np.floor(out.shape[-1] / 2)),
            processing_block_type=self.processing_block_type,
        )

        out = self.layer_dict["stem_conv"].forward(out)

        for stage_idx in range(self.num_stages):

            for block_idx in range(self.num_blocks):

                if self.dilated:
                    dilation_factor = 2 ** block_idx

                self.layer_dict[
                    "stage_{}_block_{}".format(stage_idx, block_idx)
                ] = DenseBlock(
                    num_filters=self.num_filters,
                    dilation_factor=dilation_factor,
                    downsample_output_size=None,
                    stride=1,
                    kernel_size=3,
                    processing_block_type=self.processing_block_type,
                )

                out = self.layer_dict[
                    "stage_{}_block_{}".format(stage_idx, block_idx)
                ].forward(out)

            self.layer_dict["stage_{}_dim_reduction".format(stage_idx)] = DenseBlock(
                num_filters=self.num_filters,
                dilation_factor=1,
                stride=1,
                kernel_size=3,
                downsample_output_size=int(np.floor(out.shape[-1] / 2)),
                processing_block_type=self.processing_block_type,
            )

            out = self.layer_dict["stage_{}_dim_reduction".format(stage_idx)].forward(
                out
            )

        self.is_built = True
        logging.debug(
            f"Built module {self.__class__.__name__} with input -> output sizes "
            f"{dummy_x.shape} {out.shape}"
        )

    def forward(self, x):

        if not self.is_built:
            self.build(input_shape=x.shape)

        out = x

        out = self.layer_dict["stem_conv"].forward(out)

        for stage_idx in range(self.num_stages):

            for block_idx in range(self.num_blocks):
                out = self.layer_dict[
                    "stage_{}_block_{}".format(stage_idx, block_idx)
                ].forward(out)

            out = self.layer_dict["stage_{}_dim_reduction".format(stage_idx)].forward(
                out
            )

        return out


class AutoConv2DDenseNet(ClassificationModel):
    def __init__(
        self,
        num_classes,
        num_filters,
        num_stages,
        num_blocks,
        use_squeeze_excite_attention,
        dilated=False,
        **kwargs,
    ):
        feature_embedding_modules = [
            nn.InstanceNorm2d,
            DenseNetEmbedding,
            nn.Flatten,
        ]

        feature_embeddings_args = [
            dict(num_features=3, affine=True, track_running_stats=True),
            dict(
                num_filters=num_filters,
                num_stages=num_stages,
                num_blocks=num_blocks,
                dilated=dilated,
                processing_block_type=SqueezeExciteConv2dBNLeakyReLU
                if use_squeeze_excite_attention
                else Conv2dBNLeakyReLU,
            ),
            dict(start_dim=1, end_dim=-1),
        ]

        super(AutoConv2DDenseNet, self).__init__(
            num_classes=num_classes,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )


class AutoConv1DDenseNet(ClassificationModel):
    def __init__(
        self,
        num_classes,
        num_filters,
        num_stages,
        num_blocks,
        use_squeeze_excite_attention,
        dilated=False,
        **kwargs,
    ):
        feature_embedding_modules = [
            nn.InstanceNorm1d,
            DenseNetEmbedding,
            nn.AdaptiveMaxPool1d,
        ]

        feature_embeddings_args = [
            dict(num_features=2, affine=True, track_running_stats=True),
            dict(
                num_filters=num_filters,
                num_stages=num_stages,
                num_blocks=num_blocks,
                dilated=dilated,
                processing_block_type=SqueezeExciteConv1dBNLeakyReLU
                if use_squeeze_excite_attention
                else Conv1dBNLeakyReLU,
            ),
            dict(output_size=2),
        ]

        super(AutoConv1DDenseNet, self).__init__(
            num_classes=num_classes,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )


class Permute(nn.Module):
    def __init__(self, permute_order: list):
        super().__init__()
        self.permute_order = permute_order

    def forward(self, x):
        return x.permute(self.permute_order)


class AutoConv1DDenseNetVideo(ClassificationModel):
    def __init__(
        self,
        num_classes,
        num_filters,
        num_stages,
        num_blocks,
        use_squeeze_excite_attention,
        dilated=False,
        **kwargs,
    ):
        feature_embedding_modules = [Permute, DenseNetEmbedding, nn.Flatten]

        feature_embeddings_args = [
            dict(permute_order=[0, 2, 1]),
            dict(
                num_filters=num_filters,
                num_stages=num_stages,
                num_blocks=num_blocks,
                dilated=dilated,
                processing_block_type=SqueezeExciteConv1dBNLeakyReLU
                if use_squeeze_excite_attention
                else Conv1dBNLeakyReLU,
            ),
            dict(start_dim=1, end_dim=-1),
        ]

        super(AutoConv1DDenseNetVideo, self).__init__(
            num_classes=num_classes,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )


class AutoTextNet(ClassificationModel):
    def __init__(
        self,
        vocab_size: int,
        num_filters: int,
        num_stages: int,
        num_blocks: int,
        use_squeeze_excite_attention: bool,
        dilated: bool,
        num_classes: int,
        **kwargs,
    ):
        feature_embedding_modules = [
            nn.Embedding,
            DenseNetEmbedding,
            nn.Flatten,
        ]

        feature_embeddings_args = [
            dict(num_embeddings=vocab_size, embedding_dim=num_filters),
            dict(
                num_filters=num_filters,
                num_stages=num_stages,
                num_blocks=num_blocks,
                dilated=dilated,
                processing_block_type=SqueezeExciteConv1dBNLeakyReLU
                if use_squeeze_excite_attention
                else Conv1dBNLeakyReLU,
            ),
            dict(),
        ]

        super(AutoTextNet, self).__init__(
            num_classes=num_classes,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
            input_type=torch.long,
        )
