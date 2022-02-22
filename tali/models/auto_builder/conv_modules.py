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

from tali.base import utils

log = utils.get_logger(__name__)


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


class ConvPool1DStemBlock(nn.Module):
    def __init__(
        self,
        num_filters,
        kernel_size,
        padding,
        dilation,
        pool_kernel_size,
        pool_stride,
        **kwargs,
    ):
        super(ConvPool1DStemBlock, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.is_built = False
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape)
        out = dummy_x

        self.layer_dict["stem_conv"] = nn.Conv1d(
            in_channels=out.shape[1],
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            stride=2,
            padding=self.padding,
            bias=False,
        )

        self.layer_dict["stem_leaky_relu"] = nn.LeakyReLU(0.1)

        self.layer_dict["stem_instance_norm"] = nn.InstanceNorm1d(
            self.num_filters, affine=True, track_running_stats=True
        )

        self.layer_dict["stem_pool"] = nn.AvgPool1d(
            kernel_size=self.pool_kernel_size, stride=self.pool_stride
        )

        out = self.layer_dict["stem_conv"].forward(out)
        out = self.layer_dict["stem_leaky_relu"].forward(out)
        out = self.layer_dict["stem_instance_norm"].forward(out)

        self.is_built = True
        build_string = (
            f"Built module {self.__class__.__name__} "
            f"with input -> output sizes {dummy_x.shape} {out.shape}"
        )

        logging.info(build_string)

    def forward(self, x):
        if not self.is_built:
            self.build(x.shape)

        out = x
        out = self.layer_dict["stem_conv"].forward(out)
        out = self.layer_dict["stem_leaky_relu"].forward(out)
        out = self.layer_dict["stem_instance_norm"].forward(out)

        return out


class ResNetBlock1D(nn.Module):
    def __init__(self, num_filters, kernel_size, dilation, **kwargs):
        super(ResNetBlock1D, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.is_built = False
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation = dilation

    def build(self, input_shape):

        dummy_x = torch.zeros(input_shape)

        self.layer_dict["conv_filter_pool"] = nn.Conv1d(
            in_channels=dummy_x.shape[1],
            out_channels=self.num_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        dummy_x = self.layer_dict["conv_filter_pool"].forward(dummy_x)

        log.debug(f"dummy_x shape: {dummy_x.shape}")

        out = dummy_x

        self.layer_dict["conv_0"] = nn.Conv1d(
            in_channels=out.shape[1],
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            stride=self.kernel_size,
            padding=0,
            bias=False,
        )

        out = self.layer_dict["conv_0"].forward(out)

        log.debug(f"out shape: {out.shape}")

        self.layer_dict["leaky_relu_0"] = nn.LeakyReLU(0.1)

        out = self.layer_dict["leaky_relu_0"].forward(out)

        log.debug(f"out shape: {out.shape}")

        self.layer_dict["instance_norm_0"] = nn.InstanceNorm1d(
            self.num_filters, affine=True, track_running_stats=True
        )

        out = self.layer_dict["instance_norm_0"].forward(out)

        log.debug(f"out shape: {out.shape}")

        self.layer_dict["conv_spatial_pool"] = nn.Conv1d(
            in_channels=dummy_x.shape[1],
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            stride=self.kernel_size,
            padding=0,
            bias=False,
        )

        dummy_x = self.layer_dict["conv_spatial_pool"].forward(dummy_x)

        log.debug(f"dummy_x shape: {dummy_x.shape}")

        out = dummy_x + out

        log.debug(f"out shape: {out.shape}")

        self.is_built = True

        build_string = (
            f"Built module {self.__class__.__name__} "
            f"with input -> output sizes {input_shape} {out.shape}"
        )

        logging.info(build_string)

    def forward(self, x):
        if not self.is_built:
            self.build(x.shape)

        x = self.layer_dict["conv_filter_pool"].forward(x)

        out = x

        out = self.layer_dict["conv_0"].forward(out)

        out = self.layer_dict["leaky_relu_0"].forward(out)

        out = self.layer_dict["instance_norm_0"].forward(out)

        x = self.layer_dict["conv_spatial_pool"].forward(x)

        out = x + out

        return out


class ResNetReductionBlock1D(nn.Module):
    def __init__(self, num_filters, kernel_size, dilation, reduction_rate, **kwargs):
        super(ResNetReductionBlock1D, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.is_built = False
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.reduction_rate = reduction_rate

    def build(self, input_shape):

        dummy_x = torch.zeros(input_shape)

        self.layer_dict["conv_filter_pool"] = nn.Conv1d(
            in_channels=dummy_x.shape[1],
            out_channels=self.num_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        dummy_x = self.layer_dict["conv_filter_pool"].forward(dummy_x)

        self.layer_dict["conv_1"] = nn.Conv1d(
            in_channels=dummy_x.shape[1],
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            bias=False,
        )

        self.layer_dict["reduction_pool"] = nn.AvgPool1d(
            kernel_size=self.reduction_rate, stride=self.reduction_rate
        )

        self.layer_dict["conv_2"] = nn.Conv1d(
            in_channels=self.num_filters,
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            bias=False,
        )
        self.layer_dict["gelu_0"] = nn.LeakyReLU(0.1)
        self.layer_dict["instance_norm_0"] = nn.InstanceNorm1d(
            self.num_filters, affine=True, track_running_stats=True
        )

        self.layer_dict["gelu_1"] = nn.LeakyReLU(0.1)
        self.layer_dict["instance_norm_1"] = nn.InstanceNorm1d(
            self.num_filters, affine=True, track_running_stats=True
        )

        out = dummy_x

        pool_x = self.layer_dict["reduction_pool"](dummy_x)
        out = self.layer_dict["conv_1"].forward(out)
        out = self.layer_dict["gelu_0"].forward(out)
        out = self.layer_dict["instance_norm_0"].forward(out)
        pool_out = self.layer_dict["reduction_pool"].forward(out)
        out = self.layer_dict["conv_2"].forward(pool_out)
        out = self.layer_dict["gelu_1"].forward(out)
        out = self.layer_dict["instance_norm_1"].forward(out)

        out = pool_x + out

        self.is_built = True

        build_string = (
            f"Built module {self.__class__.__name__} "
            f"with input -> output sizes {dummy_x.shape} {out.shape}"
        )

        logging.debug(build_string)

    def forward(self, x):
        if not self.is_built:
            self.build(x.shape)

        x = self.layer_dict["conv_filter_pool"].forward(x)
        pool_x = self.layer_dict["reduction_pool"](x)
        out = x

        out = self.layer_dict["conv_1"].forward(out)
        out = self.layer_dict["gelu_0"].forward(out)
        out = self.layer_dict["instance_norm_0"].forward(out)
        pool_out = self.layer_dict["reduction_pool"].forward(out)
        out = self.layer_dict["conv_2"].forward(pool_out)
        out = self.layer_dict["gelu_1"].forward(out)
        out = self.layer_dict["instance_norm_1"].forward(out)

        out = pool_x + out

        return out


class ConvNetEmbedding(nn.Module):
    def __init__(
        self,
        num_filters: int,
        num_stages: int,
        num_blocks: int,
        kernel_size: int,
        stem_block_type: nn.Module,
        processing_block_type: nn.Module,
        reduction_block_type: nn.Module,
        dilated: bool = False,
    ):
        super(ConvNetEmbedding, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.is_built = False
        self.num_filters = num_filters
        self.num_stages = num_stages
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.stem_block_type = stem_block_type
        self.processing_block_type = processing_block_type
        self.reduction_block_type = reduction_block_type
        self.dilated = dilated

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape)
        out = dummy_x

        dilation_factor = 1

        self.layer_dict["stem_conv"] = self.stem_block_type(
            num_filters=4 * self.num_filters,
            kernel_size=self.kernel_size,
            stride=self.kernel_size,
            padding=0,
            dilation=1,
            pool_kernel_size=2,
            pool_stride=2,
        )

        out = self.layer_dict["stem_conv"].forward(out)

        for stage_idx in range(self.num_stages):

            for block_idx in range(self.num_blocks):

                if self.dilated:
                    dilation_factor = 2**block_idx

                self.layer_dict[
                    f"stage_{stage_idx}_block_{block_idx}"
                ] = self.processing_block_type(
                    num_filters=(stage_idx * self.num_filters) + self.num_filters,
                    kernel_size=self.kernel_size,
                    dilation=dilation_factor,
                )

                out = self.layer_dict[f"stage_{stage_idx}_block_{block_idx}"].forward(
                    out
                )

            self.layer_dict[
                f"stage_{stage_idx}_dim_reduction"
            ] = self.reduction_block_type(
                num_filters=(stage_idx * self.num_filters) + self.num_filters,
                kernel_size=self.kernel_size,
                dilation=dilation_factor,
                reduction_rate=2,
            )

            out = self.layer_dict[f"stage_{stage_idx}_dim_reduction"].forward(out)

        self.is_built = True
        logging.info(
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
                out = self.layer_dict[f"stage_{stage_idx}_block_{block_idx}"].forward(
                    out
                )

            out = self.layer_dict[f"stage_{stage_idx}_dim_reduction"].forward(out)

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
            ConvNetEmbedding,
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
            ConvNetEmbedding,
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
        feature_embedding_modules = [Permute, ConvNetEmbedding, nn.Flatten]

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
            ConvNetEmbedding,
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
