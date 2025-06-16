# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch.cuda.amp import autocast

# from detectron2.config import configurable
# from detectron2.layers import Conv2d, ShapeSpec, get_norm
from config import ShapeSpec
# from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.position_encoding import PositionEmbeddingSine
from ..transformer_decoder.transformer import _get_clones, _get_activation_fn
from .ops.modules import MSDeformAttn

def get_norm(norm: str, out_channels: int) -> nn.Module:
    """
    Create a normalization layer.
    
    This function mimics detectron2's get_norm but works independently.
    It's crucial for Mask2Former because different normalization strategies
    can significantly impact training stability and performance.
    
    Args:
        norm: Type of normalization ("BN", "GN", "LN", etc.)
        out_channels: Number of output channels
        
    Returns:
        Appropriate normalization layer
    """
    if norm == "BN":
        return nn.BatchNorm2d(out_channels)
    elif norm == "GN":
        # GroupNorm with 32 groups is a common choice
        # It's more stable than BatchNorm for small batch sizes
        num_groups = min(32, out_channels)
        # Ensure divisibility
        while out_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        return nn.GroupNorm(num_groups, out_channels)
    elif norm == "LN":
        # LayerNorm implemented as GroupNorm with 1 group
        return nn.GroupNorm(1, out_channels)
    elif norm == "" or norm is None:
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported norm type: {norm}")
        
# class Conv2d(nn.Module):
#     """
#     Enhanced Conv2d layer that combines convolution, normalization, and activation.
    
#     This implementation provides detectron2-like functionality while being completely
#     self-contained and adaptable to your specific needs. Think of it as a "smart"
#     convolution layer that handles all the common patterns you need in modern
#     computer vision models.
    
#     The key insight is that most conv layers in modern architectures follow the
#     same pattern: Conv -> Norm -> Activation. By bundling these together, we can
#     ensure consistency, reduce boilerplate code, and make architectural changes
#     easier to implement.
#     """
    
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: Union[int, tuple] = 1,
#         stride: Union[int, tuple] = 1,
#         padding: Union[int, tuple, str] = 0,
#         dilation: Union[int, tuple] = 1,
#         groups: int = 1,
#         bias: Optional[bool] = None,
#         norm: Optional[str] = None,
#         activation: Optional[Union[str, Callable]] = None,
#     ):
#         """
#         Initialize the enhanced Conv2d layer.
        
#         Args:
#             in_channels, out_channels, kernel_size, stride, padding, dilation, groups:
#                 Standard PyTorch Conv2d parameters
#             bias: Whether to use bias. If None, automatically determined based on norm
#             norm: Normalization type ('BN', 'GN', 'LN', or None)
#             activation: Activation function ('relu', 'gelu', 'swish', or callable, or None)
#         """
#         super().__init__()
        
#         # Store configuration for debugging and introspection
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.norm_type = norm
#         self.activation_type = activation
        
#         # Determine bias usage: disable bias when using normalization
#         # This is a key insight - normalization layers typically handle the bias term,
#         # so having bias in both places is redundant and can hurt training
#         if bias is None:
#             bias = norm is None
        
#         # Create the core convolution layer
#         self.conv = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=groups,
#             bias=bias
#         )
        
#         # Add normalization layer if specified
#         self.norm = self._build_norm_layer(norm, out_channels)
        
#         # Add activation function if specified
#         self.activation = self._build_activation_layer(activation)
        
#         # Apply proper weight initialization
#         self._initialize_weights()
    
#     def _build_norm_layer(self, norm_type: Optional[str], num_features: int) -> Optional[nn.Module]:
#         """
#         Build the normalization layer based on the specified type.
        
#         This method encapsulates the logic for creating different types of
#         normalization layers, making it easy to experiment with different
#         normalization strategies without changing the core model code.
#         """
#         if norm_type is None:
#             return None
        
#         norm_type = norm_type.upper()
        
#         if norm_type == "BN":
#             # BatchNorm: normalizes across the batch dimension
#             # Good for large batch sizes, less effective for small batches
#             return nn.BatchNorm2d(num_features)
        
#         elif norm_type == "GN":
#             # GroupNorm: normalizes across groups of channels
#             # More stable than BatchNorm for small batches
#             # Use 32 groups as a reasonable default, but ensure we don't exceed num_features
#             num_groups = min(32, num_features)
#             # Ensure num_features is divisible by num_groups
#             while num_features % num_groups != 0:
#                 num_groups -= 1
#             return nn.GroupNorm(num_groups, num_features)
        
#         elif norm_type == "LN":
#             # LayerNorm: normalizes across the feature dimension
#             # Often used in transformer architectures
#             return nn.GroupNorm(1, num_features)  # GroupNorm with 1 group is equivalent to LayerNorm
        
#         elif norm_type == "IN":
#             # InstanceNorm: normalizes across spatial dimensions for each channel independently
#             # Sometimes used in style transfer and other specific applications
#             return nn.InstanceNorm2d(num_features)
        
#         else:
#             raise ValueError(f"Unsupported normalization type: {norm_type}")
    
#     def _build_activation_layer(self, activation: Optional[Union[str, Callable]]) -> Optional[nn.Module]:
#         """
#         Build the activation layer based on the specified type.
        
#         This method provides a flexible way to specify activation functions
#         either as strings (for common activations) or as callable objects
#         (for custom activations).
#         """
#         if activation is None:
#             return None
        
#         if isinstance(activation, str):
#             activation = activation.lower()
            
#             if activation == "relu":
#                 return nn.ReLU(inplace=True)
#             elif activation == "gelu":
#                 return nn.GELU()
#             elif activation == "swish" or activation == "silu":
#                 return nn.SiLU(inplace=True)
#             elif activation == "leaky_relu":
#                 return nn.LeakyReLU(0.1, inplace=True)
#             elif activation == "elu":
#                 return nn.ELU(inplace=True)
#             else:
#                 raise ValueError(f"Unsupported activation type: {activation}")
        
#         elif callable(activation):
#             # Allow custom activation functions
#             return activation
        
#         else:
#             raise ValueError(f"Activation must be string or callable, got {type(activation)}")
    
#     def _initialize_weights(self):
#         """
#         Initialize weights using appropriate strategies for the layer configuration.
        
#         This method implements initialization best practices that have been
#         developed through extensive empirical research. The initialization
#         strategy is chosen based on the activation function and normalization
#         to ensure good training dynamics from the start.
#         """
#         # Determine the appropriate initialization based on activation type
#         if self.activation_type is None:
#             # Linear activation: use Xavier/Glorot initialization
#             nn.init.xavier_uniform_(self.conv.weight)
#         elif isinstance(self.activation_type, str):
#             activation_lower = self.activation_type.lower()
#             if activation_lower in ["relu", "leaky_relu", "elu"]:
#                 # ReLU-family activations: use He initialization
#                 nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
#             elif activation_lower in ["gelu", "swish", "silu"]:
#                 # Smooth activations: use Xavier with slight modification
#                 nn.init.xavier_normal_(self.conv.weight)
#             else:
#                 # Default to Xavier for unknown activations
#                 nn.init.xavier_uniform_(self.conv.weight)
#         else:
#             # Custom activation: use conservative Xavier initialization
#             nn.init.xavier_uniform_(self.conv.weight)
        
#         # Initialize bias if present
#         if self.conv.bias is not None:
#             nn.init.constant_(self.conv.bias, 0)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass through conv -> norm -> activation pipeline.
        
#         This method demonstrates the typical flow in modern computer vision
#         architectures where these three operations are almost always used together.
#         """
#         # Apply convolution
#         x = self.conv(x)
        
#         # Apply normalization if present
#         if self.norm is not None:
#             x = self.norm(x)
        
#         # Apply activation if present
#         if self.activation is not None:
#             x = self.activation(x)
        
#         return x
    
#     def extra_repr(self) -> str:
#         """
#         Provide a string representation that shows the full configuration.
        
#         This makes it easy to understand the layer configuration when
#         printing model architectures or debugging.
#         """
#         parts = []
#         parts.append(f"{self.in_channels}, {self.out_channels}")
#         parts.append(f"kernel_size={self.conv.kernel_size}")
#         parts.append(f"stride={self.conv.stride}")
        
#         if self.conv.padding != (0, 0):
#             parts.append(f"padding={self.conv.padding}")
        
#         if self.conv.dilation != (1, 1):
#             parts.append(f"dilation={self.conv.dilation}")
        
#         if self.conv.groups != 1:
#             parts.append(f"groups={self.conv.groups}")
        
#         if self.conv.bias is None:
#             parts.append("bias=False")
        
#         if self.norm_type:
#             parts.append(f"norm={self.norm_type}")
        
#         if self.activation_type:
#             parts.append(f"activation={self.activation_type}")
        
#         return ", ".join(parts)

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

def check_if_dynamo_compiling():
    if TORCH_VERSION >= (2, 1):
        from torch._dynamo import is_compiling

        return is_compiling()
    else:
        return False

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            # Dynamo doesn't support context managers yet
            is_dynamo_compiling = check_if_dynamo_compiling()
            if not is_dynamo_compiling:
                with warnings.catch_warnings(record=True):
                    if x.numel() == 0 and self.training:
                        # https://github.com/pytorch/pytorch/issues/12013
                        assert not isinstance(
                            self.norm, torch.nn.SyncBatchNorm
                        ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

# MSDeformAttn Transformer encoder in deformable detr
class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4,
        ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        return memory, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


# @SEM_SEG_HEADS_REGISTRY.register()
class MSDeformAttnPixelDecoder(nn.Module):
    # @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        # deformable transformer encoder args
        transformer_in_features: List[str],
        common_stride: int,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }

        # this is the input shape of pixel decoder
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]
        
        # this is the input shape of transformer encoder (could use less features than pixel decoder
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape]  # starting from "res2" to "res5"
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]  # to decide extra FPN layers

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # from low resolution to high resolution (res5 -> res2)
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.mask_dim = mask_dim
        # use 1x1 conv instead
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.mask_features)
        
        self.maskformer_num_feature_levels = 3  # always use 3 scales
        self.common_stride = common_stride

        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)

            lateral_conv = Conv2d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    # @classmethod
    # def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
    #     ret = {}
    #     ret["input_shape"] = {
    #         k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
    #     }
    #     ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
    #     ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
    #     ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
    #     ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
    #     ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
    #     # ret["transformer_dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
    #     ret["transformer_dim_feedforward"] = 1024  # use 1024 for deformable transformer encoder
    #     ret[
    #         "transformer_enc_layers"
    #     ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # a separate config
    #     ret["transformer_in_features"] = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES
    #     ret["common_stride"] = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
    #     return ret

    @autocast(enabled=False)
    def forward_features(self, features):
        srcs = []
        pos = []
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()  # deformable detr does not support half precision
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))

        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]

        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = output_conv(y)
            out.append(y)

        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        return self.mask_features(out[-1]), out[0], multi_scale_features
