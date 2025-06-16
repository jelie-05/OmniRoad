# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional, Union, Callable
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import warnings
# from detectron2.config import configurable
# from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
# from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY

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

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# @TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    # @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    # @classmethod
    # def from_config(cls, cfg, in_channels, mask_classification):
    #     ret = {}
    #     ret["in_channels"] = in_channels
    #     ret["mask_classification"] = mask_classification
        
    #     ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
    #     ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
    #     ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
    #     # Transformer parameters:
    #     ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
    #     ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

    #     # NOTE: because we add learnable query features which requires supervision,
    #     # we add minus 1 to decoder layers to be consistent with our loss
    #     # implementation: that is, number of auxiliary losses is always
    #     # equal to number of decoder layers. With learnable query features, the number of
    #     # auxiliary losses equals number of decoders plus 1.
    #     assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
    #     ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
    #     ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
    #     ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

    #     ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

    #     return ret

    def forward(self, x, mask_features, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
