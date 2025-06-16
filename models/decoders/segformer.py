import torch
import torch.nn as nn

from config.base import DecoderConfig
from .base import BaseDecoder

## Modified from https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/decode_heads/segformer_head.py
class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

## Modified from https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/decode_heads/segformer_head.py
class SegFormerHead(BaseDecoder):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, config: DecoderConfig):
        super().__init__(config)
        # assert len(config.feature_strides) == len(config.in_channels)
        # assert min(config.feature_strides) == config.feature_strides[0]
        self.in_channels = config.in_channels
        # self.feature_strides = config.feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = config.embed_dim

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = nn.Conv2d(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim, 
            kernel_size=(1, 1),
        )

        self.linear_pred = nn.Conv2d(embedding_dim, config.num_classes, kernel_size=(1, 1))
        self.dropout = nn.Dropout(config.dropout)
        self.encoder_name = config.encoder_name
        in_dim = config.input_dim

        if self.encoder_name == 'dino_vits8':
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(in_dim, in_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Identity()

            self.fpn3 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.fpn4 = nn.MaxPool2d(kernel_size=4, stride=4)

        elif self.encoder_name == 'dinov2_vits14' or self.encoder_name == 'clip_vitb16':
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(in_dim, in_dim, kernel_size=2, stride=2),
                Norm2d(in_dim),
                nn.GELU(),
                nn.ConvTranspose2d(in_dim, in_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(in_dim, in_dim, kernel_size=2, stride=2),
            )

            self.fpn3 = nn.Identity()

            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)



    def _transform_inputs(self, inputs):
        if self.encoder_name in ['dino_vits8', 'dinov2_vits14', 'clip_vitb16']:
            return self.fpn1(inputs), self.fpn2(inputs), self.fpn3(inputs), self.fpn4(inputs)
        else:
            return inputs

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = torch.nn.functional.interpolate(_c4, size=c1.size()[2:], mode="bilinear", align_corners=False)
        
        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = torch.nn.functional.interpolate(_c3, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = torch.nn.functional.interpolate(_c2, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x