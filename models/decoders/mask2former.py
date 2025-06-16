import torch
import torch.nn as nn

from config.base import DecoderConfig
from .base import BaseDecoder
from .mask2former_components import MSDeformAttnPixelDecoder, MultiScaleMaskedTransformerDecoder

class Mask2FormerHead(BaseDecoder):
    def __init__(self, config: DecoderConfig):
        super().__init__(config)
        
        self.pixel_decoder = MSDeformAttnPixelDecoder(
            input_shape=config.input_shape,
            transformer_nheads=config.deformable_transformer_pixel_decoder_nheads,
            transformer_dim_feedforward=config.transformer_dim_feedforward,
            transformer_enc_layers=config.transformer_enc_layers,
            conv_dim=config.conv_dim,
            mask_dim=mask_dim,
            norm=norm,
            # deformable transformer encoder args
            transformer_in_features=deformable_transformer_in_features,
            common_stride=common_stride
        )
        self.predictor = MultiScaleMaskedTransformerDecoder(
            in_channels,
            mask_classification=True,
            num_classes=config.num_classes,
            hidden_dim=config.maskformer_hidden_dim,
            num_queries=config.num_obj_queries,
            nheads=config.maskformer_nheads,
            dim_feedforward=config.maskformer_dim_feedforward,
            dec_layers=config.dec_layers,
            pre_norm=config.pre_norm,
            mask_dim=config.mask_dim,
            enforce_input_project=config.enforce_input_project
        )
        self.transformer_in_feature = config.maskformer_in_feature

    def forward(self, inputs, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features=inputs)
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(multi_scale_features, mask_features, mask)
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert (
                    transformer_encoder_features is not None
                ), "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(transformer_encoder_features, mask_features, mask)
            elif self.transformer_in_feature == "pixel_embedding":
                predictions = self.predictor(mask_features, mask_features, mask)
            else:
                predictions = self.predictor(features[self.transformer_in_feature], mask_features, mask)
        
        return predictions