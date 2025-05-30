import torch
from config.base import EncoderConfig
from .base import BaseEncoder
import timm

class SwinTransformerWrapper(BaseEncoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)

        self.encoder = timm.create_model(
            config.name,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        
        # Determine output dimension based on the model
        if hasattr(self.encoder, 'feature_info'):
            # Get dimensions of the last feature map
            assert [m['num_chs'] for m in self.encoder.feature_info] == config.output_dim, f"Output channels {[m['num_chs'] for m in self.encoder.feature_info]} are not equal to configuration {config.output_dim}"
            
        if config.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)
        features = [feat.permute(0, 3, 1, 2) for feat in features]
        return features

