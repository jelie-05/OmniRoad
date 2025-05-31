from config.base import EncoderConfig
from .base import BaseEncoder
import timm

class SwinTransformerWrapper(BaseEncoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        
            
        self.encoder = timm.create_model(
            config.name if 'lora' not in config.name else config.name.replace("lora_", ""),
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        
        # Determine output dimension based on the model
        if hasattr(self.encoder, 'feature_info'):
            # Get dimensions of the last feature map
            assert [m['num_chs'] for m in self.encoder.feature_info] == config.output_dim, f"Output channels {[m['num_chs'] for m in self.encoder.feature_info]} are not equal to configuration {config.output_dim}"
        
        if 'lora' in config.name:
            self._apply_lora(lora_r=config.lora_r, lora_alpha=config.lora_alpha, lora_target=config.lora_target, lora_qkv_enable=config.lora_qkv_enable)
            # self._apply_lora(lora_r=config.lora_r, lora_alpha=config.lora_alpha, lora_target=config.lora_target)
        else:
            if config.freeze:
                for param in self.encoder.parameters():
                    param.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)
        features = [feat.permute(0, 3, 1, 2) for feat in features]
        return features

