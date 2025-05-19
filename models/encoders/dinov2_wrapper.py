import torch
from config.base import EncoderConfig
from .base import BaseEncoder

class Dinov2ViTWrapper(BaseEncoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)

        try:
            self.encoder = torch.hub.load('facebookresearch/dinov2', config.name)
        except Exception as e:
            raise RuntimeError(f"Failed to load DINOv2 model {config.name}: {e}")
        
        if config.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        assert self.config.patch_size == self.encoder.patch_size, f"patch_size ({self.config.patch_size}) in config doesn't match the patch_size ({self.encoder.patch_size}) of the selected backbone"
        assert self.config.output_dim == self.encoder.embed_dim, f"output_dim ({self.config.output_dim}) in config doesn't match the embed_dim ({self.encoder.embed_dim}) of the selected backbone"
        assert self.config.attention_heads == self.encoder.num_heads, f"attention_heads ({self.config.attention_heads}) in config doesn't match the num_heads ({self.encoder.num_heads}) of the selected backbone"
        

    def forward(self, x):
        batch_size, _, h, w = x.shape

        x = self.encoder.prepare_tokens_with_masks(x, None)

        for blk in self.encoder.blocks:
            x = blk(x)

        patch_features = x[:, self.encoder.num_register_tokens + 1 :]
                
        # Reshape to spatial format [B, C, H, W]
        spatial_features = patch_features.transpose(1, 2).reshape(
            batch_size, self.encoder.embed_dim, h // self.encoder.patch_size, w // self.encoder.patch_size
        )
        # print(spatial_features.shape)
        return spatial_features