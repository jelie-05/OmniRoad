import torch
from config.base import EncoderConfig
from .base import BaseEncoder
import clip

model_name_map = {
    'clip_vitb16': 'ViT-B/16',
    'clip_vitb32': 'ViT-B/32',
    'clip_vitl14': 'ViT-L/14',
    'clip_vitl144@336px': 'ViT-L/14@336px'
}

class CLIPViTWrapper(BaseEncoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)

        try:
            clip_full_model, _ = clip.load(model_name_map[config.name], download_root='/home/phd_li/.cache/clip')
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model {model_name_map[config.name]}: {e}")
        
        self.encoder = clip_full_model.visual.float()

        if config.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.patch_size = self.encoder.conv1.kernel_size[0]
        self.output_dim = self.encoder.proj.shape[0]
        self.attention_heads = self.encoder.transformer.resblocks[0].attn.num_heads

        assert self.config.patch_size == self.patch_size, f"patch_size ({self.config.patch_size}) in config doesn't match the patch_size ({self.patch_size}) of the selected backbone"
        assert self.config.output_dim == self.output_dim, f"output_dim ({self.config.output_dim}) in config doesn't match the embed_dim ({self.output_dim}) of the selected backbone"
        assert self.config.attention_heads == self.attention_heads, f"attention_heads ({self.config.attention_heads}) in config doesn't match the num_heads ({self.attention_heads}) of the selected backbone"
        

    def forward(self, x):
        batch_size, _, h, w = x.shape

        x = self.encoder.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.encoder.positional_embedding.to(x.dtype)
        x = self.encoder.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.encoder.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        patch_features = x[:, 1:, :]

        # Reshape to spatial format [B, C, H, W]
        spatial_features = patch_features.transpose(1, 2).reshape(
            batch_size, self.output_dim, h // self.patch_size, w // self.patch_size
        )
        # print(spatial_features.shape)
        return spatial_features

