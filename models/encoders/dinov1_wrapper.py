import torch
import torch.nn as nn
from config.base import EncoderConfig
from .base import BaseEncoder

class DinoViTWrapper(BaseEncoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        self.encoder = torch.hub.load('facebookresearch/dino:main', config.name if 'lora' not in config.name else config.name.replace("lora_", ""))

        if 'lora' in config.name:
            self._apply_lora(lora_r=config.lora_r, lora_alpha=config.lora_alpha, lora_target=config.lora_target, lora_qkv_enable=config.lora_qkv_enable)
        else:
            if config.freeze:
                for param in self.encoder.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        for blk in self.encoder.blocks:
            x = blk(x)
        _, N, D = x.shape
        x = x[:, 1:, :].swapaxes(
            1, 2).view(-1, D, int(N ** 0.5), int(N ** 0.5))
        return x