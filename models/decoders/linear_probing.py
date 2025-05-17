import torch
import torch.nn as nn

from config.base import DecoderConfig
from .base import BaseDecoder

class LinearProbing(BaseDecoder):
    def __init__(self, config: DecoderConfig):
        super().__init__(config)

        layers = []
        in_dim = config.input_dim

        width = config.spatial_size
        height = config.spatial_size

        # Add hidden layers
        for i, dim in enumerate(config.hidden_dims):
            layers.append(torch.nn.Conv2d(in_dim, dim, (1, 1)))
            
            # Add activation
            if config.activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif config.activation.lower() == 'gelu':
                layers.append(nn.GELU())
            elif config.activation.lower() == 'leakyrelu':
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            # Add dropout if specified
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            
            in_dim = dim
        
        # Add final layer
        layers.append(torch.nn.Conv2d(in_dim, config.num_classes, (1, 1)))

        self.mlp = nn.Sequential(*layers)

    def forward(self, feature_maps):
        return self.mlp(feature_maps)
