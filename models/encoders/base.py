import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, List

from config.base import EncoderConfig
import loralib as lora

class BaseEncoder(nn.Module):
    """Base class for all encoder models."""
    
    def __init__(self, config: EncoderConfig):
        """
        Initialize the encoder.
        
        Args:
            config: Encoder configuration
        """
        super().__init__()
        self.config = config
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Dictionary with encoded features and metadata
        """
        raise NotImplementedError("Subclasses must implement forward")
    
    # def _apply_lora(self, lora_r: int, lora_alpha: float, lora_target: List[str]):
    def _apply_lora(self, lora_r: int, lora_alpha: float, lora_target: List[str], lora_qkv_enable: List[bool] = [True, True, True]):
        """Apply LoRA to specific modules in the model using loralib."""
        lora_counter = 0
        
        # Find and replace linear layers with LoRA versions
        for name, module in self.encoder.named_modules():
            # Check if this is a target attention module
            if isinstance(module, torch.nn.Linear) and any(target in name for target in lora_target):
                # Get parent module and child name
                parent_name, child_name = name.rsplit('.', 1)
                parent = self.encoder
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                
                # Get original module parameters
                original = getattr(parent, child_name)
                in_features = original.in_features
                out_features = original.out_features
                bias = original.bias is not None
                
                if 'qkv' in name:
                    lora_layer = lora.MergedLinear(
                        in_features=in_features,
                        out_features=out_features,
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        enable_lora=lora_qkv_enable,
                        bias=bias,
                    )
                else:
                    lora_layer = lora.Linear(
                        in_features=in_features,
                        out_features=out_features,
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        bias=bias,
                    )
                # lora_layer = lora.Linear(
                #     in_features=in_features,
                #     out_features=out_features,
                #     r=lora_r,
                #     lora_alpha=lora_alpha,
                #     bias=bias,
                # )
                    
                # Copy original weights
                lora_layer.weight.data.copy_(original.weight.data)
                if bias:
                    lora_layer.bias.data.copy_(original.bias.data)
                
                # Replace the layer
                setattr(parent, child_name, lora_layer)
                # self.lora_layers[name] = lora_layer
                lora_counter += 1
        
        # Mark only LoRA parameters as trainable
        lora.mark_only_lora_as_trainable(self.encoder)
        
        print(f"Applied LoRA to {lora_counter} layers in {self.config.name}")

    def get_output_dim(self) -> int:
        """
        Get the output dimension of the encoder.
        
        Returns:
            Output dimension
        """
        return self.config.output_dim