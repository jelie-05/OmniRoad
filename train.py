import os
import random
import numpy as np
import torch
torch.hub.set_dir('/home/phd_li/.cache/torch/hub')
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from torchinfo import summary
from config import load_config_from_args, ModelRegistry, EncoderRegistry, DecoderRegistry
from models import create_model
from data import get_data_loaders
from trainer import Trainer

def main():
    """Main training function."""
    # Load configuration
    config = load_config_from_args()
    
    # Set random seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Starting experiment: {config.experiment_name}")
    print(f"Encoder: {config.model.encoder.name}, Freeze: {config.model.encoder.freeze}")
    print(f"Decoder: {config.model.decoder.name}, Output dim: {config.model.decoder.num_classes}")
    print(f"Training for {config.training.epochs} epochs with lr={config.training.optimizer.learning_rate}")
    
    # Create the model
    model = create_model(config)
    summary(model)
    print("Model created successfully!")
    print(f"Encoder output dimension: {model.encoder.get_output_dim()}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


    # Get data loaders
    train_loader, val_loader, _ = get_data_loaders(config)
    
    # Print training info
    print(f"Training configuration:")
    print(f"- Experiment: {config.experiment_name}")
    print(f"- Model: {config.model.encoder.name} encoder with {config.model.decoder.name}")
    print(f"- Device: {device}")
    print(f"- Batch size: {config.training.batch_size}")
    print(f"- Learning rate: {config.training.optimizer.learning_rate}")
    print(f"- Epochs: {config.training.epochs}")
    print(f"- Encoder freeze: {config.model.encoder.freeze}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Run training
    train_stats = trainer.train()
    
    print(f"Training complete!")
    print(f"Best validation loss: {train_stats['best_val_loss']:.4f} (epoch {train_stats['best_epoch'] + 1})")
    print(f"Best mean IoU: {train_stats['best_miou']:.4f} (epoch {train_stats['best_epoch'] + 1})")
    print(f"Training time: {train_stats['training_time'] / 60:.2f} minutes")

    # If there are final metrics, report them
    if 'final_metrics' in train_stats and train_stats['final_metrics']:
        print("\nFinal metrics:")
        metrics = train_stats['final_metrics']
        print(f"  Loss: {metrics.get('val_loss', 'N/A'):.4f}")
        print(f"  Pixel Accuracy: {metrics.get('pixel_accuracy', 'N/A'):.4f}")
        print(f"  Mean IoU: {metrics.get('mean_iou', 'N/A'):.4f}")
    
    

if __name__ == "__main__":
    main()