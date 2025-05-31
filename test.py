import os
import sys
import torch
torch.hub.set_dir('/home/phd_li/.cache/torch/hub')
os.environ['HF_HOME'] = '/home/phd_li/.cache/huggingface'

import argparse
from pathlib import Path
from torchinfo import summary

from config import load_config_from_args, load_config
from models import create_model
from data import get_data_loaders
from trainer import Trainer
from utility.distributed import (
    setup_distributed, 
    cleanup_distributed, 
    is_main_process,
    get_rank,
    get_world_size,
    init_seeds
)
import signal

def signal_handler(signum, frame):
    """Handle termination signals for graceful shutdown."""
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    cleanup_distributed()
    sys.exit(0)

def parse_test_args():
    """Parse command line arguments for testing."""
    parser = argparse.ArgumentParser(description="Test trained model")
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint (.pth)")
    
    # Optional configuration overrides (config from checkpoint is used by default)
    parser.add_argument("--config", type=str, 
                       help="Path to YAML config file (overrides checkpoint config)")
    parser.add_argument("--experiment", type=str, 
                       help="Name of registered experiment (overrides checkpoint config)")
    parser.add_argument("--model_name", type=str, 
                       help="Name of registered model (overrides checkpoint config)")
    
    # Test options
    parser.add_argument("--batch_size", type=int, help="Batch size for testing")
    parser.add_argument("--num_workers", type=int, help="Number of data loading workers")
    parser.add_argument("--save_predictions", action="store_true",
                       help="Save prediction images")
    parser.add_argument("--output_dir", type=str,
                       help="Output directory for test results (default: checkpoint directory)")
    
    # Distributed options
    parser.add_argument("--distributed", action="store_true",
                       help="Use distributed testing")
    
    return parser.parse_args()

def load_config_for_testing(args):
    """Load configuration for testing based on arguments."""
    # Priority:
    # 1. Explicit config file (--config)
    # 2. Explicit experiment + model_name (--experiment --model_name)
    # 3. Config from checkpoint (default)
    
    if args.config:
        # Explicit config file overrides everything
        config = load_config(config_path=args.config)
        print(f"Using explicit config file: {args.config}")
    elif args.experiment:
        # Explicit experiment overrides checkpoint config
        if args.model_name:
            config = load_config(experiment_name=args.experiment, model_name=args.model_name)
            print(f"Using explicit experiment: {args.experiment} with model: {args.model_name}")
        else:
            config = load_config(experiment_name=args.experiment)
            print(f"Using explicit experiment: {args.experiment}")
    else:
        # Default: Load config from checkpoint
        try:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            if 'config' in checkpoint:
                config = checkpoint['config']
                print("Using configuration from checkpoint")
            else:
                raise ValueError("No configuration found in checkpoint. Please provide --config or --experiment.")
        except Exception as e:
            raise ValueError(f"Could not load config from checkpoint: {e}. Please provide --config or --experiment.")
    
    # Apply command line overrides
    if args.batch_size is not None:
        # Ensure we have a training config for batch size
        if not hasattr(config, 'training') or config.training is None:
            from config.base import TrainingConfig
            config.training = TrainingConfig()
        
        if not hasattr(config.training, 'batch_size'):
            config.training.batch_size = args.batch_size
        else:
            original_batch_size = config.training.batch_size
            config.training.batch_size = args.batch_size
            print(f"Overriding batch size: {original_batch_size} → {args.batch_size}")
    
    if args.num_workers is not None:
        original_num_workers = config.data.num_workers if hasattr(config.data, 'num_workers') else None
        config.data.num_workers = args.num_workers
        print(f"Overriding num_workers: {original_num_workers} → {args.num_workers}")
    
    return config

def main():
    """Main testing function."""
    args = parse_test_args()
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup distributed testing if requested
    if args.distributed:
        device, rank, world_size, local_rank = setup_distributed()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rank, world_size, local_rank = 0, 1, 0
    
    try:
        # Load configuration
        config = load_config_for_testing(args)
        config.model.encoder.name = config.model.encoder.name.replace('lora_', '') if 'lora' in config.model.encoder.name else config.model.encoder.name
        # Set random seeds for reproducibility
        init_seeds(config.seed, rank)
        
        if is_main_process():
            print(f"Testing with configuration: {config.experiment_name}")
            print(f"Model: {config.model.encoder.name} + {config.model.decoder.name}")
            print(f"Dataset: {config.data.dataset_name}")
            print(f"Checkpoint: {args.checkpoint}")
            print(f"Device: {device}")
            if args.distributed:
                print(f"Distributed testing: {world_size} GPU(s), rank {rank}")
        
        # Create the model
        model = create_model(config)
        model = model.to(device)
        
        # Print model summary only on main process
        if is_main_process():
            try:
                summary(model)
                print("Model created successfully!")
                print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
            except Exception as e:
                print(f"Could not print model summary: {e}")
        
        # Get data loaders - only need test loader
        _, _, test_loader = get_data_loaders(
            config, 
            distributed=args.distributed,
            rank=rank,
            world_size=world_size
        )
        
        if test_loader is None:
            raise ValueError("No test dataset found. Please check your dataset configuration.")
        
        if is_main_process():
            print(f"Test dataset size: {len(test_loader.dataset)}")
            print(f"Test batches: {len(test_loader)}")
        
        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            # Use checkpoint directory
            checkpoint_path = Path(args.checkpoint)
            output_dir = checkpoint_path.parent.parent  # Go up from checkpoints/ to run_*/
        
        resume_from = checkpoint_path
        # Create trainer for testing
        trainer = Trainer(
            model=model,
            config=config,
            test_loader=test_loader,
            device=device,
            rank=rank,
            world_size=world_size,
            resume_from=resume_from
        )
        # trainer.output_dir = output_dir
        
        # Load checkpoint
        if is_main_process():
            print(f"Loading checkpoint: {args.checkpoint}")
        # trainer._load_checkpoint(Path(args.checkpoint))
        
        # Run test evaluation
        # if is_main_process():
        #     print("Starting test evaluation...")
        
        test_results = trainer.test()
        
        if is_main_process():
            print("Testing completed successfully!")
            test_metrics = test_results['test_metrics']
            print(f"\nFinal Test Results:")
            print(f"  Test Loss: {test_metrics.get('test_loss', 'N/A'):.4f}")
            print(f"  Pixel Accuracy: {test_metrics.get('pixel_accuracy', 'N/A'):.4f}")
            print(f"  Mean IoU: {test_metrics.get('mean_iou', 'N/A'):.4f}")
            print(f"  Test Time: {test_results['test_time']:.2f} seconds")
            
            # Save additional information
            info_file = output_dir / "test_results" / "test_info.txt"
            os.makedirs(info_file.parent, exist_ok=True)
            with open(info_file, 'w') as f:
                f.write(f"Test Information\n")
                f.write(f"================\n\n")
                f.write(f"Checkpoint: {args.checkpoint}\n")
                f.write(f"Model: {config.model.encoder.name} + {config.model.decoder.name}\n")
                f.write(f"Dataset: {config.data.dataset_name}\n")
                f.write(f"Test samples: {len(test_loader.dataset)}\n")
                f.write(f"Device: {device}\n")
                if args.distributed:
                    f.write(f"Distributed: {world_size} GPU(s)\n")
                f.write(f"Test time: {test_results['test_time']:.2f} seconds\n")
    
    finally:
        # Clean up distributed testing
        if args.distributed:
            cleanup_distributed()

if __name__ == "__main__":
    main()