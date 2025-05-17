import argparse
from pathlib import Path
from typing import Optional, Union, Dict, Any

from .base import Config
from .experiment import ExperimentRegistry
from .encoder import EncoderRegistry
from .decoder import DecoderRegistry
from .model import ModelRegistry
from .data import DataRegistry

def load_config(
    config_path: Optional[Union[str, Path]] = None,
    experiment_name: Optional[str] = None
) -> Config:
    """Load configuration from file or registry."""
    if experiment_name:
        # Load from registry
        experiment_cls = ExperimentRegistry.get(experiment_name)
        return experiment_cls()
    
    elif config_path:
        # Load from YAML file
        return Config.from_yaml(Path(config_path) if isinstance(config_path, str) else config_path)
    
    else:
        raise ValueError("Either config_path or experiment_name must be provided")

def parse_args():
    """Parse command line arguments for configuration."""
    parser = argparse.ArgumentParser(description="Training with configuration")
    
    # Configuration source (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument("--config", type=str, help="Path to YAML config file")
    config_group.add_argument("--experiment", type=str, help="Name of registered experiment")
    
    # List available configurations
    list_group = parser.add_argument_group("List options")
    list_group.add_argument("--list-experiments", action="store_true", 
                       help="List all available registered experiments")
    list_group.add_argument("--list-encoders", action="store_true", 
                       help="List all available encoders")
    list_group.add_argument("--list-decoders", action="store_true", 
                       help="List all available decoders")
    list_group.add_argument("--list-models", action="store_true", 
                       help="List all available models")
    list_group.add_argument("--list-datasets", action="store_true", 
                       help="List all available datasets")
    list_group.add_argument("--list-all", action="store_true", 
                       help="List all available configurations")
    
    # Common overrides
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--freeze", action="store_true", help="Freeze encoder")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    
    args = parser.parse_args()
    
    # List available experiments if requested
    # Handle listing options
    if args.list_experiments:
        print("Available experiments:")
        for name, desc in ExperimentRegistry.list_available().items():
            print(f"  - {name}: {desc}")
        exit(0)
    elif args.list_encoders:
        print("Available encoders:")
        for name, desc in EncoderRegistry.list_available().items():
            print(f"  - {name}: {desc}")
        exit(0)
    elif args.list_decoders:
        print("Available decoders:")
        for name, desc in DecoderRegistry.list_available().items():
            print(f"  - {name}: {desc}")
        exit(0)
    elif args.list_models:
        print("Available models:")
        for name, desc in ModelRegistry.list_available().items():
            print(f"  - {name}: {desc}")
        exit(0)
    elif args.list_datasets:
        print("Available datasets:")
        for name, desc in DataRegistry.list_available().items():
            print(f"  - {name}: {desc}")
        exit(0)
    elif args.list_all:
        print("Available experiments:")
        for name, desc in ExperimentRegistry.list_available().items():
            print(f"  - {name}: {desc}")
        print("\nAvailable models:")
        for name, desc in ModelRegistry.list_available().items():
            print(f"  - {name}: {desc}")
        print("\nAvailable encoders:")
        for name, desc in EncoderRegistry.list_available().items():
            print(f"  - {name}: {desc}")
        print("\nAvailable decoders:")
        for name, desc in DecoderRegistry.list_available().items():
            print(f"  - {name}: {desc}")
        print("\nAvailable datasets:")
        for name, desc in DataConfigRegistry.list_available().items():
            print(f"  - {name}: {desc}")
        exit(0)
    
    return args

def load_config_from_args() -> Config:
    """Load configuration based on command line arguments."""
    args = parse_args()
    
    # Load base configuration
    if args.config:
        config = load_config(config_path=args.config)
    elif args.experiment:
        config = load_config(experiment_name=args.experiment)
    else:
        # This should not happen due to validation in parse_args
        raise ValueError("Either --config or --experiment is required")
    
    # Apply command line overrides
    if args.learning_rate is not None:
        config.training.optimizer.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.freeze:
        config.model.encoder.freeze = True
    
    return config