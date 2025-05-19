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
    experiment_name: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Config:
    """Load configuration from file or registry."""
    if experiment_name:
        if not model_name:
            raise ValueError("Both experiment_name and model_name must be provided")
        # Load from registry
        experiment_cls = ExperimentRegistry.get(experiment_name)

        return experiment_cls(model_name=model_name)
    
    elif config_path:
        # Load from YAML file
        return Config.from_yaml(Path(config_path) if isinstance(config_path, str) else config_path)
    
    else:
        raise ValueError("Either config_path or experiment_name must be provided")

def parse_args():
    """Parse command line arguments for configuration."""
    parser = argparse.ArgumentParser(description="Training with configuration")
    
    # Configuration source (mutually exclusive)
    # config_group = parser.add_mutually_exclusive_group(False)
    config_group = parser.add_argument_group("Config options")

    config_group.add_argument("--config", type=str, help="Path to YAML config file")
    config_group.add_argument("--experiment", type=str, help="Name of registered experiment")
    config_group.add_argument("--model_name", type=str, help="Name of registered model")
    
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
    
    # Resume training options
    resume_group = parser.add_argument_group("Resume training options")
    resume_group.add_argument("--resume", action="store_true", 
                            help="Resume training from the latest checkpoint")
    resume_group.add_argument("--resume_from", type=str,
                            help="Resume from a specific checkpoint path")
    resume_group.add_argument("--resume_run_id", type=str,
                            help="Resume from the latest checkpoint of a specific run")
                            
    # Common overrides
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--num_workers", type=int, help="Override number of workers")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--freeze", action="store_true", help="Freeze encoder")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--run_id", type=str, 
                        help="Unique run identifier. If not provided, timestamp will be used.")

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
        for name, desc in DataRegistry.list_available().items():
            print(f"  - {name}: {desc}")
        exit(0)
    
    # Custom validation for config options
    if args.config:
        # If --config is provided, --experiment and --model_name should not be used
        if args.experiment or args.model_name:
            parser.error("Cannot use --experiment or --model_name with --config. "
                        "Choose either --config OR (--experiment and --model_name).")
    else:
        # If --config is not provided, both --experiment and --model_name are required
        if not args.experiment:
            parser.error("--experiment is required when not using --config")
        if not args.model_name:
            parser.error("--model_name is required when not using --config")
    
    
    return args

def load_config_from_args() -> Config:
    """Load configuration based on command line arguments."""
    args = parse_args()
    
    # Load base configuration
    if args.config:
        config = load_config(config_path=args.config)
    elif args.experiment:
        config = load_config(experiment_name=args.experiment, model_name=args.model_name)
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
    if args.num_workers:
        config.data.num_workers = args.num_workers
    # if args.run_id:
    #     config.run_id = args.run_id
    
    return config, args