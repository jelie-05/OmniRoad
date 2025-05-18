import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import yaml
import datetime
import uuid

from config.base import Config
from utility.metrics import AverageMeter, ConfusionMatrix
from tqdm import tqdm
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import io
from PIL import Image
import torch.nn.functional as F

class Trainer:
    """Trainer class for handling the training loop."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        run_id: Optional[str] = None,
        resume_from: Optional[Union[str, Path]] = None,
        resume_training: bool = False
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Set device
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Create base experiment directory
        self.experiment_dir = Path(f"outputs/{config.experiment_name}")
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Handle resume logic
        self.start_epoch = 0
        self.resume_checkpoint = None

        if resume_from or resume_training:
            self.resume_checkpoint = self._find_checkpoint_to_resume(resume_from, run_id)
            if self.resume_checkpoint:
                # Use the run_id from the checkpoint if resuming
                checkpoint_run_id = self._extract_run_id_from_checkpoint(self.resume_checkpoint)
                if checkpoint_run_id:
                    run_id = checkpoint_run_id
                    print(f"Resuming training from {self.resume_checkpoint}")

        # Generate a unique run ID if not provided
        if run_id is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"{timestamp}"
        
        # Create run-specific directory
        self.output_dir = self.experiment_dir / f"run_{run_id}"
        if not self.resume_checkpoint and self.output_dir.exists():
            print(f"Warning: Run directory {self.output_dir} already exists. Adding unique suffix.")
            import uuid
            self.output_dir = self.experiment_dir / f"run_{run_id}__{uuid.uuid4().hex[:6]}"

        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Saving outputs to {self.output_dir}")

        # Initialize model on device
        self.model = self.model.to(device)

        # Set up optimizer
        self.optimizer = self._create_optimizer()
        
        # Set up learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Set up loss function based on task type
        if hasattr(config.data, 'task_type') and config.data.task_type == 'segmentation':
            # For segmentation tasks
            self.criterion = nn.CrossEntropyLoss(ignore_index=config.data.ignore_index 
                                                if hasattr(config.data, 'ignore_index') else 255)
        else:
            # Default for classification
            self.criterion = nn.CrossEntropyLoss()
        
        # Set up gradient scaler for mixed precision training
        # self.scaler = torch.cuda.amp.GradScaler() if config.training.mixed_precision else None
        
        # Initialize training state variables
        self.best_val_loss = float('inf')
        self.best_miou = 0.0
        self.best_epoch = 0
        self.best_metrics = {}
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'mean_iou': [],
            'pixel_accuracy': [],
            'learning_rate': []
        }
        
        # Resume from checkpoint if specified
        if self.resume_checkpoint:
            self._load_checkpoint(self.resume_checkpoint)
        else:
            # Save initial configuration for new runs
            self.save_run_config(config)

        # Initialize TensorBoard writer
        tensorboard_dir = self.output_dir / "tensorboard"
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        
        # Log model graph if not resuming
        if not self.resume_checkpoint:
            self.log_model_graph()
        
        # For visualization, store class colors
        if hasattr(config.data, 'label_colors_list'):
            self.class_colors = torch.tensor(config.data.label_colors_list, 
                                           dtype=torch.float32) / 255.0
        else:
            # Default colormap if not provided
            self.class_colors = None
    
    def _find_checkpoint_to_resume(self, resume_from: Optional[str], run_id: Optional[str]) -> Optional[Path]:
        """Find the checkpoint to resume from."""
        if resume_from:
            # Direct path to checkpoint specified
            checkpoint_path = Path(resume_from)
            if checkpoint_path.exists():
                return checkpoint_path
            else:
                print(f"Checkpoint not found at {resume_from}")
                return None
        
        # Try to find latest checkpoint in current or specified run
        search_dirs = []
        
        if run_id:
            # Look in specific run directory
            search_dirs.append(self.experiment_dir / f"run_{run_id}")
        else:
            # Look in all run directories, prioritizing the most recent
            search_dirs.extend(sorted(self.experiment_dir.glob("run_*"), reverse=True))
        
        for search_dir in search_dirs:
            if search_dir.exists():
                checkpoints_dir = search_dir / "checkpoints"
                if checkpoints_dir.exists():
                    # Look for best model first, then latest checkpoint
                    best_checkpoint = checkpoints_dir / "best_model.pth"
                    if best_checkpoint.exists():
                        return best_checkpoint
                    
                    # Find latest numbered checkpoint
                    checkpoint_files = list(checkpoints_dir.glob("checkpoint_epoch_*.pth"))
                    if checkpoint_files:
                        # Sort by epoch number
                        latest_checkpoint = max(checkpoint_files, 
                                              key=lambda x: int(x.stem.split('_')[-1]))
                        return latest_checkpoint
        
        print("No checkpoint found to resume from")
        return None

    def _extract_run_id_from_checkpoint(self, checkpoint_path: Path) -> Optional[str]:
        """Extract run_id from checkpoint path."""
        run_dir = checkpoint_path.parent.parent  # Go up from checkpoints/checkpoint.pth to run_*/
        if run_dir.name.startswith("run_"):
            return run_dir.name[4:]  # Remove "run_" prefix
        return None

    def _load_checkpoint(self, checkpoint_path: Path):
        """Load checkpoint and restore training state."""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if 'config' in checkpoint:
                loaded_config = checkpoint['config']
                if loaded_config.experiment_name != self.config.experiment_name:
                    print(f"Warning: Experiment name mismatch!")
                    print(f"  Checkpoint: {loaded_config.experiment_name}")
                    print(f"  Current: {self.config.experiment_name}")
            # Load model state
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if available
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training progress
            self.start_epoch = checkpoint.get('epoch', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_miou = checkpoint.get('best_miou', 0.0)
            self.best_epoch = checkpoint.get('best_epoch', 0)
            self.best_metrics = checkpoint.get('best_metrics', {})
            
            # Load metrics history if available
            if 'metrics_history' in checkpoint:
                self.metrics_history = checkpoint['metrics_history']
            
            print(f"Resumed from epoch {self.start_epoch}")
            print(f"Best validation mIoU so far: {self.best_miou:.4f} (epoch {self.best_epoch + 1})")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch...")
            self.start_epoch = 0

    def save_run_config(self, config):
        """Save experiment configuration as YAML for reproducibility."""
        try:
            # Convert config to dictionary recursively
            config_dict = self._config_to_dict(config)
            
            # Save to YAML file
            config_path = self.output_dir / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
                
            print(f"Configuration saved to {config_path}")
        except Exception as e:
            print(f"Warning: Failed to save configuration: {e}")
    
    def _config_to_dict(self, obj):
        """
        Recursively convert configuration objects to dictionaries.
        Handles nested dataclasses, lists, and primitive types.
        """
        if hasattr(obj, '__dict__'):
            # If it's an object with attributes, convert to dict
            result = {}
            for key, value in vars(obj).items():
                # Skip private attributes (starting with _)
                if not key.startswith('_'):
                    result[key] = self._config_to_dict(value)
            return result
        elif isinstance(obj, (list, tuple)):
            # If it's a list or tuple, convert each element
            return [self._config_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            # If it's already a dict, convert its values
            return {key: self._config_to_dict(value) for key, value in obj.items()}
        elif isinstance(obj, (int, float, str, bool, type(None))):
            # Primitive types can be used as-is
            return obj
        else:
            # For any other types, convert to string
            return str(obj)

    def log_model_graph(self):
        """Log model architecture to TensorBoard."""
        try:
            # Create a sample input based on the first batch
            sample_input = next(iter(self.train_loader))[0][:1].to(self.device)
            self.writer.add_graph(self.model, sample_input)
            print("Model graph added to TensorBoard")
        except Exception as e:
            print(f"Failed to add model graph to TensorBoard: {e}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        opt_config = self.config.training.optimizer
        
        # Get model parameters with separate learning rates for encoder and decoder
        params = []
        
        # Add encoder parameters
        encoder_params = []
        for name, param in self.model.named_parameters():
            if 'encoder' in name and param.requires_grad:
                encoder_params.append(param)
        
        if encoder_params and not self.config.model.encoder.freeze:
            params.append({
                'params': encoder_params,
                'lr': opt_config.learning_rate * 0.1  # Lower LR for pretrained encoder
            })
        
        # Add decoder parameters
        decoder_params = []
        for name, param in self.model.named_parameters():
            if 'decoder' in name and param.requires_grad:
                decoder_params.append(param)
        
        if decoder_params:
            params.append({
                'params': decoder_params,
                'lr': opt_config.learning_rate
            })
        
        # Create optimizer
        if opt_config.name.lower() == 'adam':
            return optim.Adam(
                params,
                betas=getattr(opt_config, 'betas', (0.9, 0.999)),
                weight_decay=opt_config.weight_decay
            )
        elif opt_config.name.lower() == 'adamw':
            return optim.AdamW(
                params,
                betas=getattr(opt_config, 'betas', (0.9, 0.999)),
                weight_decay=opt_config.weight_decay
            )
        elif opt_config.name.lower() == 'sgd':
            return optim.SGD(
                params,
                momentum=getattr(opt_config, 'momentum', 0.9),
                weight_decay=opt_config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.name}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        sched_config = self.config.training.scheduler
        if sched_config is None:
            return None
        
        if sched_config.name.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs - getattr(sched_config, 'warmup_epochs', 0),
                eta_min=getattr(sched_config, 'min_lr', 1e-6)
            )
        elif sched_config.name.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=getattr(sched_config, 'step_size', 30),
                gamma=getattr(sched_config, 'gamma', 0.1)
            )
        elif sched_config.name.lower() == 'multistep':
            return optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=getattr(sched_config, 'milestones', [50, 80]),
                gamma=getattr(sched_config, 'gamma', 0.1)
            )
        elif sched_config.name.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=getattr(sched_config, 'factor', 0.1),
                patience=getattr(sched_config, 'patience', 5),
                min_lr=getattr(sched_config, 'min_lr', 1e-6)
            )
        else:
            return None
    
    # trainer/trainer.py - update the train method
    def _save_metrics_plot(self, metrics_history: Dict[str, List[float]]) -> None:
        """
        Save plots of training and validation metrics.
        
        Args:
            metrics_history: Dictionary of metrics history
        """
        if not metrics_history:
            return
        
        # import matplotlib.pyplot as plt
        
        # Create plots directory
        plots_dir = self.output_dir / "plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot loss
        fig, ax = plt.subplots(figsize=(10, 5))
        epochs = range(1, len(metrics_history['train_loss']) + 1)
        
        ax.plot(epochs, metrics_history['train_loss'], 'b-', label='Training Loss')
        if 'val_loss' in metrics_history and metrics_history['val_loss']:
            ax.plot(epochs, metrics_history['val_loss'], 'r-', label='Validation Loss')
        
        ax.xlabel('Epochs')
        ax.ylabel('Loss')
        ax.title('Training and Validation Loss')
        ax.legend()
        ax.grid(True)

        ax.savefig(plots_dir / "loss_history.png", bbox_inches='tight')
        ax.close(fig)
        
        # Plot mIoU and pixel accuracy
        if 'mean_iou' in metrics_history and metrics_history['mean_iou']:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.figure(figsize=(10, 5))
            ax.plot(epochs, metrics_history['mean_iou'], 'g-', label='Mean IoU')
            
            if 'pixel_accuracy' in metrics_history and metrics_history['pixel_accuracy']:
                ax.plot(epochs, metrics_history['pixel_accuracy'], 'm-', label='Pixel Accuracy')
            
            ax.xlabel('Epochs')
            ax.ylabel('Metric Value')
            ax.title('Validation Metrics')
            ax.legend()
            ax.grid(True)
            ax.savefig(plots_dir / "metrics_history.png", bbox_inches='tight')
            ax.close(fig)
        
        # Plot learning rate
        if 'learning_rate' in metrics_history and metrics_history['learning_rate']:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.figure(figsize=(10, 5))
            ax.plot(epochs, metrics_history['learning_rate'], 'k-', label='Learning Rate')
            ax.xlabel('Epochs')
            ax.ylabel('Learning Rate')
            ax.title('Learning Rate Schedule')
            ax.yscale('log')  # Log scale often better for learning rate
            ax.grid(True)
            ax.savefig(plots_dir / "lr_history.png", bbox_inches='tight')
            ax.close(fig)
            
    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.
        
        Returns:
            Dictionary with training statistics
        """
        print(f"Starting training from epoch {self.start_epoch + 1}")
        start_time = time.time()
        
        # Initialize tracking variables
        # self.best_val_loss = float('inf')
        # self.best_miou = 0.0
        # self.best_epoch = 0
        # self.best_metrics = {}
        
        # Create metrics history for plotting
        # metrics_history = {
        #     'train_loss': [],
        #     'val_loss': [],
        #     'mean_iou': [],
        #     'pixel_accuracy': [],
        #     'learning_rate': []
        # }
        
        for epoch in range(self.start_epoch, self.config.training.epochs):
            # Training step
            train_metrics = self.train_epoch(epoch)
            self.metrics_history['train_loss'].append(train_metrics['train_loss'])
            
            # Current learning rate
            lr = self.optimizer.param_groups[0]['lr']
            self.metrics_history['learning_rate'].append(lr)
            
            # Validation step
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self.validate_epoch(epoch)
                
                # Update metrics history
                self.metrics_history['val_loss'].append(val_metrics.get('val_loss', 0))
                self.metrics_history['mean_iou'].append(val_metrics.get('mean_iou', 0))
                self.metrics_history['pixel_accuracy'].append(val_metrics.get('pixel_accuracy', 0))
                
                # Check for best model (using mIoU)
                current_miou = val_metrics.get('mean_iou', 0.0)
                if current_miou > self.best_miou:
                    self.best_miou = current_miou
                    self.best_val_loss = val_metrics.get('val_loss', float('inf'))
                    self.best_epoch = epoch
                    self.best_metrics = val_metrics.copy()
                    
                    # Save best model checkpoint with metrics
                    self.save_checkpoint(epoch, is_best=True, metrics=val_metrics)
                
                # Also keep track of best loss
                if val_metrics.get('val_loss', float('inf')) < self.best_val_loss and epoch != self.best_epoch:
                    self.best_val_loss = val_metrics.get('val_loss', float('inf'))
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('val_loss', 0))
                else:
                    self.scheduler.step()
            
            # Save regular checkpoint every 10 epochs or at the end
            if (epoch + 1) % 10 == 0 or epoch == self.config.training.epochs - 1:
                self.save_checkpoint(epoch, metrics=val_metrics)
            
            # Print epoch summary
            print(f"Epoch {epoch + 1}/{self.config.training.epochs} - "
                f"Train loss: {train_metrics['train_loss']:.4f}, "
                f"Val loss: {val_metrics.get('val_loss', 'N/A'):.4f}, "
                f"mIoU: {val_metrics.get('mean_iou', 'N/A'):.4f}, "
                f"LR: {lr:.6f}")
        
        # Final validation
        # if self.val_loader is not None:
        #     final_metrics = self.validate_epoch(self.config.training.epochs - 1)
        # else:
        #     final_metrics = {}
        
        # Save metrics history plot
        self._save_metrics_plot(self.metrics_history)
        
        # Calculate training time
        total_time = time.time() - start_time
        
        # Return final statistics
        return {
            'total_epochs': self.config.training.epochs,
            'resumed_from_epoch': self.start_epoch + 1,
            'best_val_loss': self.best_val_loss,
            'best_miou': self.best_miou,
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'training_time': total_time,
            'metrics_history': self.metrics_history
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        loss_meter = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        cuda_mem = AverageMeter()
        
        start = time.time()
        pbar = tqdm(self.train_loader)
        pbar.set_description(f'[TRAINING] Epoch {epoch + 1}')

        total_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(pbar):
            # Measure data loading time
            data_time.update(time.time() - start)
            
            # Get data
            if isinstance(batch, dict):
                inputs = batch['image'].to(self.device)
                targets = batch['target'].to(self.device)
            else:  # Assume tuple (inputs, targets)
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Mixed precision training
            # if self.config.training.mixed_precision:
            #     with torch.cuda.amp.autocast():
            #         outputs = self.model(inputs)
            #         loss = self.criterion(outputs, targets)
                
            #     # Backward pass with gradient scaling
            #     self.scaler.scale(loss).backward()
                
            #     # Gradient clipping if configured
            #     if hasattr(self.config.training, 'grad_clip_val') and self.config.training.grad_clip_val:
            #         self.scaler.unscale_(self.optimizer)
            #         torch.nn.utils.clip_grad_norm_(
            #             self.model.parameters(), 
            #             self.config.training.grad_clip_val
            #         )
                
            #     # Update weights
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()
            # else:
            # Standard precision training
            outputs = self.model(inputs)
            outputs = torch.nn.functional.interpolate(outputs, size=targets.shape[1:], mode="bilinear", align_corners=False)

            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping if configured
            if hasattr(self.config.training, 'grad_clip_val') and self.config.training.grad_clip_val:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.grad_clip_val
                )
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            loss_meter.update(loss.item(), inputs.size(0))
            batch_time.update(time.time() - start)
            cuda_mem.update(torch.cuda.max_memory_allocated(device=None) / (1024 * 1024 * 1024))
            # Log progress every 10 batches
            # if batch_idx % 10 == 0:
            #     print(f"Epoch: [{epoch + 1}][{batch_idx}/{len(self.train_loader)}] "
            #           f"Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
            #           f"Time: {batch_time.val:.2f}s ({batch_time.avg:.2f}s) "
            #           f"Data: {data_time.val:.2f}s ({data_time.avg:.2f}s)")
            
            monitor = {
                'Loss': f'{loss_meter.val:.4f} ({loss_meter.avg:.4f})',
                'CUDA': f'{cuda_mem.val:.2f} ({cuda_mem.avg:.2f})'
            }

            start = time.time()

            if batch_idx % 10 == 0:
                global_step = epoch * total_batches + batch_idx
                self.writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                
                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning_Rate', current_lr, global_step)
                
                # Log GPU memory usage
                # if torch.cuda.is_available():
                #     mem_allocated = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
                #     self.writer.add_scalar('Resources/GPU_Memory_MB', mem_allocated, global_step)
            
            # pbar.set_postfix({'CUDA': f'{(torch.cuda.max_memory_allocated(device=None) / (1024 * 1024 * 1024)):.1f} GB'})
            pbar.set_postfix(monitor)
        # Log average epoch loss
        self.writer.add_scalar('Loss/train', loss_meter.avg, epoch)
        return {'train_loss': loss_meter.avg}
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        loss_meter = AverageMeter()
        cuda_mem = AverageMeter()

        # Initialize confusion matrix for IoU calculation
        confusion_matrix = ConfusionMatrix(self.config.data.num_classes, self.config.data.ignore_index)
        
        # Create storage for visualizing a few samples
        num_samples_to_visualize = min(4, self.config.training.batch_size)
        samples_visualized = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader)
            pbar.set_description(f'[VALIDATION] Epoch {epoch + 1}')

            for batch_idx, batch in enumerate(pbar):
                # Get data
                if isinstance(batch, dict):
                    inputs = batch['image'].to(self.device)
                    targets = batch['target'].to(self.device)
                else:  # Assume tuple (inputs, targets)
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                
                # Run forward pass
                # if self.config.training.mixed_precision:
                #     with torch.cuda.amp.autocast():
                #         outputs = self.model(inputs)
                #         outputs = torch.nn.functional.interpolate(outputs, size=targets.shape[1:], mode="bilinear", align_corners=False)

                #         loss = self.criterion(outputs, targets)
                        
                # else:
                outputs = self.model(inputs)
                outputs = torch.nn.functional.interpolate(outputs, size=targets.shape[1:], mode="bilinear", align_corners=False)

                loss = self.criterion(outputs, targets)
                
                # Update metrics
                loss_meter.update(loss.item(), inputs.size(0))
                cuda_mem.update(torch.cuda.max_memory_allocated(device=None) / (1024 * 1024 * 1024))
                # Update confusion matrix for IoU calculation
                confusion_matrix.update(outputs, targets)
                monitor = {
                    'Loss': f'{loss_meter.val:.4f} ({loss_meter.avg:.4f})',
                    'CUDA': f'{cuda_mem.val:.2f} ({cuda_mem.avg:.2f})'
                }
                # pbar.set_postfix({'CUDA': f'{(torch.cuda.max_memory_allocated(device=None) / (1024 * 1024 * 1024)):.1f} GB'})
                pbar.set_postfix(monitor)

                try:
                    # Visualize a few samples
                    if samples_visualized < num_samples_to_visualize:
                        # Select a random batch sample
                        sample_idx = np.random.randint(0, inputs.shape[0])
                        self.log_segmentation_sample(
                            inputs[sample_idx], 
                            outputs[sample_idx], 
                            targets[sample_idx],
                            epoch, 
                            f"sample_{samples_visualized}"
                        )
                        samples_visualized += 1
                except:
                    print("Skipping writing sample image to tensorboard because of an exception.")

        # Compute IoU and other metrics
        metrics = confusion_matrix.compute()
        
        # Print detailed validation results
        print(f"Validation Epoch: [{epoch + 1}]")
        print(f"  Loss: {loss_meter.avg:.4f}")
        print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        print(f"  Mean Accuracy: {metrics['mean_accuracy']:.4f}")
        print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"  Freq Weighted IoU: {metrics['fw_iou']:.4f}")
        
        # Print IoU for each class
        class_names = self.config.data.class_names if hasattr(self.config.data, 'class_names') else None
        class_iou = metrics['iou']
        print("  IoU per class:")
        for i, iou in enumerate(class_iou):
            class_name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
            print(f"    {class_name}: {iou:.4f}")
        
        # Log final metrics
        self.writer.add_scalar('Loss/val', loss_meter.avg, epoch)
        self.writer.add_scalar('Metrics/pixel_accuracy', metrics['pixel_accuracy'], epoch)
        self.writer.add_scalar('Metrics/mean_accuracy', metrics['mean_accuracy'], epoch)
        self.writer.add_scalar('Metrics/mean_iou', metrics['mean_iou'], epoch)
        self.writer.add_scalar('Metrics/fw_iou', metrics['fw_iou'], epoch)

        # Log per-class IoU as a bar chart
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        iou_per_class = metrics['iou']
        class_names = self.config.data.class_names if hasattr(self.config.data, 'class_names') else [f"Class {i}" for i in range(len(iou_per_class))]
        ax.bar(np.arange(len(iou_per_class)), iou_per_class)
        ax.set_xticks(np.arange(len(iou_per_class)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('IoU')
        ax.set_title(f'IoU per Class (Epoch {epoch+1})')
        plt.tight_layout()
        
        # Convert plt figure to image and add to TensorBoard
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        iou_plot = Image.open(buf)
        plt.close(fig)
        self.writer.add_image('Charts/IoU_per_class', np.array(iou_plot).transpose(2, 0, 1), epoch)
        buf.close()  # Close the buffer too
        # Return combined metrics
        return {
            'val_loss': loss_meter.avg,
            'pixel_accuracy': metrics['pixel_accuracy'],
            'mean_accuracy': metrics['mean_accuracy'],
            'mean_iou': metrics['mean_iou'],
            'fw_iou': metrics['fw_iou']
        }
    
    def log_segmentation_sample(self, image, prediction, target, epoch, tag):
        """
        Log segmentation sample to TensorBoard.
        
        Args:
            image: Input image tensor [C, H, W]
            prediction: Prediction tensor [C, H, W]
            target: Target tensor [H, W]
            epoch: Current epoch
            tag: Sample tag
        """
        # Create a figure with 3 subplots: input, prediction, ground truth
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Denormalize image
        mean = torch.tensor(self.config.data.mean).view(3, 1, 1).to(image.device)
        std = torch.tensor(self.config.data.std).view(3, 1, 1).to(image.device)
        image_denorm = image * std + mean
        image_np = image_denorm.cpu().numpy().transpose(1, 2, 0)
        image_np = np.clip(image_np, 0, 1)
        
        # Get predicted class indices
        if prediction.dim() == 3:  # [C, H, W]
            predicted_mask = torch.argmax(prediction, dim=0).cpu().numpy()
        else:  # [H, W]
            predicted_mask = prediction.cpu().numpy()
        
        # Get target mask
        target_mask = target.cpu().numpy()
        
        # Show original image
        ax1.imshow(image_np)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Create a colored prediction mask
        if self.class_colors is not None:
            # Create RGB prediction using class colors
            pred_colored = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3))
            for class_idx in range(len(self.class_colors)):
                class_mask = predicted_mask == class_idx
                pred_colored[class_mask] = self.class_colors[class_idx].cpu().numpy()
                
            # Show prediction as overlay
            ax2.imshow(image_np)
            ax2.imshow(pred_colored, alpha=0.5)
            ax2.set_title("Prediction Overlay")
            ax2.axis('off')
            
            # Create RGB ground truth using class colors
            target_colored = np.zeros((target_mask.shape[0], target_mask.shape[1], 3))
            for class_idx in range(len(self.class_colors)):
                class_mask = target_mask == class_idx
                target_colored[class_mask] = self.class_colors[class_idx].cpu().numpy()
                
            # Show ground truth as overlay
            ax3.imshow(image_np)
            ax3.imshow(target_colored, alpha=0.5)
            ax3.set_title("Ground Truth Overlay")
            ax3.axis('off')
        else:
            # Fallback if no color map is provided
            ax2.imshow(predicted_mask, cmap='nipy_spectral')
            ax2.set_title("Prediction")
            ax2.axis('off')
            
            ax3.imshow(target_mask, cmap='nipy_spectral')
            ax3.set_title("Ground Truth")
            ax3.axis('off')
        
        # Save plot to TensorBoard
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        sample_img = Image.open(buf)
        self.writer.add_image(f'Samples/{tag}', np.array(sample_img).transpose(2, 0, 1), epoch)
        plt.close(fig)
        
        # Also add the segmentation prediction as image
        if self.class_colors is not None:
            self.writer.add_image(
                f'Masks/{tag}_pred', 
                torch.from_numpy(pred_colored.transpose(2, 0, 1)), 
                epoch
            )
            self.writer.add_image(
                f'Masks/{tag}_gt', 
                torch.from_numpy(target_colored.transpose(2, 0, 1)), 
                epoch
            )

    # trainer/trainer.py - update save_checkpoint method

    def save_checkpoint(self, epoch: int, is_best: bool = False, metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Save a checkpoint of the model.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            metrics: Dictionary of validation metrics to save
        """
        # Create checkpoint directory
        checkpoint_dir = self.output_dir / "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        
        
        # Get model state dict (handle DDP or DP)
        if hasattr(self.model, 'module'):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        # Create checkpoint with comprehensive information
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_miou': self.best_miou,
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'config': self.config,
            'metrics_history': self.metrics_history,
        }
        
        # Add scheduler state if available
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Add current metrics if provided
        if metrics:
            checkpoint['current_metrics'] = metrics

        # Format checkpoint path
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Add metrics if provided
        # if metrics:
        #     checkpoint['metrics'] = metrics
            
        #     # Also save metrics to a separate JSON file for easy analysis
        #     metrics_dir = self.output_dir / "metrics"
        #     os.makedirs(metrics_dir, exist_ok=True)
            
        #     import json
        #     # Convert any non-serializable values (like numpy types) to standard Python types
        #     serializable_metrics = {}
        #     for k, v in metrics.items():
        #         if isinstance(v, (np.float32, np.float64)):
        #             serializable_metrics[k] = float(v)
        #         elif isinstance(v, (np.int32, np.int64)):
        #             serializable_metrics[k] = int(v)
        #         elif isinstance(v, np.ndarray):
        #             serializable_metrics[k] = v.tolist()
        #         else:
        #             serializable_metrics[k] = v
            
        #     metrics_path = metrics_dir / f"metrics_epoch_{epoch + 1}.json"
        #     with open(metrics_path, 'w') as f:
        #         json.dump(serializable_metrics, f, indent=2)
        
        # If this is the best model, create a copy
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model checkpoint to {best_path}")
            
            # Also save a text file with the best metrics for quick reference
            if metrics:
                best_metrics_path = self.output_dir / "best_metrics.txt"
                with open(best_metrics_path, 'w') as f:
                    f.write(f"Best model (epoch {epoch + 1}):\n")
                    for k, v in metrics.items():
                        f.write(f"{k}: {v}\n")
        # Save latest checkpoint (for easy resuming)
        latest_path = checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)