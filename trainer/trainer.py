import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
from pathlib import Path

from config.base import Config
from utility.metrics import AverageMeter, ConfusionMatrix
from tqdm import tqdm
import numpy as np


class Trainer:
    """Trainer class for handling the training loop."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None
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
        
        # Track best validation metrics
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Create output directory
        self.output_dir = Path(f"outputs/{config.experiment_name}")
        os.makedirs(self.output_dir, exist_ok=True)
    
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
        
        import matplotlib.pyplot as plt
        
        # Create plots directory
        plots_dir = self.output_dir / "plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot loss
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(metrics_history['train_loss']) + 1)
        
        plt.plot(epochs, metrics_history['train_loss'], 'b-', label='Training Loss')
        if 'val_loss' in metrics_history and metrics_history['val_loss']:
            plt.plot(epochs, metrics_history['val_loss'], 'r-', label='Validation Loss')
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / "loss_history.png", bbox_inches='tight')
        plt.close()
        
        # Plot mIoU and pixel accuracy
        if 'mean_iou' in metrics_history and metrics_history['mean_iou']:
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, metrics_history['mean_iou'], 'g-', label='Mean IoU')
            
            if 'pixel_accuracy' in metrics_history and metrics_history['pixel_accuracy']:
                plt.plot(epochs, metrics_history['pixel_accuracy'], 'm-', label='Pixel Accuracy')
            
            plt.xlabel('Epochs')
            plt.ylabel('Metric Value')
            plt.title('Validation Metrics')
            plt.legend()
            plt.grid(True)
            plt.savefig(plots_dir / "metrics_history.png", bbox_inches='tight')
            plt.close()
        
        # Plot learning rate
        if 'learning_rate' in metrics_history and metrics_history['learning_rate']:
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, metrics_history['learning_rate'], 'k-', label='Learning Rate')
            plt.xlabel('Epochs')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.yscale('log')  # Log scale often better for learning rate
            plt.grid(True)
            plt.savefig(plots_dir / "lr_history.png", bbox_inches='tight')
            plt.close()
            
    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.
        
        Returns:
            Dictionary with training statistics
        """
        start_time = time.time()
        
        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.best_miou = 0.0
        self.best_epoch = 0
        self.best_metrics = {}
        
        # Create metrics history for plotting
        metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'mean_iou': [],
            'pixel_accuracy': [],
            'learning_rate': []
        }
        
        for epoch in range(self.config.training.epochs):
            # Training step
            train_metrics = self.train_epoch(epoch)
            metrics_history['train_loss'].append(train_metrics['train_loss'])
            
            # Current learning rate
            lr = self.optimizer.param_groups[0]['lr']
            metrics_history['learning_rate'].append(lr)
            
            # Validation step
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self.validate_epoch(epoch)
                
                # Update metrics history
                metrics_history['val_loss'].append(val_metrics.get('val_loss', 0))
                metrics_history['mean_iou'].append(val_metrics.get('mean_iou', 0))
                metrics_history['pixel_accuracy'].append(val_metrics.get('pixel_accuracy', 0))
                
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
        if self.val_loader is not None:
            final_metrics = self.validate_epoch(self.config.training.epochs - 1)
        else:
            final_metrics = {}
        
        # Save metrics history plot
        self._save_metrics_plot(metrics_history)
        
        # Calculate training time
        total_time = time.time() - start_time
        
        # Return final statistics
        return {
            'total_epochs': self.config.training.epochs,
            'best_val_loss': self.best_val_loss,
            'best_miou': self.best_miou,
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'final_metrics': final_metrics,
            'training_time': total_time,
            'metrics_history': metrics_history
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
            # pbar.set_postfix({'CUDA': f'{(torch.cuda.max_memory_allocated(device=None) / (1024 * 1024 * 1024)):.1f} GB'})
            pbar.set_postfix(monitor)
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
        
        # Return combined metrics
        return {
            'val_loss': loss_meter.avg,
            'pixel_accuracy': metrics['pixel_accuracy'],
            'mean_accuracy': metrics['mean_accuracy'],
            'mean_iou': metrics['mean_iou'],
            'fw_iou': metrics['fw_iou']
        }
    
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
        
        # Format checkpoint path
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        
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
            'best_miou': getattr(self, 'best_miou', 0.0),
            'best_epoch': getattr(self, 'best_epoch', epoch),
            'config': self.config,  # Save configuration for reproducibility
        }
        
        # Add scheduler state if available
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Add metrics if provided
        if metrics:
            checkpoint['metrics'] = metrics
            
            # Also save metrics to a separate JSON file for easy analysis
            metrics_dir = self.output_dir / "metrics"
            os.makedirs(metrics_dir, exist_ok=True)
            
            import json
            # Convert any non-serializable values (like numpy types) to standard Python types
            serializable_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (np.float32, np.float64)):
                    serializable_metrics[k] = float(v)
                elif isinstance(v, (np.int32, np.int64)):
                    serializable_metrics[k] = int(v)
                elif isinstance(v, np.ndarray):
                    serializable_metrics[k] = v.tolist()
                else:
                    serializable_metrics[k] = v
            
            metrics_path = metrics_dir / f"metrics_epoch_{epoch + 1}.json"
            with open(metrics_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
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