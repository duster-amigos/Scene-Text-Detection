import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
from tqdm import tqdm
import numpy as np
from src.models.model import Model
from src.data.icdar2015_dataset import ICDAR2015Dataset, get_transforms
from src.models.losses import DBLoss
import json
from src.utils.logger import logger, print_device_info, print_model_summary, print_config_summary, print_training_progress, print_dataset_info, print_checkpoint_info

torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, config):
        logger.header("Initializing DBNet Trainer")
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.gpu_info(f"Using device: {self.device}")
        
        # Print configuration summary
        print_config_summary(config)
        
        # Create model
        try:
            logger.section("Model Creation")
            self.model = Model(config['model']).to(self.device)
            print_model_summary(self.model)
            logger.end_section()
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
        
        # Create loss function
        try:
            logger.section("Loss Function")
            self.criterion = DBLoss(**config['loss'])
            logger.success("Loss function created successfully")
            logger.end_section()
        except Exception as e:
            logger.error(f"Error creating loss function: {e}")
            raise
        
        # Create optimizer
        try:
            logger.section("Optimizer")
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
            logger.success("Optimizer created successfully")
            logger.end_section()
        except Exception as e:
            logger.error(f"Error creating optimizer: {e}")
            raise
        
        # Create scheduler
        try:
            logger.section("Learning Rate Scheduler")
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config['training']['lr_step_size'],
                gamma=config['training']['lr_gamma']
            )
            logger.success("Scheduler created successfully")
            logger.end_section()
        except Exception as e:
            logger.error(f"Error creating scheduler: {e}")
            raise
        
        # Create datasets and dataloaders
        try:
            logger.section("Data Loading")
            self._create_dataloaders()
            logger.success("Data loaders created successfully")
            logger.end_section()
        except Exception as e:
            logger.error(f"Error creating data loaders: {e}")
            raise
        
        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')
        
        # Load checkpoint if exists
        if config['training']['resume'] and os.path.exists(config['training']['checkpoint_path']):
            logger.checkpoint_info("Resuming from checkpoint")
            self._load_checkpoint()
        
        logger.success("Trainer initialization completed successfully!")
    
    def _create_dataloaders(self):
        """Create training and validation data loaders"""
        try:
            logger.subheader("Creating Data Transforms")
            
            # Training transforms
            train_transform = get_transforms(
                image_size=self.config['training']['image_size'],
                is_training=True
            )
            logger.success("Training transforms created")
            
            # Validation transforms
            val_transform = get_transforms(
                image_size=self.config['training']['image_size'],
                is_training=False
            )
            logger.success("Validation transforms created")
            
            logger.subheader("Loading Training Dataset")
            # Training dataset
            self.train_dataset = ICDAR2015Dataset(
                data_dir=self.config['data']['train_images'],
                gt_dir=self.config['data']['train_labels'],
                transform=train_transform,
                is_training=True
            )
            print_dataset_info(self.train_dataset, "Training")
            
            # Validation dataset (only if validation paths are provided)
            if (self.config['data']['val_images'] and 
                self.config['data']['val_labels'] and 
                os.path.exists(self.config['data']['val_images']) and 
                os.path.exists(self.config['data']['val_labels'])):
                
                logger.subheader("Loading Validation Dataset")
                self.val_dataset = ICDAR2015Dataset(
                    data_dir=self.config['data']['val_images'],
                    gt_dir=self.config['data']['val_labels'],
                    transform=val_transform,
                    is_training=False
                )
                print_dataset_info(self.val_dataset, "Validation")
            else:
                logger.warning("No validation data provided or validation directories don't exist. Training without validation.")
                self.val_dataset = None
            
            logger.subheader("Creating Data Loaders")
            # Data loaders
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=self.config['training']['num_workers'],
                pin_memory=True
            )
            logger.success("Training data loader created")
            
            if self.val_dataset is not None:
                self.val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=self.config['training']['batch_size'],
                    shuffle=False,
                    num_workers=self.config['training']['num_workers'],
                    pin_memory=True
                )
                logger.success("Validation data loader created")
            else:
                self.val_loader = None
                logger.info("No validation loader created")
            
        except Exception as e:
            logger.error(f"Error creating data loaders: {e}")
            raise
    
    def _load_checkpoint(self):
        """Load training checkpoint"""
        try:
            checkpoint = torch.load(self.config['training']['checkpoint_path'], map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            print(f"Loaded checkpoint from epoch {self.start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Save training checkpoint"""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_loss': self.best_loss,
                'config': self.config
            }
            
            # Save regular checkpoint
            torch.save(checkpoint, self.config['training']['checkpoint_path'])
            
            # Save best model
            if is_best:
                best_path = self.config['training']['checkpoint_path'].replace('.pth', '_best.pth')
                torch.save(checkpoint, best_path)
                print(f"Saved best model to {best_path}")
                
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_shrink_loss = 0.0
        total_threshold_loss = 0.0
        total_binary_loss = 0.0
        
        try:
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Move data to device
                    images = batch['image'].to(self.device)
                    shrink_maps = batch['shrink_map'].to(self.device)
                    shrink_masks = batch['shrink_mask'].to(self.device)
                    threshold_maps = batch['threshold_map'].to(self.device)
                    threshold_masks = batch['threshold_mask'].to(self.device)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    predictions = self.model(images)
                    
                    # Prepare batch for loss computation
                    loss_batch = {
                        'shrink_map': shrink_maps,
                        'shrink_mask': shrink_masks,
                        'threshold_map': threshold_maps,
                        'threshold_mask': threshold_masks
                    }
                    
                    # Compute loss
                    loss_dict = self.criterion(predictions, loss_batch)
                    loss = loss_dict['loss']
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    # Update metrics
                    total_loss += loss.item()
                    total_shrink_loss += loss_dict['loss_shrink_maps'].item()
                    total_threshold_loss += loss_dict['loss_threshold_maps'].item()
                    if 'loss_binary_maps' in loss_dict:
                        total_binary_loss += loss_dict['loss_binary_maps'].item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'Loss': f"{loss.item():.4f}",
                        'Shrink': f"{loss_dict['loss_shrink_maps'].item():.4f}",
                        'Thresh': f"{loss_dict['loss_threshold_maps'].item():.4f}"
                    })
                    
                except Exception as e:
                    print(f"Error in training batch {batch_idx}: {e}")
                    continue
            
            # Calculate average losses
            num_batches = len(self.train_loader)
            avg_loss = total_loss / num_batches
            avg_shrink_loss = total_shrink_loss / num_batches
            avg_threshold_loss = total_threshold_loss / num_batches
            avg_binary_loss = total_binary_loss / num_batches if total_binary_loss > 0 else 0
            
            print(f"Training - Epoch {epoch}: Loss={avg_loss:.4f}, Shrink={avg_shrink_loss:.4f}, "
                  f"Thresh={avg_threshold_loss:.4f}, Binary={avg_binary_loss:.4f}")
            
            return avg_loss
            
        except Exception as e:
            print(f"Error in training epoch {epoch}: {e}")
            return float('inf')
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        if self.val_loader is None:
            print("No validation data available, skipping validation")
            return float('inf')
            
        self.model.eval()
        total_loss = 0.0
        total_shrink_loss = 0.0
        total_threshold_loss = 0.0
        
        try:
            with torch.no_grad():
                pbar = tqdm(self.val_loader, desc=f'Validation {epoch}')
                for batch_idx, batch in enumerate(pbar):
                    try:
                        # Move data to device
                        images = batch['image'].to(self.device)
                        shrink_maps = batch['shrink_map'].to(self.device)
                        shrink_masks = batch['shrink_mask'].to(self.device)
                        threshold_maps = batch['threshold_map'].to(self.device)
                        threshold_masks = batch['threshold_mask'].to(self.device)
                        
                        # Forward pass
                        predictions = self.model(images)
                        
                        # Prepare batch for loss computation
                        loss_batch = {
                            'shrink_map': shrink_maps,
                            'shrink_mask': shrink_masks,
                            'threshold_map': threshold_maps,
                            'threshold_mask': threshold_masks
                        }
                        
                        # Compute loss
                        loss_dict = self.criterion(predictions, loss_batch)
                        loss = loss_dict['loss']
                        
                        # Update metrics
                        total_loss += loss.item()
                        total_shrink_loss += loss_dict['loss_shrink_maps'].item()
                        total_threshold_loss += loss_dict['loss_threshold_maps'].item()
                        
                        # Update progress bar
                        pbar.set_postfix({
                            'Loss': f"{loss.item():.4f}",
                            'Shrink': f"{loss_dict['loss_shrink_maps'].item():.4f}",
                            'Thresh': f"{loss_dict['loss_threshold_maps'].item():.4f}"
                        })
                        
                    except Exception as e:
                        print(f"Error in validation batch {batch_idx}: {e}")
                        continue
            
            # Calculate average losses
            num_batches = len(self.val_loader)
            avg_loss = total_loss / num_batches
            avg_shrink_loss = total_shrink_loss / num_batches
            avg_threshold_loss = total_threshold_loss / num_batches
            
            print(f"Validation - Epoch {epoch}: Loss={avg_loss:.4f}, Shrink={avg_shrink_loss:.4f}, "
                  f"Thresh={avg_threshold_loss:.4f}")
            
            return avg_loss
            
        except Exception as e:
            print(f"Error in validation epoch {epoch}: {e}")
            return float('inf')
    
    def train(self):
        """Main training loop"""
        logger.header("Starting DBNet Training")
        
        try:
            for epoch in range(self.start_epoch, self.config['training']['epochs']):
                # Print training progress
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Training
                train_loss = self.train_epoch(epoch + 1)
                
                # Validation (if validation data is available)
                if self.val_loader is not None:
                    val_loss = self.validate_epoch(epoch + 1)
                    
                    # Save checkpoint based on validation loss
                    is_best = val_loss < self.best_loss
                    if is_best:
                        self.best_loss = val_loss
                        logger.success(f"New best validation loss: {self.best_loss:.4f}")
                else:
                    # No validation data, save checkpoint based on training loss
                    val_loss = float('inf')
                    is_best = train_loss < self.best_loss
                    if is_best:
                        self.best_loss = train_loss
                        logger.success(f"New best training loss: {self.best_loss:.4f}")
                
                # Print epoch summary
                print_training_progress(
                    epoch + 1, 
                    self.config['training']['epochs'], 
                    train_loss, 
                    val_loss if self.val_loader is not None else None,
                    current_lr
                )
                
                # Update scheduler
                self.scheduler.step()
                
                # Save checkpoint
                self._save_checkpoint(epoch + 1, is_best)
                
        except Exception as e:
            logger.error(f"Error in training: {e}")
            raise
        
        logger.success("Training completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Train DBNet for text detection')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    args = parser.parse_args()
    
    try:
        logger.header("DBNet Text Detection Training")
        
        # Print device information
        print_device_info()
        
        # Load configuration
        logger.section("Loading Configuration")
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.success(f"Configuration loaded from: {args.config}")
        logger.end_section()
        
        # Create trainer and start training
        trainer = Trainer(config)
        trainer.train()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == '__main__':
    main() 