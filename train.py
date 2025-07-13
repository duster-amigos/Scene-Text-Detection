import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
from tqdm import tqdm
import numpy as np
from model import Model
from dataset import ICDAR2015Dataset, get_transforms
from losses import DBLoss
import json

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model
        try:
            self.model = Model(config['model']).to(self.device)
            print(f"Model created: {self.model.name}")
            print(f"Total parameters: {sum(p.numel() for p in self.model.parameters())}")
        except Exception as e:
            print(f"Error creating model: {e}")
            raise
        
        # Create loss function
        try:
            self.criterion = DBLoss(**config['loss'])
            print("Loss function created")
        except Exception as e:
            print(f"Error creating loss function: {e}")
            raise
        
        # Create optimizer
        try:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
            print("Optimizer created")
        except Exception as e:
            print(f"Error creating optimizer: {e}")
            raise
        
        # Create scheduler
        try:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config['training']['lr_step_size'],
                gamma=config['training']['lr_gamma']
            )
            print("Scheduler created")
        except Exception as e:
            print(f"Error creating scheduler: {e}")
            raise
        
        # Create datasets and dataloaders
        try:
            self._create_dataloaders()
            print("Data loaders created")
        except Exception as e:
            print(f"Error creating data loaders: {e}")
            raise
        
        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')
        
        # Load checkpoint if exists
        if config['training']['resume'] and os.path.exists(config['training']['checkpoint_path']):
            self._load_checkpoint()
    
    def _create_dataloaders(self):
        """Create training and validation data loaders"""
        try:
            # Training transforms
            train_transform = get_transforms(
                image_size=self.config['training']['image_size'],
                is_training=True
            )
            
            # Validation transforms
            val_transform = get_transforms(
                image_size=self.config['training']['image_size'],
                is_training=False
            )
            
            # Training dataset
            self.train_dataset = ICDAR2015Dataset(
                data_dir=self.config['data']['train_images'],
                gt_dir=self.config['data']['train_labels'],
                transform=train_transform,
                is_training=True
            )
            
            # Validation dataset
            self.val_dataset = ICDAR2015Dataset(
                data_dir=self.config['data']['val_images'],
                gt_dir=self.config['data']['val_labels'],
                transform=val_transform,
                is_training=False
            )
            
            # Data loaders
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=self.config['training']['num_workers'],
                pin_memory=True
            )
            
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=False,
                num_workers=self.config['training']['num_workers'],
                pin_memory=True
            )
            
            print(f"Training samples: {len(self.train_dataset)}")
            print(f"Validation samples: {len(self.val_dataset)}")
            
        except Exception as e:
            print(f"Error creating data loaders: {e}")
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
        print("Starting training...")
        
        try:
            for epoch in range(self.start_epoch, self.config['training']['epochs']):
                print(f"\nEpoch {epoch + 1}/{self.config['training']['epochs']}")
                
                # Training
                train_loss = self.train_epoch(epoch + 1)
                
                # Validation
                val_loss = self.validate_epoch(epoch + 1)
                
                # Update scheduler
                self.scheduler.step()
                
                # Save checkpoint
                is_best = val_loss < self.best_loss
                if is_best:
                    self.best_loss = val_loss
                    print(f"New best validation loss: {self.best_loss:.4f}")
                
                self._save_checkpoint(epoch + 1, is_best)
                
                # Print learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Learning rate: {current_lr:.6f}")
                
        except Exception as e:
            print(f"Error in training: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Train DBNet for text detection')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    args = parser.parse_args()
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Create trainer and start training
        trainer = Trainer(config)
        trainer.train()
        
    except Exception as e:
        print(f"Error in main: {e}")
        raise

if __name__ == '__main__':
    main() 