#!/usr/bin/env python3
"""
Fine-tuning script for DBNet.
Supports loading pre-trained weights and fine-tuning with different configurations.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR
from tqdm import tqdm

from datasets.icdar2015 import ICDAR2015Dataset
from model import Model
from losses import DBLoss
from utils.metrics import compute_batch_metrics
from utils.postprocess import process_predictions

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune DBNet')
    parser.add_argument('--data_path', type=str, required=True, help='Path to ICDAR2015 dataset')
    parser.add_argument('--pretrained_weights', type=str, required=True, help='Path to pre-trained weights')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Fine-tuning learning rate (lower than training)')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum epochs for fine-tuning')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default='finetune_checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone layers')
    parser.add_argument('--freeze_neck', action='store_true', help='Freeze neck layers')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['poly', 'step', 'cosine'], help='Learning rate scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--finetune_layers', type=str, default='all', 
                       choices=['all', 'head_only', 'neck_head', 'backbone_head'],
                       help='Which layers to fine-tune')
    return parser.parse_args()

def build_model_config():
    """Build model configuration as per DBNet paper."""
    return {
        'backbone': {
            'type': 'MobileNetV3',
            'pretrained': False,  # We'll load our own weights
            'in_channels': 3
        },
        'neck': {
            'type': 'FPEM_FFM',
            'inner_channels': 256
        },
        'head': {
            'type': 'DBHead',
            'out_channels': 2,
            'k': 50
        }
    }

def load_pretrained_model(weights_path, model, device):
    """Load pre-trained weights into model."""
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights
        model.load_state_dict(state_dict, strict=False)
        print(f'Successfully loaded pre-trained weights from {weights_path}')
        
        # Print checkpoint info if available
        if 'epoch' in checkpoint:
            print(f'Pre-trained model was trained for {checkpoint["epoch"]} epochs')
        if 'metrics' in checkpoint:
            print(f'Pre-trained model metrics: {checkpoint["metrics"]}')
            
        return True
        
    except Exception as e:
        print(f'Error loading pre-trained weights: {e}')
        return False

def setup_finetuning(model, args):
    """Setup fine-tuning by freezing/unfreezing layers."""
    print(f'Setting up fine-tuning for: {args.finetune_layers}')
    
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print('Backbone layers frozen')
    
    if args.freeze_neck:
        for param in model.neck.parameters():
            param.requires_grad = False
        print('Neck layers frozen')
    
    # Fine-tune specific layers
    if args.finetune_layers == 'head_only':
        # Freeze backbone and neck
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.neck.parameters():
            param.requires_grad = False
        print('Fine-tuning head only')
        
    elif args.finetune_layers == 'neck_head':
        # Freeze backbone only
        for param in model.backbone.parameters():
            param.requires_grad = False
        print('Fine-tuning neck and head')
        
    elif args.finetune_layers == 'backbone_head':
        # Freeze neck only
        for param in model.neck.parameters():
            param.requires_grad = False
        print('Fine-tuning backbone and head')
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)')

def get_optimizer(model, args):
    """Get optimizer based on arguments."""
    # Filter parameters that require gradients
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    if args.optimizer.lower() == 'adam':
        optimizer = Adam(trainable_params, 
                        lr=args.learning_rate, 
                        weight_decay=args.weight_decay)
        print(f'Using Adam optimizer with lr={args.learning_rate}')
    else:
        optimizer = SGD(trainable_params, 
                       lr=args.learning_rate,
                       momentum=args.momentum,
                       weight_decay=args.weight_decay)
        print(f'Using SGD optimizer with lr={args.learning_rate}, momentum={args.momentum}')
    
    return optimizer

def get_scheduler(optimizer, args):
    """Get learning rate scheduler based on arguments."""
    if args.scheduler == 'poly':
        scheduler = LambdaLR(optimizer, 
                           lr_lambda=lambda epoch: (1 - epoch / args.max_epochs) ** 0.9)
        print('Using polynomial learning rate scheduler')
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        print('Using step learning rate scheduler')
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        print('Using cosine annealing learning rate scheduler')
    
    return scheduler

def finetune_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Fine-tune for one epoch."""
    model.train()
    total_loss = 0
    total_metrics = np.zeros(3)  # precision, recall, f1
    num_batches = len(train_loader)
    
    with tqdm(total=num_batches, desc=f'Fine-tune Epoch {epoch}') as pbar:
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Move data to device
            images = images.to(device)
            for k, v in targets.items():
                targets[k] = v.to(device)
            
            # Forward pass
            predictions = model(images)
            losses = criterion(predictions, targets)
            loss = losses['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Process predictions for metrics
            batch_boxes, batch_scores = [], []
            for pred in predictions.detach().cpu():
                boxes, scores = process_predictions(pred)
                batch_boxes.append(boxes)
                batch_scores.append(scores)
            
            # Get ground truth boxes
            batch_gt_boxes = [target['boxes'].cpu().numpy() for target in targets]
            
            # Compute metrics
            precision, recall, f1 = compute_batch_metrics(
                batch_gt_boxes, batch_boxes, batch_scores
            )
            total_metrics += np.array([precision, recall, f1])
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            avg_metrics = total_metrics / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'prec': f'{avg_metrics[0]:.4f}',
                'rec': f'{avg_metrics[1]:.4f}',
                'f1': f'{avg_metrics[2]:.4f}'
            })
            pbar.update()
    
    # Return average metrics
    return {
        'loss': total_loss / num_batches,
        'precision': total_metrics[0] / num_batches,
        'recall': total_metrics[1] / num_batches,
        'f1': total_metrics[2] / num_batches
    }

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_metrics = np.zeros(3)
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for images, targets in val_loader:
            # Move data to device
            images = images.to(device)
            for k, v in targets.items():
                targets[k] = v.to(device)
            
            # Forward pass
            predictions = model(images)
            losses = criterion(predictions, targets)
            total_loss += losses['loss'].item()
            
            # Process predictions
            batch_boxes, batch_scores = [], []
            for pred in predictions.cpu():
                boxes, scores = process_predictions(pred)
                batch_boxes.append(boxes)
                batch_scores.append(scores)
            
            # Get ground truth boxes
            batch_gt_boxes = [target['boxes'].cpu().numpy() for target in targets]
            
            # Compute metrics
            precision, recall, f1 = compute_batch_metrics(
                batch_gt_boxes, batch_boxes, batch_scores
            )
            total_metrics += np.array([precision, recall, f1])
    
    # Return average metrics
    return {
        'loss': total_loss / num_batches,
        'precision': total_metrics[0] / num_batches,
        'recall': total_metrics[1] / num_batches,
        'f1': total_metrics[2] / num_batches
    }

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path, args):
    """Save fine-tuning checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'finetune_config': {
            'learning_rate': args.learning_rate,
            'optimizer': args.optimizer,
            'scheduler': args.scheduler,
            'finetune_layers': args.finetune_layers,
            'freeze_backbone': args.freeze_backbone,
            'freeze_neck': args.freeze_neck
        }
    }, save_path)
    print(f'Fine-tuning checkpoint saved to {save_path}')

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create datasets and dataloaders
    train_dataset = ICDAR2015Dataset(args.data_path, is_training=True)
    val_dataset = ICDAR2015Dataset(args.data_path, is_training=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Build model
    model_config = build_model_config()
    model = Model(model_config).to(device)
    
    # Load pre-trained weights
    if not load_pretrained_model(args.pretrained_weights, model, device):
        print('Failed to load pre-trained weights. Exiting.')
        return
    
    # Setup fine-tuning configuration
    setup_finetuning(model, args)
    
    # Initialize criterion and optimizer
    criterion = DBLoss().to(device)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    
    # Fine-tuning loop
    best_f1 = 0
    start_time = time.time()
    
    try:
        for epoch in range(args.max_epochs):
            print(f'\nFine-tuning Epoch {epoch + 1}/{args.max_epochs}')
            
            # Fine-tune
            train_metrics = finetune_epoch(model, train_loader, criterion, optimizer, device, epoch)
            print(f'Training metrics: {train_metrics}')
            
            # Validate
            val_metrics = validate(model, val_loader, criterion, device)
            print(f'Validation metrics: {val_metrics}')
            
            # Update learning rate
            if scheduler:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Current learning rate: {current_lr:.6f}')
            
            # Save checkpoint
            if (epoch + 1) % args.save_interval == 0:
                save_path = os.path.join(args.save_dir, f'finetune_epoch_{epoch + 1}.pth')
                save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, save_path, args)
            
            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                save_path = os.path.join(args.save_dir, 'best_finetuned_model.pth')
                save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, save_path, args)
    
    except KeyboardInterrupt:
        print('Fine-tuning interrupted. Saving checkpoint...')
        save_path = os.path.join(args.save_dir, 'interrupted_finetune_checkpoint.pth')
        save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, save_path, args)
    
    total_time = time.time() - start_time
    print(f'Fine-tuning completed in {total_time / 3600:.2f} hours')
    print(f'Best F1 score: {best_f1:.4f}')

if __name__ == '__main__':
    main() 