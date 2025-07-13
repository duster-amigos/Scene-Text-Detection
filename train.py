import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from datasets.icdar2015 import ICDAR2015Dataset
from model import Model
from losses import DBLoss
from utils.metrics import compute_batch_metrics
from utils.postprocess import process_predictions

def parse_args():
    parser = argparse.ArgumentParser(description='Train DBNet')
    parser.add_argument('--data_path', type=str, required=True, help='Path to ICDAR2015 dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.007, help='Initial learning rate')
    parser.add_argument('--max_epochs', type=int, default=1200, help='Maximum epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--finetune', action='store_true', help='Enable fine-tuning mode (lower learning rate)')
    return parser.parse_args()

def build_model_config():
    """Build model configuration as per DBNet paper."""
    return {
        'backbone': {
            'type': 'MobileNetV3',
            'pretrained': True,
            'in_channels': 3
        },
        'neck': {
            'type': 'FPEM_FFM',
            'inner_channels': 256  # As per paper
        },
        'head': {
            'type': 'DBHead',
            'out_channels': 2,
            'k': 50  # Differentiable Binarization parameter
        }
    }

def poly_lr_scheduler(epoch, num_epochs, power=0.9):
    """Polynomial learning rate decay as per paper."""
    return (1 - epoch / num_epochs) ** power

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_metrics = np.zeros(3)  # precision, recall, f1
    num_batches = len(train_loader)
    
    with tqdm(total=num_batches, desc=f'Epoch {epoch}') as pbar:
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

def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, save_path)
    print(f'Checkpoint saved to {save_path}')

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
    
    # Load pre-trained weights if resuming
    start_epoch = 0
    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f'Loaded checkpoint from {args.resume}')
            
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f'Resuming from epoch {start_epoch}')
                
        except Exception as e:
            print(f'Error loading checkpoint: {e}')
            print('Starting training from scratch')
    
    # Adjust learning rate for fine-tuning
    if args.finetune:
        args.learning_rate *= 0.1  # Reduce learning rate for fine-tuning
        print(f'Fine-tuning mode: reduced learning rate to {args.learning_rate}')
    
    # Initialize criterion and optimizer
    criterion = DBLoss().to(device)
    optimizer = SGD(model.parameters(), 
                   lr=args.learning_rate,
                   momentum=0.9,
                   weight_decay=1e-4)
    
    # Load optimizer state if resuming
    if args.resume and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Loaded optimizer state')
    
    # Learning rate scheduler
    scheduler = LambdaLR(optimizer, 
                        lr_lambda=lambda epoch: poly_lr_scheduler(epoch, args.max_epochs))
    
    # Training loop
    best_f1 = 0
    start_time = time.time()
    
    try:
        for epoch in range(start_epoch, args.max_epochs):
            print(f'\nEpoch {epoch + 1}/{args.max_epochs}')
            
            # Train
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
            print(f'Training metrics: {train_metrics}')
            
            # Validate
            val_metrics = validate(model, val_loader, criterion, device)
            print(f'Validation metrics: {val_metrics}')
            
            # Update learning rate
            scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % args.save_interval == 0:
                save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                save_checkpoint(model, optimizer, epoch, val_metrics, save_path)
            
            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                save_path = os.path.join(args.save_dir, 'best_model.pth')
                save_checkpoint(model, optimizer, epoch, val_metrics, save_path)
    
    except KeyboardInterrupt:
        print('Training interrupted. Saving checkpoint...')
        save_path = os.path.join(args.save_dir, 'interrupted_checkpoint.pth')
        save_checkpoint(model, optimizer, epoch, val_metrics, save_path)
    
    total_time = time.time() - start_time
    print(f'Training completed in {total_time / 3600:.2f} hours')

if __name__ == '__main__':
    main() 