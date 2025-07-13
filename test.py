import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.icdar2015 import ICDAR2015Dataset
from model import Model
from utils.metrics import compute_batch_metrics
from utils.postprocess import process_predictions

def parse_args():
    parser = argparse.ArgumentParser(description='Test DBNet')
    parser.add_argument('--data_path', type=str, required=True, help='Path to ICDAR2015 test dataset')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    return parser.parse_args()

def load_model(weights_path, device):
    """Load model from checkpoint."""
    try:
        # Build model with same config as training
        model_config = {
            'backbone': {
                'type': 'MobileNetV3',
                'pretrained': False,  # No need for pretrained during testing
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
        model = Model(model_config).to(device)
        
        # Load weights
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Model loaded from {weights_path}')
        return model
    
    except Exception as e:
        print(f'Error loading model: {e}')
        return None

def test(model, test_loader, device):
    """Test the model."""
    model.eval()
    total_metrics = []
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc='Testing'):
            try:
                # Move data to device
                images = images.to(device)
                
                # Forward pass
                predictions = model(images)
                
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
                total_metrics.append([precision, recall, f1])
            
            except Exception as e:
                print(f'Error processing batch: {e}')
                continue
    
    # Calculate final metrics
    total_metrics = torch.tensor(total_metrics)
    mean_metrics = total_metrics.mean(dim=0)
    std_metrics = total_metrics.std(dim=0)
    
    metrics = {
        'precision': {
            'mean': mean_metrics[0].item(),
            'std': std_metrics[0].item()
        },
        'recall': {
            'mean': mean_metrics[1].item(),
            'std': std_metrics[1].item()
        },
        'f1': {
            'mean': mean_metrics[2].item(),
            'std': std_metrics[2].item()
        }
    }
    
    return metrics

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = load_model(args.weights, device)
    if model is None:
        return
    
    # Create dataset and dataloader
    test_dataset = ICDAR2015Dataset(args.data_path, is_training=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Test model
    try:
        metrics = test(model, test_loader, device)
        
        # Print results
        print('\nTest Results:')
        print(f"Precision: {metrics['precision']['mean']:.4f} ± {metrics['precision']['std']:.4f}")
        print(f"Recall: {metrics['recall']['mean']:.4f} ± {metrics['recall']['std']:.4f}")
        print(f"F1 Score: {metrics['f1']['mean']:.4f} ± {metrics['f1']['std']:.4f}")
        
    except Exception as e:
        print(f'Error during testing: {e}')

if __name__ == '__main__':
    main() 