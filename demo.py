#!/usr/bin/env python3
"""
Demo script for DBNet text detection.
"""

import argparse
import cv2
import numpy as np
import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import Model
from utils.postprocess import process_predictions
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_model(weights_path, device):
    """Load model from checkpoint."""
    try:
        model_config = {
            'backbone': {
                'type': 'MobileNetV3',
                'pretrained': False,
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
        
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Model loaded from {weights_path}')
        return model
    
    except Exception as e:
        print(f'Error loading model: {e}')
        return None

def preprocess_image(image_path):
    """Preprocess image for inference."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_h, original_w = image.shape[:2]
    
    # Apply preprocessing
    transform = A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)
    return transformed['image'].unsqueeze(0), (original_h, original_w), image

def visualize_detection(image, boxes, scores, threshold=0.3):
    """Visualize detection results."""
    image = image.copy()
    
    for box, score in zip(boxes, scores):
        if score < threshold:
            continue
            
        box = box.astype(np.int32)
        cv2.polylines(image, [box], True, (0, 255, 0), 2)
        
        # Add confidence score
        text = f'{score:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x, text_y = box[0][0], box[0][1] - 5
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
    
    return image

def main():
    parser = argparse.ArgumentParser(description='DBNet Demo')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--output_path', type=str, default='output.jpg', help='Path to save output image')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--score_threshold', type=float, default=0.3, help='Detection confidence threshold')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = load_model(args.weights, device)
    if model is None:
        return
    
    try:
        # Preprocess image
        input_tensor, (original_h, original_w), original_image = preprocess_image(args.image_path)
        input_tensor = input_tensor.to(device)
        
        # Run inference
        model.eval()
        with torch.no_grad():
            predictions = model(input_tensor)
        
        # Process predictions
        pred = predictions[0]  # Get first (and only) prediction
        boxes, scores = process_predictions(
            pred,
            min_size=3,
            box_thresh=args.score_threshold,
            unclip_ratio=1.5
        )
        
        # Visualize results
        result_image = visualize_detection(original_image, boxes, scores, args.score_threshold)
        
        # Save result
        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output_path, result_image)
        
        print(f'Detected {len(boxes)} text regions')
        print(f'Results saved to {args.output_path}')
        
    except Exception as e:
        print(f'Error during inference: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 