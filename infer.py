import os
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import Model
from utils.postprocess import process_predictions

def parse_args():
    parser = argparse.ArgumentParser(description='DBNet Inference')
    parser.add_argument('--image_path', type=str, required=True, help='Path to image or directory')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--score_threshold', type=float, default=0.3, help='Detection confidence threshold')
    return parser.parse_args()

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

def get_transform():
    """Get preprocessing transform."""
    return A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])

def load_images(image_path, batch_size):
    """Load images in batches."""
    if os.path.isfile(image_path):
        image_paths = [image_path]
    else:
        image_paths = [os.path.join(image_path, f) for f in os.listdir(image_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    transform = get_transform()
    batches = []
    batch = []
    original_sizes = []
    
    for path in image_paths:
        try:
            # Load and preprocess image
            image = cv2.imread(path)
            if image is None:
                print(f'Error reading image: {path}')
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            original_sizes.append((h, w))
            
            # Apply transform
            transformed = transform(image=image)
            batch.append(transformed['image'])
            
            if len(batch) == batch_size:
                batches.append((torch.stack(batch), original_sizes))
                batch = []
                original_sizes = []
        
        except Exception as e:
            print(f'Error processing image {path}: {e}')
            continue
    
    if batch:
        batches.append((torch.stack(batch), original_sizes))
    
    return batches, image_paths

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
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = load_model(args.weights, device)
    if model is None:
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load images
    batches, image_paths = load_images(args.image_path, args.batch_size)
    if not batches:
        print('No valid images found')
        return
    
    # Process batches
    model.eval()
    current_idx = 0
    
    with torch.no_grad():
        for batch_imgs, original_sizes in tqdm(batches, desc='Processing'):
            try:
                # Move batch to device
                batch_imgs = batch_imgs.to(device)
                
                # Get predictions
                predictions = model(batch_imgs)
                
                # Process each image in batch
                for i, pred in enumerate(predictions):
                    try:
                        # Get original image size
                        orig_h, orig_w = original_sizes[i]
                        
                        # Get boxes and scores
                        boxes, scores = process_predictions(
                            pred,
                            min_size=3,
                            box_thresh=args.score_threshold,
                            unclip_ratio=1.5
                        )
                        
                        # Load original image for visualization
                        image_path = image_paths[current_idx]
                        image = cv2.imread(image_path)
                        
                        # Visualize detections
                        result = visualize_detection(image, boxes, scores, args.score_threshold)
                        
                        # Save result
                        output_path = os.path.join(args.output_dir, os.path.basename(image_path))
                        cv2.imwrite(output_path, result)
                        
                        current_idx += 1
                    
                    except Exception as e:
                        print(f'Error processing prediction {current_idx}: {e}')
                        current_idx += 1
                        continue
            
            except Exception as e:
                print(f'Error processing batch: {e}')
                continue
    
    print(f'Results saved to {args.output_dir}')

if __name__ == '__main__':
    main() 