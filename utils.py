import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import json
import time
from typing import List, Dict, Tuple, Optional

def create_directories(dirs: List[str]) -> None:
    """
    Create directories if they don't exist
    
    Args:
        dirs: List of directory paths to create
    """
    try:
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Created directory: {dir_path}")
    except Exception as e:
        print(f"Error creating directories: {e}")

def save_config(config: Dict, path: str) -> None:
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        path: Path to save the configuration
    """
    try:
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {path}")
    except Exception as e:
        print(f"Error saving configuration: {e}")

def load_config(path: str) -> Dict:
    """
    Load configuration from JSON file
    
    Args:
        path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(path, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded from {path}")
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {}

def draw_text_boxes(image: np.ndarray, boxes: List[Dict], 
                   color: Tuple[int, int, int] = (0, 255, 0), 
                   thickness: int = 2) -> np.ndarray:
    """
    Draw text detection boxes on image
    
    Args:
        image: Input image
        boxes: List of detection boxes with 'bbox' and 'confidence' keys
        color: BGR color for boxes
        thickness: Line thickness
        
    Returns:
        Image with drawn boxes
    """
    try:
        result = image.copy()
        for box in boxes:
            x1, y1, x2, y2 = box['bbox']
            confidence = box.get('confidence', 0.0)
            
            # Draw rectangle
            cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
            # Draw confidence score
            text = f"{confidence:.2f}"
            cv2.putText(result, text, (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result
    except Exception as e:
        print(f"Error drawing text boxes: {e}")
        return image

def visualize_predictions(image: np.ndarray, shrink_map: np.ndarray, 
                         threshold_map: np.ndarray, boxes: List[Dict],
                         save_path: Optional[str] = None) -> None:
    """
    Visualize model predictions with heatmaps and boxes
    
    Args:
        image: Original image
        shrink_map: Predicted shrink map
        threshold_map: Predicted threshold map
        boxes: Detected text boxes
        save_path: Path to save visualization
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original image with boxes
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image with Detections')
        axes[0, 0].axis('off')
        
        # Draw boxes on original image
        for box in boxes:
            x1, y1, x2, y2 = box['bbox']
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='red', facecolor='none')
            axes[0, 0].add_patch(rect)
        
        # Shrink map
        axes[0, 1].imshow(shrink_map, cmap='hot')
        axes[0, 1].set_title('Shrink Map')
        axes[0, 1].axis('off')
        
        # Threshold map
        axes[1, 0].imshow(threshold_map, cmap='hot')
        axes[1, 0].set_title('Threshold Map')
        axes[1, 0].axis('off')
        
        # Binary map (thresholded shrink map)
        binary_map = (shrink_map > 0.3).astype(np.float32)
        axes[1, 1].imshow(binary_map, cmap='gray')
        axes[1, 1].set_title('Binary Map')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error in visualization: {e}")

def calculate_metrics(predictions: List[Dict], ground_truth: List[Dict], 
                     iou_threshold: float = 0.5) -> Dict:
    """
    Calculate precision, recall, and F1-score
    
    Args:
        predictions: List of predicted boxes
        ground_truth: List of ground truth boxes
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dictionary with metrics
    """
    try:
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives
        
        # Match predictions with ground truth
        matched_gt = set()
        matched_pred = set()
        
        for i, pred_box in enumerate(predictions):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(ground_truth):
                if j in matched_gt:
                    continue
                
                iou = calculate_iou(pred_box['bbox'], gt_box['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                tp += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(i)
            else:
                fp += 1
        
        # Count unmatched ground truth as false negatives
        fn = len(ground_truth) - len(matched_gt)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {'precision': 0, 'recall': 0, 'f1_score': 0, 'tp': 0, 'fp': 0, 'fn': 0}

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union between two boxes
    
    Args:
        box1, box2: Boxes in format [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    try:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
        
    except Exception as e:
        print(f"Error calculating IoU: {e}")
        return 0

def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Input image
        target_size: Target size (width, height)
        keep_aspect_ratio: Whether to keep aspect ratio
        
    Returns:
        Resized image
    """
    try:
        if keep_aspect_ratio:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h))
            
            # Create padded image
            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return padded
        else:
            return cv2.resize(image, target_size)
            
    except Exception as e:
        print(f"Error resizing image: {e}")
        return image

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    try:
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image
    except Exception as e:
        print(f"Error normalizing image: {e}")
        return image

def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Denormalize image from [0, 1] to [0, 255] range
    
    Args:
        image: Input normalized image
        
    Returns:
        Denormalized image
    """
    try:
        if image.max() <= 1.0:
            return (image * 255).astype(np.uint8)
        return image.astype(np.uint8)
    except Exception as e:
        print(f"Error denormalizing image: {e}")
        return image

def get_device_info() -> Dict:
    """
    Get information about available devices
    
    Returns:
        Dictionary with device information
    """
    try:
        info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None
        }
        
        if torch.cuda.is_available():
            info['device_name'] = torch.cuda.get_device_name(0)
            info['memory_allocated'] = torch.cuda.memory_allocated(0)
            info['memory_reserved'] = torch.cuda.memory_reserved(0)
        
        return info
    except Exception as e:
        print(f"Error getting device info: {e}")
        return {}

def print_device_info() -> None:
    """
    Print information about available devices
    """
    try:
        info = get_device_info()
        print("Device Information:")
        print(f"CUDA Available: {info['cuda_available']}")
        if info['cuda_available']:
            print(f"Device Count: {info['device_count']}")
            print(f"Current Device: {info['current_device']}")
            print(f"Device Name: {info['device_name']}")
            print(f"Memory Allocated: {info['memory_allocated'] / 1024**2:.2f} MB")
            print(f"Memory Reserved: {info['memory_reserved'] / 1024**2:.2f} MB")
        else:
            print("Using CPU")
    except Exception as e:
        print(f"Error printing device info: {e}")

def time_function(func):
    """
    Decorator to time function execution
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function with timing
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper 