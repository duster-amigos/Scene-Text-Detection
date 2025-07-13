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
from pathlib import Path
import argparse
from typing import Dict, Any, List, Optional, Tuple

class BeautifulLogger:
    """
    Professional logging utility with beautiful formatting and hierarchy
    """
    
    def __init__(self, name: str = "DBNet", level: str = "INFO"):
        self.name = name
        self.level = level
        self.indent_level = 0
        self.start_time = time.time()
        
        # Colors for different log levels
        self.colors = {
            'INFO': '\033[94m',      # Blue
            'SUCCESS': '\033[92m',   # Green
            'WARNING': '\033[93m',   # Yellow
            'ERROR': '\033[91m',     # Red
            'HEADER': '\033[95m',    # Purple
            'SUBHEADER': '\033[96m', # Cyan
            'RESET': '\033[0m'       # Reset
        }
        
        # Unicode symbols for hierarchy
        self.symbols = {
            'start': '🚀',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌',
            'info': 'ℹ️',
            'step': '➤',
            'substep': '  ↳',
            'progress': '📊',
            'time': '⏱️',
            'memory': '💾',
            'gpu': '🎮',
            'model': '🧠',
            'data': '📁',
            'config': '⚙️',
            'checkpoint': '💾',
            'training': '🏋️',
            'validation': '📈',
            'testing': '🧪',
            'inference': '🔍'
        }
    
    def _get_timestamp(self) -> str:
        """Get formatted timestamp"""
        return time.strftime("%H:%M:%S", time.localtime())
    
    def _get_indent(self) -> str:
        """Get current indent string"""
        return "  " * self.indent_level
    
    def _format_message(self, level: str, symbol: str, message: str) -> str:
        """Format message with colors, symbols, and hierarchy"""
        timestamp = self._get_timestamp()
        indent = self._get_indent()
        color = self.colors.get(level, self.colors['INFO'])
        reset = self.colors['RESET']
        
        return f"{color}{timestamp} {symbol} {indent}{message}{reset}"
    
    def header(self, message: str):
        """Print main header"""
        print("\n" + "="*80)
        print(self._format_message('HEADER', self.symbols['start'], f"{self.name}: {message}"))
        print("="*80)
    
    def subheader(self, message: str):
        """Print subheader"""
        print("\n" + "-"*60)
        print(self._format_message('SUBHEADER', self.symbols['info'], message))
        print("-"*60)
    
    def info(self, message: str):
        """Print info message"""
        print(self._format_message('INFO', self.symbols['step'], message))
    
    def success(self, message: str):
        """Print success message"""
        print(self._format_message('SUCCESS', self.symbols['success'], message))
    
    def warning(self, message: str):
        """Print warning message"""
        print(self._format_message('WARNING', self.symbols['warning'], message))
    
    def error(self, message: str):
        """Print error message"""
        print(self._format_message('ERROR', self.symbols['error'], message))
    
    def step(self, message: str):
        """Print step message"""
        print(self._format_message('INFO', self.symbols['step'], message))
    
    def substep(self, message: str):
        """Print substep message"""
        print(self._format_message('INFO', self.symbols['substep'], message))
    
    def progress(self, message: str):
        """Print progress message"""
        print(self._format_message('INFO', self.symbols['progress'], message))
    
    def model_info(self, message: str):
        """Print model-related info"""
        print(self._format_message('INFO', self.symbols['model'], message))
    
    def data_info(self, message: str):
        """Print data-related info"""
        print(self._format_message('INFO', self.symbols['data'], message))
    
    def config_info(self, message: str):
        """Print config-related info"""
        print(self._format_message('INFO', self.symbols['config'], message))
    
    def training_info(self, message: str):
        """Print training-related info"""
        print(self._format_message('INFO', self.symbols['training'], message))
    
    def validation_info(self, message: str):
        """Print validation-related info"""
        print(self._format_message('INFO', self.symbols['validation'], message))
    
    def testing_info(self, message: str):
        """Print testing-related info"""
        print(self._format_message('INFO', self.symbols['testing'], message))
    
    def inference_info(self, message: str):
        """Print inference-related info"""
        print(self._format_message('INFO', self.symbols['inference'], message))
    
    def checkpoint_info(self, message: str):
        """Print checkpoint-related info"""
        print(self._format_message('INFO', self.symbols['checkpoint'], message))
    
    def gpu_info(self, message: str):
        """Print GPU-related info"""
        print(self._format_message('INFO', self.symbols['gpu'], message))
    
    def memory_info(self, message: str):
        """Print memory-related info"""
        print(self._format_message('INFO', self.symbols['memory'], message))
    
    def time_info(self, message: str):
        """Print time-related info"""
        print(self._format_message('INFO', self.symbols['time'], message))
    
    def section(self, title: str):
        """Start a new section with indentation"""
        print(f"\n{self._get_indent()}{self.colors['SUBHEADER']}📋 {title}{self.colors['RESET']}")
        self.indent_level += 1
    
    def end_section(self):
        """End current section and reduce indentation"""
        self.indent_level = max(0, self.indent_level - 1)
    
    def table(self, data: Dict[str, Any], title: str = "Configuration"):
        """Print data in a beautiful table format"""
        print(f"\n{self._get_indent()}{self.colors['SUBHEADER']}📊 {title}{self.colors['RESET']}")
        
        max_key_length = max(len(str(k)) for k in data.keys())
        
        for key, value in data.items():
            key_str = str(key).ljust(max_key_length)
            value_str = str(value)
            print(f"{self._get_indent()}  {self.colors['INFO']}{key_str}{self.colors['RESET']}: {value_str}")
    
    def metrics(self, metrics: Dict[str, float], title: str = "Metrics"):
        """Print metrics in a beautiful format"""
        print(f"\n{self._get_indent()}{self.colors['SUBHEADER']}📈 {title}{self.colors['RESET']}")
        
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                formatted_value = f"{metric_value:.4f}"
            else:
                formatted_value = str(metric_value)
            print(f"{self._get_indent()}  {self.colors['INFO']}{metric_name}{self.colors['RESET']}: {formatted_value}")
    
    def summary(self, data: Dict[str, Any], title: str = "Summary"):
        """Print a summary in a beautiful format"""
        print(f"\n{self._get_indent()}{self.colors['HEADER']}📋 {title}{self.colors['RESET']}")
        
        for key, value in data.items():
            print(f"{self._get_indent()}  {self.colors['INFO']}{key}{self.colors['RESET']}: {value}")
    
    def elapsed_time(self):
        """Print elapsed time since logger creation"""
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        self.time_info(f"Elapsed time: {minutes:02d}:{seconds:02d}")
    
    def separator(self):
        """Print a separator line"""
        print(f"{self._get_indent()}{'-' * 50}")

# Global logger instance
logger = BeautifulLogger("DBNet")

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

def print_device_info():
    """Print beautiful device information"""
    logger.header("Device Information")
    
    # CUDA Information
    if torch.cuda.is_available():
        logger.gpu_info(f"CUDA Available: {torch.cuda.is_available()}")
        logger.gpu_info(f"CUDA Version: {torch.version.cuda}")
        logger.gpu_info(f"GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.gpu_info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Current GPU
        current_gpu = torch.cuda.current_device()
        logger.gpu_info(f"Current GPU: {current_gpu}")
        
        # Memory usage
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        logger.memory_info(f"GPU Memory Allocated: {allocated:.2f} GB")
        logger.memory_info(f"GPU Memory Cached: {cached:.2f} GB")
    else:
        logger.warning("CUDA not available, using CPU")
    
    # PyTorch Information
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"Default Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

def print_model_summary(model, input_shape=(1, 3, 640, 640)):
    """Print beautiful model summary"""
    logger.header("Model Summary")
    
    # Model name
    logger.model_info(f"Model Name: {getattr(model, 'name', 'Unknown')}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.model_info(f"Total Parameters: {total_params:,}")
    logger.model_info(f"Trainable Parameters: {trainable_params:,}")
    logger.model_info(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    
    # Model size estimation
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    logger.model_info(f"Estimated Model Size: {model_size_mb:.2f} MB")
    
    # Test forward pass
    try:
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_shape).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        logger.model_info(f"Input Shape: {dummy_input.shape}")
        logger.model_info(f"Output Shape: {output.shape}")
        logger.success("Forward pass test successful")
        
    except Exception as e:
        logger.error(f"Forward pass test failed: {e}")

def print_config_summary(config: Dict[str, Any]):
    """Print beautiful configuration summary"""
    logger.header("Configuration Summary")
    
    # Model configuration
    if 'model' in config:
        logger.section("Model Configuration")
        logger.table(config['model'], "Model Settings")
        logger.end_section()
    
    # Training configuration
    if 'training' in config:
        logger.section("Training Configuration")
        logger.table(config['training'], "Training Settings")
        logger.end_section()
    
    # Data configuration
    if 'data' in config:
        logger.section("Data Configuration")
        logger.table(config['data'], "Data Settings")
        logger.end_section()
    
    # Loss configuration
    if 'loss' in config:
        logger.section("Loss Configuration")
        logger.table(config['loss'], "Loss Settings")
        logger.end_section()

def print_training_progress(epoch: int, total_epochs: int, train_loss: float, 
                          val_loss: Optional[float] = None, lr: Optional[float] = None):
    """Print beautiful training progress"""
    progress = (epoch / total_epochs) * 100
    progress_bar = "█" * int(progress / 5) + "░" * (20 - int(progress / 5))
    
    print(f"\n{logger._get_indent()}{logger.colors['HEADER']}🏋️  Epoch {epoch}/{total_epochs} ({progress:.1f}%){logger.colors['RESET']}")
    print(f"{logger._get_indent()}{logger.colors['INFO']}Progress: [{progress_bar}]{logger.colors['RESET']}")
    
    metrics = {"Training Loss": f"{train_loss:.4f}"}
    if val_loss is not None:
        metrics["Validation Loss"] = f"{val_loss:.4f}"
    if lr is not None:
        metrics["Learning Rate"] = f"{lr:.6f}"
    
    logger.metrics(metrics, "Epoch Metrics")

def print_dataset_info(dataset, name: str = "Dataset"):
    """Print beautiful dataset information"""
    logger.header(f"{name} Information")
    
    logger.data_info(f"Dataset Size: {len(dataset):,} samples")
    logger.data_info(f"Dataset Type: {type(dataset).__name__}")
    
    # Try to get sample info
    try:
        sample = dataset[0]
        if isinstance(sample, dict):
            logger.data_info("Sample Keys:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    logger.substep(f"  {key}: {value.shape} ({value.dtype})")
                else:
                    logger.substep(f"  {key}: {type(value).__name__}")
        else:
            logger.data_info(f"Sample Type: {type(sample).__name__}")
    except Exception as e:
        logger.warning(f"Could not get sample info: {e}")

def print_checkpoint_info(checkpoint_path: str, is_best: bool = False):
    """Print beautiful checkpoint information"""
    if is_best:
        logger.checkpoint_info(f"💾 Saving BEST model to: {checkpoint_path}")
    else:
        logger.checkpoint_info(f"💾 Saving checkpoint to: {checkpoint_path}")
    
    # File size
    if os.path.exists(checkpoint_path):
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        logger.checkpoint_info(f"File Size: {size_mb:.2f} MB")

def print_inference_results(results: Dict[str, Any], image_path: str):
    """Print beautiful inference results"""
    logger.header("Inference Results")
    
    logger.inference_info(f"Image: {image_path}")
    
    if 'text_regions' in results:
        logger.inference_info(f"Text Regions Detected: {len(results['text_regions'])}")
        
        for i, region in enumerate(results['text_regions']):
            logger.substep(f"Region {i+1}: {region}")
    
    if 'confidence' in results:
        logger.inference_info(f"Average Confidence: {results['confidence']:.3f}")
    
    if 'processing_time' in results:
        logger.time_info(f"Processing Time: {results['processing_time']:.3f}s")

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