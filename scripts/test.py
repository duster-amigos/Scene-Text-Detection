import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
import time
from tqdm import tqdm
import json
from src.models.model import Model
from src.data.icdar2015_dataset import ICDAR2015Dataset, get_transforms
from src.utils.logger import logger
from src.inference.text_detector import TextDetector

class TextDetectionEvaluator:
    def __init__(self, model_path, config_path, device=None):
        """
        Initialize text detection evaluator
        
        Args:
            model_path (str): Path to trained model checkpoint
            config_path (str): Path to model configuration file
            device (str): Device to run evaluation on
        """
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            # Set device
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
            print(f"Using device: {self.device}")
            
            # Initialize detector
            self.detector = TextDetector(model_path, config_path, device)
            
        except Exception as e:
            print(f"Error initializing evaluator: {e}")
            raise
    
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union between two boxes
        
        Args:
            box1, box2: Boxes in format [x1, y1, x2, y2]
            
        Returns:
            float: IoU value
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
    
    def load_ground_truth(self, gt_path):
        """
        Load ground truth annotations
        
        Args:
            gt_path (str): Path to ground truth file
            
        Returns:
            list: List of ground truth boxes
        """
        try:
            gt_boxes = []
            if os.path.exists(gt_path):
                with open(gt_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split(',')
                            if len(parts) >= 8:
                                coords = [float(x) for x in parts[:8]]
                                # Convert polygon to bounding box
                                x_coords = coords[::2]
                                y_coords = coords[1::2]
                                x1, y1 = min(x_coords), min(y_coords)
                                x2, y2 = max(x_coords), max(y_coords)
                                gt_boxes.append([x1, y1, x2, y2])
            return gt_boxes
            
        except Exception as e:
            print(f"Error loading ground truth: {e}")
            return []
    
    def evaluate_single_image(self, image_path, gt_path, iou_threshold=0.5):
        """
        Evaluate detection on a single image
        
        Args:
            image_path (str): Path to image
            gt_path (str): Path to ground truth file
            iou_threshold (float): IoU threshold for matching
            
        Returns:
            dict: Evaluation results
        """
        try:
            # Load ground truth
            gt_boxes = self.load_ground_truth(gt_path)
            
            # Detect text
            pred_boxes = self.detector.detect_text(image_path)
            pred_boxes = [box['bbox'] for box in pred_boxes]
            
            # Calculate metrics
            tp = 0  # True positives
            fp = 0  # False positives
            fn = 0  # False negatives
            
            # Match predictions with ground truth
            matched_gt = set()
            matched_pred = set()
            
            for i, pred_box in enumerate(pred_boxes):
                best_iou = 0
                best_gt_idx = -1
                
                for j, gt_box in enumerate(gt_boxes):
                    if j in matched_gt:
                        continue
                    
                    iou = self.calculate_iou(pred_box, gt_box)
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
            fn = len(gt_boxes) - len(matched_gt)
            
            return {
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'gt_count': len(gt_boxes),
                'pred_count': len(pred_boxes)
            }
            
        except Exception as e:
            print(f"Error evaluating image {image_path}: {e}")
            return {
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'gt_count': 0,
                'pred_count': 0
            }
    
    def evaluate_dataset(self, test_images_dir, test_labels_dir, iou_threshold=0.5):
        """
        Evaluate detection on entire test dataset
        
        Args:
            test_images_dir (str): Directory containing test images
            test_labels_dir (str): Directory containing test labels
            iou_threshold (float): IoU threshold for matching
            
        Returns:
            dict: Overall evaluation metrics
        """
        try:
            # Get list of test images
            image_files = [f for f in os.listdir(test_images_dir) 
                          if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            print(f"Evaluating {len(image_files)} images...")
            
            total_tp = 0
            total_fp = 0
            total_fn = 0
            total_gt = 0
            total_pred = 0
            
            # Process each image
            pbar = tqdm(image_files, desc="Evaluating")
            for img_file in pbar:
                try:
                    image_path = os.path.join(test_images_dir, img_file)
                    gt_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
                    gt_path = os.path.join(test_labels_dir, gt_file)
                    
                    # Evaluate single image
                    results = self.evaluate_single_image(image_path, gt_path, iou_threshold)
                    
                    # Accumulate results
                    total_tp += results['tp']
                    total_fp += results['fp']
                    total_fn += results['fn']
                    total_gt += results['gt_count']
                    total_pred += results['pred_count']
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'TP': total_tp,
                        'FP': total_fp,
                        'FN': total_fn
                    })
                    
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
                    continue
            
            # Calculate final metrics
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'total_tp': total_tp,
                'total_fp': total_fp,
                'total_fn': total_fn,
                'total_gt': total_gt,
                'total_pred': total_pred,
                'iou_threshold': iou_threshold
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating dataset: {e}")
            return {}

def main():
    parser = argparse.ArgumentParser(description='Evaluate text detection model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--test_images', type=str, required=True, help='Path to test images directory')
    parser.add_argument('--test_labels', type=str, required=True, help='Path to test labels directory')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for evaluation')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = TextDetectionEvaluator(args.model, args.config, args.device)
        
        # Evaluate dataset
        print("Starting evaluation...")
        start_time = time.time()
        metrics = evaluator.evaluate_dataset(
            args.test_images, 
            args.test_labels, 
            args.iou_threshold
        )
        evaluation_time = time.time() - start_time
        
        # Print results
        print(f"\nEvaluation completed in {evaluation_time:.2f} seconds")
        print(f"IoU Threshold: {metrics['iou_threshold']}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Total Ground Truth: {metrics['total_gt']}")
        print(f"Total Predictions: {metrics['total_pred']}")
        print(f"True Positives: {metrics['total_tp']}")
        print(f"False Positives: {metrics['total_fp']}")
        print(f"False Negatives: {metrics['total_fn']}")
        
        # Save results
        results_file = 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to {results_file}")
        
    except Exception as e:
        print(f"Error in main: {e}")
        raise

if __name__ == '__main__':
    main() 