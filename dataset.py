import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import random
from PIL import Image
import torchvision.transforms as transforms
from utils import logger

class ICDAR2015Dataset(Dataset):
    """
    Dataset class for ICDAR 2015 text detection dataset
    """
    def __init__(self, data_dir, gt_dir, transform=None, is_training=True, 
                 min_text_size=8, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7):
        """
        Args:
            data_dir (str): Directory containing images
            gt_dir (str): Directory containing ground truth annotations
            transform: Optional transform to be applied on images
            is_training (bool): Whether this is training or validation set
            min_text_size (int): Minimum text size to consider
            shrink_ratio (float): Ratio for shrinking text regions
            thresh_min (float): Minimum threshold for threshold map
            thresh_max (float): Maximum threshold for threshold map
        """
        try:
            dataset_type = "Training" if is_training else "Validation"
            logger.data_info(f"Initializing ICDAR2015 {dataset_type} Dataset")
            
            self.data_dir = data_dir
            self.gt_dir = gt_dir
            self.transform = transform
            self.is_training = is_training
            self.min_text_size = min_text_size
            self.shrink_ratio = shrink_ratio
            self.thresh_min = thresh_min
            self.thresh_max = thresh_max
            
            # Check if directories exist
            if not os.path.exists(data_dir):
                logger.warning(f"Data directory does not exist: {data_dir}")
            if not os.path.exists(gt_dir):
                logger.warning(f"Ground truth directory does not exist: {gt_dir}")
            
            self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            logger.data_info(f"Found {len(self.image_files)} images")
            
            if len(self.image_files) == 0:
                logger.warning("No image files found in data directory")
                
        except Exception as e:
            logger.error(f"Error initializing dataset: {e}")
            self.image_files = []
            raise
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        try:
            # Load image
            img_name = self.image_files[idx]
            img_path = os.path.join(self.data_dir, img_name)
            
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Could not load image: {img_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Load ground truth
            gt_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
            gt_path = os.path.join(self.gt_dir, gt_name)
            
            polygons = []
            if os.path.exists(gt_path):
                try:
                    with open(gt_path, 'r', encoding='utf-8-sig') as f:
                        all_lines = [l for l in f if l.strip()]
                        for line_num, line in enumerate(all_lines):
                            line = line.strip()
                            if line:
                                # ICDAR 2015 format: x1,y1,x2,y2,x3,y3,x4,y4,text
                                parts = line.split(',')
                                if len(parts) >= 8:
                                    coords = [float(x) for x in parts[:8]]
                                    polygon = np.array(coords).reshape(-1, 2)
                                    polygons.append(polygon)
                                else:
                                    logger.warning(f"Invalid line format at line {line_num + 1}: {line}")
                except Exception as e:
                    logger.error(f"Error reading annotation file {gt_path}: {e}")
            else:
                logger.warning(f"Ground truth file not found: {gt_path}")
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Generate ground truth maps at original size first
            shrink_map, shrink_mask, threshold_map, threshold_mask = self.generate_gt_maps(
                h, w, polygons
            )
            
            # Resize ground truth maps to match transformed image size
            if self.transform:
                target_size = (640, 640)  # Match the transform resize
                
                # Resize shrink map
                shrink_map = F.interpolate(
                    shrink_map.unsqueeze(0), 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                
                # Resize shrink mask
                shrink_mask = F.interpolate(
                    shrink_mask.unsqueeze(0).unsqueeze(0), 
                    size=target_size, 
                    mode='nearest'
                ).squeeze(0).squeeze(0)
                
                # Resize threshold map
                threshold_map = F.interpolate(
                    threshold_map.unsqueeze(0), 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                
                # Resize threshold mask
                threshold_mask = F.interpolate(
                    threshold_mask.unsqueeze(0).unsqueeze(0), 
                    size=target_size, 
                    mode='nearest'
                ).squeeze(0).squeeze(0)
            
            result = {
                'image': image,
                'shrink_map': shrink_map,
                'shrink_mask': shrink_mask,
                'threshold_map': threshold_map,
                'threshold_mask': threshold_mask,
                'filename': img_name
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return a dummy sample
            logger.warning("Returning dummy sample due to error")
            dummy_image = torch.zeros(3, 640, 640)
            dummy_map = torch.zeros(1, 640, 640)
            dummy_mask = torch.zeros(640, 640)
            return {
                'image': dummy_image,
                'shrink_map': dummy_map,
                'shrink_mask': dummy_mask,
                'threshold_map': dummy_map,
                'threshold_mask': dummy_mask,
                'filename': 'dummy.jpg'
            }
    
    def generate_gt_maps(self, h, w, polygons):
        """
        Generate ground truth maps for training
        """
        try:
            shrink_map = np.zeros((h, w), dtype=np.float32)
            shrink_mask = np.zeros((h, w), dtype=np.float32)
            threshold_map = np.zeros((h, w), dtype=np.float32)
            threshold_mask = np.zeros((h, w), dtype=np.float32)
            
            for i, polygon in enumerate(polygons):
                # Create shrink polygon
                shrink_polygon = self.shrink_polygon(polygon, self.shrink_ratio)
                
                # Draw shrink map
                cv2.fillPoly(shrink_map, [shrink_polygon.astype(np.int32)], 1.0)
                cv2.fillPoly(shrink_mask, [shrink_polygon.astype(np.int32)], 1.0)
                
                # Create threshold map
                thresh_polygon = self.shrink_polygon(polygon, 0.7)
                cv2.fillPoly(threshold_map, [thresh_polygon.astype(np.int32)], 0.5)
                cv2.fillPoly(threshold_mask, [thresh_polygon.astype(np.int32)], 1.0)
            
            # Convert to tensors
            shrink_map = torch.from_numpy(shrink_map).unsqueeze(0)
            shrink_mask = torch.from_numpy(shrink_mask)
            threshold_map = torch.from_numpy(threshold_map).unsqueeze(0)
            threshold_mask = torch.from_numpy(threshold_mask)
            
            return shrink_map, shrink_mask, threshold_map, threshold_mask
            
        except Exception as e:
            logger.error(f"Error generating GT maps: {e}")
            # Return empty maps
            shrink_map = torch.zeros(1, h, w)
            shrink_mask = torch.zeros(h, w)
            threshold_map = torch.zeros(1, h, w)
            threshold_mask = torch.zeros(h, w)
            return shrink_map, shrink_mask, threshold_map, threshold_mask
    
    def shrink_polygon(self, polygon, ratio):
        """
        Shrink polygon by given ratio
        """
        try:
            center = np.mean(polygon, axis=0)
            shrunk_polygon = (polygon - center) * ratio + center
            return shrunk_polygon
        except Exception as e:
            logger.error(f"Error shrinking polygon: {e}")
            return polygon

def get_transforms(image_size=640, is_training=True):
    """
    Get transforms for data augmentation
    """
    try:
        if is_training:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        return transform
    except Exception as e:
        logger.error(f"Error creating transforms: {e}")
        raise 