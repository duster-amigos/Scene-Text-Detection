import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from shapely.geometry import Polygon
import pyclipper
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ICDAR2015Dataset(Dataset):
    def __init__(self, data_dir, is_training=True, size=(640, 640), transform=None):
        """
        ICDAR2015 dataset loader
        Args:
            data_dir: str, path to icdar2015 dataset
            is_training: bool, if True, load training data
            size: tuple, target size (h, w)
            transform: albumentations transform pipeline
        """
        self.data_dir = data_dir
        self.is_training = is_training
        self.size = size
        
        # Set up image and label directories
        split = 'train' if is_training else 'test'
        self.img_dir = os.path.join(data_dir, split, 'images')
        self.label_dir = os.path.join(data_dir, split, 'labels')
        
        # Get all image files
        self.img_files = [f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))]
        self.img_files.sort()
        
        # Default augmentation pipeline
        if transform is None:
            self.transform = self.get_transform(is_training)
        else:
            self.transform = transform

    def get_transform(self, is_training):
        """Get default augmentation pipeline"""
        if is_training:
            return A.Compose([
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.Normalize(),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Normalize(),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(self.label_dir, label_name)
        polygons = []
        
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        # ICDAR2015 format: x1,y1,x2,y2,x3,y3,x4,y4,text
                        parts = line.strip().split(',')
                        points = [float(x) for x in parts[:8]]
                        polygon = np.array(points).reshape(4, 2)
                        polygons.append(polygon)
                    except Exception as e:
                        print(f"Error parsing line in {label_path}: {e}")
                        continue
        except Exception as e:
            print(f"Error reading {label_path}: {e}")
            polygons = []

        # Generate target maps
        target = self.generate_target(img.shape[:2], polygons)
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=img, 
                                     masks=[target['shrink_map'],
                                           target['threshold_map'],
                                           target['shrink_mask'],
                                           target['threshold_mask']])
            img = augmented['image']
            
            # Handle tensor conversion properly
            if isinstance(augmented['masks'][0], torch.Tensor):
                target['shrink_map'] = augmented['masks'][0].unsqueeze(0)
            else:
                target['shrink_map'] = torch.from_numpy(augmented['masks'][0]).unsqueeze(0)
                
            if isinstance(augmented['masks'][1], torch.Tensor):
                target['threshold_map'] = augmented['masks'][1].unsqueeze(0)
            else:
                target['threshold_map'] = torch.from_numpy(augmented['masks'][1]).unsqueeze(0)
                
            if isinstance(augmented['masks'][2], torch.Tensor):
                target['shrink_mask'] = augmented['masks'][2]
            else:
                target['shrink_mask'] = torch.from_numpy(augmented['masks'][2])
                
            if isinstance(augmented['masks'][3], torch.Tensor):
                target['threshold_mask'] = augmented['masks'][3]
            else:
                target['threshold_mask'] = torch.from_numpy(augmented['masks'][3])

        return img, target

    def generate_target(self, img_size, polygons, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7):
        """Generate shrink map and threshold map from polygons"""
        h, w = img_size
        shrink_map = np.zeros((h, w), dtype=np.float32)
        threshold_map = np.zeros((h, w), dtype=np.float32)
        shrink_mask = np.ones((h, w), dtype=np.float32)
        threshold_mask = np.ones((h, w), dtype=np.float32)

        for polygon in polygons:
            try:
                # Convert to Shapely polygon
                polygon_shape = Polygon(polygon)
                if not polygon_shape.is_valid:
                    continue
                
                # Generate shrink map
                distance = polygon_shape.area * (1 - shrink_ratio * shrink_ratio) / polygon_shape.length
                subject = [tuple(p) for p in polygon]
                pco = pyclipper.PyclipperOffset()
                pco.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrink = np.array(pco.Execute(-distance)[0])
                cv2.fillPoly(shrink_map, [shrink.astype(np.int32)], 1)
                
                # Generate threshold map
                polygon_shape = Polygon(shrink)
                if not polygon_shape.is_valid:
                    continue
                    
                distance = polygon_shape.area * (1 - thresh_min * thresh_min) / polygon_shape.length
                pco = pyclipper.PyclipperOffset()
                pco.AddPath([tuple(p) for p in shrink], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                thresh_inner = np.array(pco.Execute(-distance)[0])
                
                distance = polygon_shape.area * (1 - thresh_max * thresh_max) / polygon_shape.length
                pco = pyclipper.PyclipperOffset()
                pco.AddPath([tuple(p) for p in shrink], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                thresh_outer = np.array(pco.Execute(distance)[0])
                
                thresh_inner = thresh_inner.reshape(-1, 2)
                thresh_outer = thresh_outer.reshape(-1, 2)
                
                for i in range(thresh_inner.shape[0]):
                    cv2.line(threshold_map, 
                            tuple(thresh_inner[i].astype(np.int32)),
                            tuple(thresh_inner[(i + 1) % thresh_inner.shape[0]].astype(np.int32)),
                            thresh_max, 1)
                    
                for i in range(thresh_outer.shape[0]):
                    cv2.line(threshold_map, 
                            tuple(thresh_outer[i].astype(np.int32)),
                            tuple(thresh_outer[(i + 1) % thresh_outer.shape[0]].astype(np.int32)),
                            thresh_min, 1)
                
                thresh_mask = np.zeros_like(threshold_map)
                cv2.fillPoly(thresh_mask, [thresh_outer.astype(np.int32)], 1)
                cv2.fillPoly(thresh_mask, [thresh_inner.astype(np.int32)], 0)
                
                # Ensure thresh_mask is uint8 for distanceTransform
                thresh_mask = thresh_mask.astype(np.uint8)
                threshold_map = cv2.distanceTransform(thresh_mask, cv2.DIST_L2, 0)
                threshold_map = threshold_map / (np.max(threshold_map) + 1e-6)
                threshold_map = np.clip(threshold_map, thresh_min, thresh_max)
                
            except Exception as e:
                print(f"Error generating target for polygon: {e}")
                continue

        return {
            'shrink_map': shrink_map.astype(np.float32),
            'threshold_map': threshold_map.astype(np.float32),
            'shrink_mask': shrink_mask,
            'threshold_mask': threshold_mask,
            'boxes': np.array(polygons, dtype=np.float32) if polygons else np.zeros((0, 4, 2), dtype=np.float32)
        } 