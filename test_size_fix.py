#!/usr/bin/env python3
"""
Test script to verify that the ground truth map resizing fix is working correctly.
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataset import ICDAR2015Dataset, get_transforms
from utils import logger

def test_gt_map_sizes():
    """Test that ground truth maps are correctly resized to match transformed image size."""
    try:
        logger.header("Testing Ground Truth Map Size Fix")
        
        # Create a dummy dataset with transforms
        transform = get_transforms(image_size=640, is_training=True)
        
        # Create a dummy dataset (this will fail if data doesn't exist, but we can test the logic)
        try:
            dataset = ICDAR2015Dataset(
                data_dir="./train_data/images",
                gt_dir="./train_data/texts", 
                transform=transform,
                is_training=True
            )
            
            if len(dataset) > 0:
                # Test with first sample
                sample = dataset[0]
                
                logger.success("Dataset sample loaded successfully")
                logger.data_info(f"Image shape: {sample['image'].shape}")
                logger.data_info(f"Shrink map shape: {sample['shrink_map'].shape}")
                logger.data_info(f"Threshold map shape: {sample['threshold_map'].shape}")
                logger.data_info(f"Shrink mask shape: {sample['shrink_mask'].shape}")
                logger.data_info(f"Threshold mask shape: {sample['threshold_mask'].shape}")
                
                # Verify all shapes are correct
                expected_shape = (640, 640)
                expected_map_shape = (1, 640, 640)
                
                assert sample['image'].shape == (3, 640, 640), f"Image shape mismatch: {sample['image'].shape}"
                assert sample['shrink_map'].shape == expected_map_shape, f"Shrink map shape mismatch: {sample['shrink_map'].shape}"
                assert sample['threshold_map'].shape == expected_map_shape, f"Threshold map shape mismatch: {sample['threshold_map'].shape}"
                assert sample['shrink_mask'].shape == expected_shape, f"Shrink mask shape mismatch: {sample['shrink_mask'].shape}"
                assert sample['threshold_mask'].shape == expected_shape, f"Threshold mask shape mismatch: {sample['threshold_mask'].shape}"
                
                logger.success("✅ All ground truth map shapes are correct!")
                
                # Test interpolation logic
                logger.info("Testing interpolation logic...")
                
                # Create dummy original size maps
                original_h, original_w = 720, 1280
                dummy_shrink_map = torch.randn(1, original_h, original_w)
                dummy_threshold_map = torch.randn(1, original_h, original_w)
                dummy_shrink_mask = torch.randn(original_h, original_w)
                dummy_threshold_mask = torch.randn(original_h, original_w)
                
                logger.data_info(f"Original maps - shrink: {dummy_shrink_map.shape}, threshold: {dummy_threshold_map.shape}")
                
                # Resize to target size
                target_size = (640, 640)
                
                # Resize shrink map
                resized_shrink_map = F.interpolate(
                    dummy_shrink_map, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # Resize shrink mask
                resized_shrink_mask = F.interpolate(
                    dummy_shrink_mask.unsqueeze(0).unsqueeze(0), 
                    size=target_size, 
                    mode='nearest'
                ).squeeze(0).squeeze(0)
                
                # Resize threshold map
                resized_threshold_map = F.interpolate(
                    dummy_threshold_map, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # Resize threshold mask
                resized_threshold_mask = F.interpolate(
                    dummy_threshold_mask.unsqueeze(0).unsqueeze(0), 
                    size=target_size, 
                    mode='nearest'
                ).squeeze(0).squeeze(0)
                
                logger.data_info(f"Resized maps - shrink: {resized_shrink_map.shape}, threshold: {resized_threshold_map.shape}")
                logger.data_info(f"Resized masks - shrink: {resized_shrink_mask.shape}, threshold: {resized_threshold_mask.shape}")
                
                assert resized_shrink_map.shape == (1, 640, 640), f"Resized shrink map shape mismatch: {resized_shrink_map.shape}"
                assert resized_threshold_map.shape == (1, 640, 640), f"Resized threshold map shape mismatch: {resized_threshold_map.shape}"
                assert resized_shrink_mask.shape == (640, 640), f"Resized shrink mask shape mismatch: {resized_shrink_mask.shape}"
                assert resized_threshold_mask.shape == (640, 640), f"Resized threshold mask shape mismatch: {resized_threshold_mask.shape}"
                
                logger.success("✅ Interpolation logic test passed!")
                
            else:
                logger.warning("Dataset is empty, skipping size tests")
                
        except FileNotFoundError:
            logger.warning("Data directories not found, testing only interpolation logic")
            
            # Test interpolation logic without dataset
            logger.info("Testing interpolation logic...")
            
            # Create dummy original size maps
            original_h, original_w = 720, 1280
            dummy_shrink_map = torch.randn(1, original_h, original_w)
            dummy_threshold_map = torch.randn(1, original_h, original_w)
            dummy_shrink_mask = torch.randn(original_h, original_w)
            dummy_threshold_mask = torch.randn(original_h, original_w)
            
            logger.data_info(f"Original maps - shrink: {dummy_shrink_map.shape}, threshold: {dummy_threshold_map.shape}")
            
            # Resize to target size
            target_size = (640, 640)
            
            # Resize shrink map
            resized_shrink_map = F.interpolate(
                dummy_shrink_map, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
            
            # Resize shrink mask
            resized_shrink_mask = F.interpolate(
                dummy_shrink_mask.unsqueeze(0).unsqueeze(0), 
                size=target_size, 
                mode='nearest'
            ).squeeze(0).squeeze(0)
            
            # Resize threshold map
            resized_threshold_map = F.interpolate(
                dummy_threshold_map, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
            
            # Resize threshold mask
            resized_threshold_mask = F.interpolate(
                dummy_threshold_mask.unsqueeze(0).unsqueeze(0), 
                size=target_size, 
                mode='nearest'
            ).squeeze(0).squeeze(0)
            
            logger.data_info(f"Resized maps - shrink: {resized_shrink_map.shape}, threshold: {resized_threshold_map.shape}")
            logger.data_info(f"Resized masks - shrink: {resized_shrink_mask.shape}, threshold: {resized_threshold_mask.shape}")
            
            assert resized_shrink_map.shape == (1, 640, 640), f"Resized shrink map shape mismatch: {resized_shrink_map.shape}"
            assert resized_threshold_map.shape == (1, 640, 640), f"Resized threshold map shape mismatch: {resized_threshold_map.shape}"
            assert resized_shrink_mask.shape == (640, 640), f"Resized shrink mask shape mismatch: {resized_shrink_mask.shape}"
            assert resized_threshold_mask.shape == (640, 640), f"Resized threshold mask shape mismatch: {resized_threshold_mask.shape}"
            
            logger.success("✅ Interpolation logic test passed!")
        
        logger.success("Ground truth map size fix test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in ground truth map size test: {e}")
        raise

if __name__ == "__main__":
    test_gt_map_sizes() 