#!/usr/bin/env python3
"""
Test script to verify training works without validation data
"""

import json
import os
import tempfile
import shutil

def create_test_config():
    """Create a test config with no validation data"""
    config = {
        "model": {
            "backbone": {
                "type": "MobileNetV3",
                "pretrained": True,
                "in_channels": 3
            },
            "neck": {
                "type": "FPEM_FFM",
                "inner_channels": 128,
                "fpem_repeat": 2
            },
            "head": {
                "type": "DBHead",
                "out_channels": 2,
                "k": 50
            }
        },
        "loss": {
            "alpha": 1.0,
            "beta": 10.0,
            "ohem_ratio": 3.0,
            "reduction": "mean",
            "eps": 1e-6
        },
        "training": {
            "epochs": 2,  # Just 2 epochs for testing
            "batch_size": 2,  # Small batch size
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "lr_step_size": 30,
            "lr_gamma": 0.1,
            "image_size": 640,
            "num_workers": 0,  # No multiprocessing for testing
            "resume": False,
            "checkpoint_path": "test_checkpoint.pth"
        },
        "data": {
            "train_images": "test_data/images",
            "train_labels": "test_data/annotations",
            "val_images": "",  # Empty - no validation
            "val_labels": "",  # Empty - no validation
            "test_images": "",
            "test_labels": ""
        },
        "inference": {
            "min_area": 100,
            "thresh": 0.3,
            "box_thresh": 0.5,
            "max_candidates": 1000
        },
        "evaluation": {
            "iou_threshold": 0.5,
            "save_predictions": True,
            "output_dir": "test_results"
        }
    }
    return config

def create_test_data():
    """Create minimal test data"""
    try:
        print("Creating test data...")
        
        # Create directories
        os.makedirs("test_data/images", exist_ok=True)
        os.makedirs("test_data/annotations", exist_ok=True)
        
        # Create a simple test image (just a small random image)
        import numpy as np
        import cv2
        
        # Create 2 test images
        for i in range(2):
            # Create a simple image
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img_path = f"test_data/images/test_img_{i}.jpg"
            cv2.imwrite(img_path, img)
            
            # Create annotation file
            ann_path = f"test_data/annotations/test_img_{i}.txt"
            with open(ann_path, 'w') as f:
                # Add one text region per image
                coords = [10, 10, 50, 10, 50, 30, 10, 30]
                line = ','.join([str(x) for x in coords] + ['test'])
                f.write(line + '\n')
            
            print(f"Created test_img_{i}.jpg with annotation")
        
        print("Test data created successfully")
        
    except Exception as e:
        print(f"Error creating test data: {e}")
        raise

def test_training():
    """Test training without validation"""
    try:
        print("Testing training without validation data...")
        
        # Create test config
        config = create_test_config()
        
        # Save config
        with open('test_config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        print("Test config saved to test_config.json")
        
        # Import and test trainer
        from train import Trainer
        
        print("Creating trainer...")
        trainer = Trainer(config)
        
        print("Starting training (2 epochs)...")
        trainer.train()
        
        print("Training completed successfully!")
        
        # Clean up
        if os.path.exists('test_checkpoint.pth'):
            os.remove('test_checkpoint.pth')
        if os.path.exists('test_checkpoint_best.pth'):
            os.remove('test_checkpoint_best.pth')
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
        raise

def cleanup():
    """Clean up test files"""
    try:
        print("Cleaning up test files...")
        
        # Remove test data
        if os.path.exists('test_data'):
            shutil.rmtree('test_data')
        
        # Remove test config
        if os.path.exists('test_config.json'):
            os.remove('test_config.json')
        
        # Remove checkpoints
        if os.path.exists('test_checkpoint.pth'):
            os.remove('test_checkpoint.pth')
        if os.path.exists('test_checkpoint_best.pth'):
            os.remove('test_checkpoint_best.pth')
        
        print("Cleanup completed")
        
    except Exception as e:
        print(f"Error in cleanup: {e}")

if __name__ == '__main__':
    try:
        create_test_data()
        test_training()
        cleanup()
        print("All tests passed! Training without validation works correctly.")
    except Exception as e:
        print(f"Test failed: {e}")
        cleanup()
        raise 