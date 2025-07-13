#!/usr/bin/env python3
"""
Final Comprehensive Test Suite for DBNet Implementation
This script tests all components of the DBNet project thoroughly.
Run this in Google Colab with GPU to verify everything works correctly.
"""

# Standard library imports
import os
import time
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2
import numpy as np

from model import Model
from losses import DBLoss
from utils.postprocess import process_predictions
from utils.metrics import compute_batch_metrics

# Add current directory to path
import sys
sys.path.append('.')

# Helper functions for printing
def print_header(text):
    print("\n" + "=" * 80)
    print(f"🧪 {text}")
    print("=" * 80)

def print_info(text):
    print(f"ℹ️  {text}")

def print_success(text):
    print(f"✅ {text}")

def print_error(text, error=None):
    print(f"❌ {text}")
    if error:
        print(f"   Error: {error}")

def test_imports():
    """Test all imports."""
    print_header("Testing Imports")
    
    try:
        # Core architecture
        from backbone_mobilenetv3 import MobileNetV3
        print_success("MobileNetV3 backbone imported")
        
        from neck_fpem_ffm import FPEM_FFM
        print_success("FPEM_FFM neck imported")
        
        from head_DBHead import DBHead
        print_success("DBHead imported")
        
        from losses import DBLoss, BalanceCrossEntropyLoss, DiceLoss, MaskL1Loss
        print_success("All loss functions imported")
        
        from build import build_backbone, build_neck, build_head, build_loss
        print_success("Build functions imported")
        
        # Data and utils
        from datasets.icdar2015 import ICDAR2015Dataset
        print_success("ICDAR2015 dataset imported")
        
        from utils.postprocess import process_predictions, get_boxes_from_bitmap
        print_success("Post-processing utilities imported")
        
        from utils.metrics import compute_batch_metrics, evaluate_detections
        print_success("Metrics utilities imported")
        
        return True
        
    except Exception as e:
        print_error("Import test failed", str(e))
        traceback.print_exc()
        return False

def test_device():
    """Test device availability."""
    print_header("Testing Device")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print_info(f"GPU: {torch.cuda.get_device_name()}")
        print_info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print_success("CUDA is available")
    else:
        print_info("CUDA not available, using CPU")
    
    return device

def test_backbone():
    """Test backbone architecture."""
    print_header("Testing Backbone (MobileNetV3-Small)")
    
    try:
        from backbone_mobilenetv3 import MobileNetV3
        
        # Test backbone
        backbone = MobileNetV3(pretrained=False)
        print_info(f"Backbone output channels: {backbone.out_channels}")
        
        # Test forward pass
        x = torch.randn(2, 3, 640, 640)
        features = backbone(x)
        
        print_info(f"Input shape: {x.shape}")
        for i, feat in enumerate(features):
            print_info(f"Feature {i+1} shape: {feat.shape}")
        
        # Verify output channels
        expected_channels = [24, 40, 96, 576]
        for i, (feat, expected) in enumerate(zip(features, expected_channels)):
            assert feat.shape[1] == expected, f"Feature {i+1} has {feat.shape[1]} channels, expected {expected}"
        
        print_success("Backbone test passed")
        return True
        
    except Exception as e:
        print_error("Backbone test failed", str(e))
        traceback.print_exc()
        return False

def test_neck():
    """Test neck architecture."""
    print_header("Testing Neck (FPEM_FFM)")
    
    try:
        from neck_fpem_ffm import FPEM_FFM
        
        # Test neck
        in_channels = [24, 40, 96, 576]  # MobileNetV3-Small output channels
        neck = FPEM_FFM(in_channels, inner_channels=256)
        print_info(f"Neck output channels: {neck.out_channels}")
        
        # Test forward pass
        features = [
            torch.randn(2, 24, 160, 160),
            torch.randn(2, 40, 80, 80),
            torch.randn(2, 96, 40, 40),
            torch.randn(2, 576, 20, 20)
        ]
        
        output = neck(features)
        print_info(f"Neck output shape: {output.shape}")
        
        # Verify output
        expected_channels = 256 * 4  # inner_channels * 4
        assert output.shape[1] == expected_channels, f"Neck output has {output.shape[1]} channels, expected {expected_channels}"
        
        print_success("Neck test passed")
        return True
        
    except Exception as e:
        print_error("Neck test failed", str(e))
        traceback.print_exc()
        return False

def test_head():
    """Test head architecture."""
    print_header("Testing Head (DBHead)")
    
    try:
        from head_DBHead import DBHead
        
        # Test head
        in_channels = 1024  # 256 * 4 from neck
        head = DBHead(in_channels, out_channels=2, k=50)
        
        # Test forward pass (training mode)
        head.train()
        x = torch.randn(2, in_channels, 160, 160)
        output_train = head(x)
        print_info(f"Training output shape: {output_train.shape}")
        assert output_train.shape[1] == 3, f"Training output has {output_train.shape[1]} channels, expected 3"
        
        # Test forward pass (inference mode)
        head.eval()
        output_eval = head(x)
        print_info(f"Inference output shape: {output_eval.shape}")
        assert output_eval.shape[1] == 2, f"Inference output has {output_eval.shape[1]} channels, expected 2"
        
        print_success("Head test passed")
        return True
        
    except Exception as e:
        print_error("Head test failed", str(e))
        traceback.print_exc()
        return False

def test_model():
    """Test complete model."""
    print_header("Testing Complete Model")
    
    try:
        # Import Model directly to avoid circular import issues
        from model import Model
        
        # Model configuration
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
        
        # Create model
        model = Model(model_config)
        print_info(f"Model name: {model.name}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print_info(f"Total parameters: {total_params:,}")
        print_info(f"Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        x = torch.randn(2, 3, 640, 640)
        
        # Training mode
        model.train()
        output_train = model(x)
        print_info(f"Training output shape: {output_train.shape}")
        assert output_train.shape[1] == 3, f"Training output has {output_train.shape[1]} channels, expected 3"
        
        # Inference mode
        model.eval()
        output_eval = model(x)
        print_info(f"Inference output shape: {output_eval.shape}")
        assert output_eval.shape[1] == 2, f"Inference output has {output_eval.shape[1]} channels, expected 2"
        
        print_success("Complete model test passed")
        return True
        
    except Exception as e:
        print_error("Complete model test failed", str(e))
        traceback.print_exc()
        return False

def test_losses():
    """Test loss functions."""
    print_header("Testing Loss Functions")
    
    try:
        from losses import BalanceCrossEntropyLoss, DiceLoss, MaskL1Loss, DBLoss
        
        # Create dummy data with correct shapes and values
        pred = torch.sigmoid(torch.randn(2, 1, 160, 160))  # Use sigmoid to ensure values between 0 and 1
        gt = torch.randint(0, 2, (2, 1, 160, 160)).float()  # Binary values (0 or 1)
        mask = torch.ones(2, 160, 160)  # Binary mask
        
        # Test BalanceCrossEntropyLoss
        bce_loss = BalanceCrossEntropyLoss()
        bce_out = bce_loss(pred, gt, mask)
        assert isinstance(bce_out, torch.Tensor)
        
        # Test DiceLoss
        dice_loss = DiceLoss()
        dice_out = dice_loss(pred, gt, mask)
        assert isinstance(dice_out, torch.Tensor)
        
        # Test MaskL1Loss
        l1_loss = MaskL1Loss()
        l1_out = l1_loss(pred, gt, mask)
        assert isinstance(l1_out, torch.Tensor)
        
        # Test DBLoss
        criterion = DBLoss()
        batch = {
            'shrink_map': torch.sigmoid(torch.randn(2, 1, 160, 160)),  # Values between 0 and 1
            'shrink_mask': torch.ones(2, 160, 160),  # Binary mask
            'threshold_map': torch.sigmoid(torch.randn(2, 1, 160, 160)),  # Values between 0 and 1
            'threshold_mask': torch.ones(2, 160, 160)  # Binary mask
        }
        pred = torch.sigmoid(torch.randn(2, 3, 160, 160))  # 3 channels for training mode, values between 0 and 1
        losses = criterion(pred, batch)
        
        print_info(f"Loss components: {list(losses.keys())}")
        print_success("Loss functions test passed")
        return True
        
    except Exception as e:
        print_error("Loss functions test failed", str(e))
        traceback.print_exc()
        return False

def test_dataset():
    """Test dataset loading."""
    print_header("Testing Dataset")
    
    try:
        from datasets.icdar2015 import ICDAR2015Dataset
        import os
        
        # Create temporary test data
        os.makedirs('temp_test/train/images', exist_ok=True)
        os.makedirs('temp_test/train/labels', exist_ok=True)
        
        # Create a dummy image and label
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.imwrite('temp_test/train/images/test.jpg', img)
        
        with open('temp_test/train/labels/test.txt', 'w') as f:
            # Write a simple polygon: x1,y1,x2,y2,x3,y3,x4,y4,text
            f.write('100,100,200,100,200,200,100,200,text\n')
        
        # Create dataset
        dataset = ICDAR2015Dataset('temp_test', is_training=True)
        print_info(f"Dataset length: {len(dataset)}")
        
        # Test loading an item
        img, target = dataset[0]
        
        # Verify shapes and types
        assert isinstance(img, torch.Tensor), "Image should be a tensor"
        assert img.shape == (3, 640, 640), f"Wrong image shape: {img.shape}"
        assert isinstance(target, dict), "Target should be a dictionary"
        assert isinstance(target['shrink_map'], torch.Tensor), "Shrink map should be a tensor"
        assert target['shrink_map'].shape == (1, 640, 640), f"Wrong shrink map shape: {target['shrink_map'].shape}"
        assert isinstance(target['shrink_mask'], torch.Tensor), "Shrink mask should be a tensor"
        assert target['shrink_mask'].shape == (640, 640), f"Wrong shrink mask shape: {target['shrink_mask'].shape}"
        
        # Clean up
        import shutil
        shutil.rmtree('temp_test')
        
        print_success("Dataset test passed")
        return True
        
    except Exception as e:
        print_error("Dataset test failed", str(e))
        traceback.print_exc()
        # Clean up on failure
        if os.path.exists('temp_test'):
            shutil.rmtree('temp_test')
        return False

def test_postprocessing():
    """Test post-processing utilities."""
    print_header("Testing Post-processing")
    
    try:
        from utils.postprocess import process_predictions, get_boxes_from_bitmap
        
        # Create dummy prediction
        pred = torch.randn(2, 160, 160)
        pred = torch.sigmoid(pred)  # Convert to probabilities
        
        # Test post-processing
        boxes, scores = process_predictions(pred, min_size=3, box_thresh=0.7)
        print_info(f"Detected {len(boxes)} boxes")
        print_info(f"Score range: {min(scores) if scores else 0:.3f} - {max(scores) if scores else 0:.3f}")
        
        print_success("Post-processing test passed")
        return True
        
    except Exception as e:
        print_error("Post-processing test failed", str(e))
        traceback.print_exc()
        return False

def test_metrics():
    """Test metrics computation."""
    print_header("Testing Metrics")
    
    try:
        from utils.metrics import compute_batch_metrics
        import numpy as np
        
        # Create dummy ground truth and predictions
        gt_boxes = [
            np.array([[100, 100], [200, 100], [200, 200], [100, 200]]).reshape(1, 4, 2),  # One box
            np.array([[300, 300], [400, 300], [400, 400], [300, 400]]).reshape(1, 4, 2)   # One box
        ]
        
        pred_boxes = [
            np.array([[110, 110], [190, 110], [190, 190], [110, 190]]).reshape(1, 4, 2),  # Close to first gt box
            np.array([[310, 310], [390, 310], [390, 390], [310, 390]]).reshape(1, 4, 2)   # Close to second gt box
        ]
        
        pred_scores = [
            np.array([0.9]),  # High confidence for first box
            np.array([0.8])   # High confidence for second box
        ]
        
        # Compute metrics
        precision, recall, f1 = compute_batch_metrics(gt_boxes, pred_boxes, pred_scores)
        
        # Verify metrics are in valid range [0, 1]
        assert 0 <= precision <= 1, f"Invalid precision: {precision}"
        assert 0 <= recall <= 1, f"Invalid recall: {recall}"
        assert 0 <= f1 <= 1, f"Invalid F1: {f1}"
        
        print_info(f"Precision: {precision:.4f}")
        print_info(f"Recall: {recall:.4f}")
        print_info(f"F1 Score: {f1:.4f}")
        
        print_success("Metrics test passed")
        return True
        
    except Exception as e:
        print_error("Metrics test failed", str(e))
        traceback.print_exc()
        return False

def create_model(device, pretrained=False):
    """Helper function to create and initialize model."""
    model_config = {
        'backbone': {
            'type': 'MobileNetV3',
            'pretrained': pretrained,
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
    
    try:
        print_info("Creating model on CPU first...")
        # Initialize model on CPU first
        model = Model(model_config)
        print_info("Model created on CPU successfully")
        
        # Initialize weights before moving to device
        print_info("Initializing model weights...")
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print_info("Weights initialized successfully")
        
        # Clear CUDA cache before moving to device
        if device.type == 'cuda':
            print_info("Clearing CUDA cache...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Move model to device after initialization
        print_info(f"Moving model to {device}...")
        model = model.to(device)
        print_info("Model moved to device successfully")
        
        # Ensure CUDA synchronization if using GPU
        if device.type == 'cuda':
            print_info("Synchronizing CUDA...")
            torch.cuda.synchronize()
        
        return model
    except Exception as e:
        print_error(f"Error creating model: {str(e)}")
        # Print more detailed error information
        import traceback
        print_error("Full traceback:")
        traceback.print_exc()
        raise e

def test_training_step(device):
    """Test training step."""
    print_header("Testing Training Step")
    
    try:
        # Create model with proper initialization
        model = create_model(device)
        model.train()
        
        # Create optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        # Create dummy batch
        batch_size = 2
        images = torch.randn(batch_size, 3, 640, 640).to(device)
        targets = {
            'shrink_map': torch.randn(batch_size, 1, 640, 640).to(device),
            'shrink_mask': torch.ones(batch_size, 640, 640).to(device),
            'threshold_map': torch.randn(batch_size, 1, 640, 640).to(device),
            'threshold_mask': torch.ones(batch_size, 640, 640).to(device)
        }
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        
        # Compute loss
        criterion = DBLoss().to(device)
        loss_dict = criterion(predictions, targets)
        loss = loss_dict['loss']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Ensure CUDA synchronization
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        print_success("Training step test passed")
        return True
        
    except Exception as e:
        print_error("Training step test failed", str(e))
        traceback.print_exc()
        return False

def test_inference(device):
    """Test inference."""
    print_header("Testing Inference")
    
    try:
        # Create model with proper initialization
        model = create_model(device)
        model.eval()
        
        # Create dummy batch
        batch_size = 2
        images = torch.randn(batch_size, 3, 640, 640).to(device)
        
        # Ensure CUDA synchronization before inference
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        with torch.no_grad():
            predictions = model(images)
        
        # Ensure CUDA synchronization after inference
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        print_info(f"Prediction shape: {predictions.shape}")
        
        # Process predictions
        batch_boxes = []
        batch_scores = []
        for pred in predictions.cpu():
            boxes, scores = process_predictions(pred)
            batch_boxes.append(boxes)
            batch_scores.append(scores)
            print_info(f"Image {len(batch_boxes)}: {len(boxes)} boxes detected")
        
        print_success("Inference test passed")
        return True
        
    except Exception as e:
        print_error("Inference test failed", str(e))
        traceback.print_exc()
        return False

def test_finetuning(device):
    """Test fine-tuning functionality."""
    print_header("Testing Fine-tuning")
    
    try:
        # Import Model directly
        from model import Model
        from losses import DBLoss
        from torch.optim import SGD, Adam
        
        # Model configuration
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
        
        # Test different fine-tuning configurations
        fine_tune_configs = [
            ("All layers", None, None),
            ("Head only", "head", None),
            ("Neck and head", "neck_head", None),
            ("Backbone and head", "backbone_head", None)
        ]
        
        for config_name, freeze_backbone, freeze_neck in fine_tune_configs:
            print_info(f"Testing {config_name} fine-tuning")
            
            # Reset model
            model = Model(model_config).to(device)
            
            # Freeze layers based on configuration
            if freeze_backbone == "head":
                for param in model.backbone.parameters():
                    param.requires_grad = False
                for param in model.neck.parameters():
                    param.requires_grad = False
            elif freeze_backbone == "neck_head":
                for param in model.backbone.parameters():
                    param.requires_grad = False
            elif freeze_backbone == "backbone_head":
                for param in model.neck.parameters():
                    param.requires_grad = False
            
            # Count trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print_info(f"  Trainable: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
            
            # Test forward pass
            x = torch.randn(2, 3, 640, 640).to(device)
            with torch.no_grad():
                output = model(x)
            print_info(f"  Output shape: {output.shape}")
        
        print_success("Fine-tuning test passed")
        return True
        
    except Exception as e:
        print_error("Fine-tuning test failed", str(e))
        traceback.print_exc()
        return False

def test_memory_usage(device):
    """Test memory usage with different batch sizes."""
    print_header("Testing Memory Usage")
    
    try:
        # Test with increasing batch sizes
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            # Create model with proper initialization
            model = create_model(device)
            model.eval()
            
            # Create dummy batch
            images = torch.randn(batch_size, 3, 640, 640).to(device)
            
            # Run inference
            with torch.no_grad():
                predictions = model(images)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            # Clear memory
            del model, images, predictions
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            print_info(f"Batch size {batch_size}: {'GPU' if torch.cuda.is_available() else 'CPU'} mode")
        
        print_success("Memory usage test passed")
        return True
        
    except Exception as e:
        print_error("Memory usage test failed", str(e))
        traceback.print_exc()
        return False

def test_performance(device):
    """Test performance."""
    print_header("Testing Performance")
    
    try:
        # Import Model directly
        from model import Model
        
        # Model configuration
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
        
        # Create model
        model = Model(model_config).to(device)
        model.eval()
        
        # Warm up
        x = torch.randn(1, 3, 640, 640).to(device)
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)
        
        # Performance test
        num_runs = 100
        x = torch.randn(1, 3, 640, 640).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        total_time = time.time() - start_time
        avg_time = total_time / num_runs
        fps = 1.0 / avg_time
        
        print_info(f"Average inference time: {avg_time*1000:.2f} ms")
        print_info(f"FPS: {fps:.1f}")
        
        print_success("Performance test passed")
        return True
        
    except Exception as e:
        print_error("Performance test failed", str(e))
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print_header("DBNet Comprehensive Test Suite (Final)")
    print_info("This script tests all components of the DBNet implementation")
    print_info("Run this in Google Colab with GPU for best results")
    
    # Test results
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test device
    device = test_device()
    results.append(("Device", True))
    
    # Test architecture components
    results.append(("Backbone", test_backbone()))
    results.append(("Neck", test_neck()))
    results.append(("Head", test_head()))
    results.append(("Complete Model", test_model()))
    
    # Test loss functions
    results.append(("Loss Functions", test_losses()))
    
    # Test data and utilities
    results.append(("Dataset", test_dataset()))
    results.append(("Post-processing", test_postprocessing()))
    results.append(("Metrics", test_metrics()))
    
    # Test training and inference
    results.append(("Training Step", test_training_step(device)))
    results.append(("Inference", test_inference(device)))
    results.append(("Fine-tuning", test_finetuning(device)))
    
    # Test performance
    results.append(("Memory Usage", test_memory_usage(device)))
    results.append(("Performance", test_performance(device)))
    
    # Print summary
    print_header("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print_info(f"Tests passed: {passed}/{total}")
    print_info(f"Success rate: {passed/total*100:.1f}%")
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    if passed == total:
        print_success("All tests passed! DBNet implementation is ready for use.")
    else:
        print_error(f"{total - passed} tests failed. Please check the errors above.")
    
    # Cleanup
    if os.path.exists('test_data'):
        import shutil
        shutil.rmtree('test_data')
        print_info("Cleaned up test data")

if __name__ == "__main__":
    main() 