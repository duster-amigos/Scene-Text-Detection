#!/usr/bin/env python3
"""
Debug script to test CUDA model creation step by step.
"""

import os
import sys
import torch
import torch.nn as nn
import traceback

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_cuda_availability():
    """Test CUDA availability and basic operations."""
    print("=" * 60)
    print("Testing CUDA Availability")
    print("=" * 60)
    
    try:
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name()}")
            print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Test basic CUDA operations
            x = torch.randn(2, 3, 64, 64)
            x_cuda = x.cuda()
            print(f"Basic tensor moved to CUDA: {x_cuda.device}")
            
            # Test simple model on CUDA
            simple_model = nn.Conv2d(3, 64, 3, padding=1)
            simple_model = simple_model.cuda()
            output = simple_model(x_cuda)
            print(f"Simple model output shape: {output.shape}")
            
            return True
        else:
            print("CUDA not available")
            return False
            
    except Exception as e:
        print(f"CUDA test failed: {e}")
        traceback.print_exc()
        return False

def test_backbone_cuda():
    """Test backbone on CUDA."""
    print("\n" + "=" * 60)
    print("Testing Backbone on CUDA")
    print("=" * 60)
    
    try:
        from backbone_mobilenetv3 import MobileNetV3
        
        # Create backbone on CPU first
        print("Creating backbone on CPU...")
        backbone = MobileNetV3(pretrained=False, in_channels=3)
        print("Backbone created on CPU successfully")
        
        # Move to CUDA
        print("Moving backbone to CUDA...")
        backbone = backbone.cuda()
        print("Backbone moved to CUDA successfully")
        
        # Test forward pass
        print("Testing forward pass...")
        x = torch.randn(2, 3, 640, 640).cuda()
        features = backbone(x)
        print(f"Backbone output shapes: {[f.shape for f in features]}")
        
        return True
        
    except Exception as e:
        print(f"Backbone CUDA test failed: {e}")
        traceback.print_exc()
        return False

def test_neck_cuda():
    """Test neck on CUDA."""
    print("\n" + "=" * 60)
    print("Testing Neck on CUDA")
    print("=" * 60)
    
    try:
        from neck_fpem_ffm import FPEM_FFM
        
        # Create neck on CPU first
        print("Creating neck on CPU...")
        neck = FPEM_FFM(in_channels=[24, 40, 96, 576], inner_channels=256)
        print("Neck created on CPU successfully")
        
        # Move to CUDA
        print("Moving neck to CUDA...")
        neck = neck.cuda()
        print("Neck moved to CUDA successfully")
        
        # Test forward pass
        print("Testing forward pass...")
        features = [
            torch.randn(2, 24, 80, 80).cuda(),
            torch.randn(2, 40, 40, 40).cuda(),
            torch.randn(2, 96, 20, 20).cuda(),
            torch.randn(2, 576, 20, 20).cuda()
        ]
        output = neck(features)
        print(f"Neck output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"Neck CUDA test failed: {e}")
        traceback.print_exc()
        return False

def test_head_cuda():
    """Test head on CUDA."""
    print("\n" + "=" * 60)
    print("Testing Head on CUDA")
    print("=" * 60)
    
    try:
        from head_DBHead import DBHead
        
        # Create head on CPU first
        print("Creating head on CPU...")
        head = DBHead(in_channels=1024, out_channels=2, k=50)
        print("Head created on CPU successfully")
        
        # Move to CUDA
        print("Moving head to CUDA...")
        head = head.cuda()
        print("Head moved to CUDA successfully")
        
        # Test forward pass
        print("Testing forward pass...")
        x = torch.randn(2, 1024, 160, 160).cuda()
        
        # Test training mode
        head.train()
        y_train = head(x)
        print(f"Training output shape: {y_train.shape}")
        
        # Test inference mode
        head.eval()
        y_eval = head(x)
        print(f"Inference output shape: {y_eval.shape}")
        
        return True
        
    except Exception as e:
        print(f"Head CUDA test failed: {e}")
        traceback.print_exc()
        return False

def test_complete_model_cuda():
    """Test complete model on CUDA."""
    print("\n" + "=" * 60)
    print("Testing Complete Model on CUDA")
    print("=" * 60)
    
    try:
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
        
        # Create model on CPU first
        print("Creating model on CPU...")
        model = Model(model_config)
        print("Model created on CPU successfully")
        
        # Initialize weights
        print("Initializing weights...")
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print("Weights initialized successfully")
        
        # Clear CUDA cache
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Move model to CUDA
        print("Moving model to CUDA...")
        model = model.cuda()
        print("Model moved to CUDA successfully")
        
        # Test forward pass
        print("Testing forward pass...")
        x = torch.randn(2, 3, 640, 640).cuda()
        
        # Test training mode
        model.train()
        y_train = model(x)
        print(f"Training output shape: {y_train.shape}")
        
        # Test inference mode
        model.eval()
        y_eval = model(x)
        print(f"Inference output shape: {y_eval.shape}")
        
        return True
        
    except Exception as e:
        print(f"Complete model CUDA test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all CUDA tests."""
    print("Starting CUDA Debug Tests")
    print("=" * 60)
    
    tests = [
        ("CUDA Availability", test_cuda_availability),
        ("Backbone CUDA", test_backbone_cuda),
        ("Neck CUDA", test_neck_cuda),
        ("Head CUDA", test_head_cuda),
        ("Complete Model CUDA", test_complete_model_cuda)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CUDA is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main() 