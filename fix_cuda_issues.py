#!/usr/bin/env python3
"""
Comprehensive fix for CUDA device-side assert issues.
"""

import os
import sys
import torch
import torch.nn as nn
import traceback

# Set environment variables to help with debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def safe_weight_init(module):
    """Safely initialize weights for any module."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def fix_backbone():
    """Fix backbone CUDA issues."""
    print("=" * 60)
    print("Fixing Backbone CUDA Issues")
    print("=" * 60)
    
    try:
        from backbone_mobilenetv3 import MobileNetV3
        
        # Create backbone with safe initialization
        print("Creating backbone...")
        backbone = MobileNetV3(pretrained=False, in_channels=3)
        
        # Apply safe weight initialization
        print("Initializing weights safely...")
        backbone.apply(safe_weight_init)
        
        # Test on CPU first
        print("Testing on CPU...")
        x_cpu = torch.zeros(2, 3, 640, 640, dtype=torch.float32)
        features_cpu = backbone(x_cpu)
        print(f"CPU features: {[f.shape for f in features_cpu]}")
        
        # Move to CUDA with explicit device and synchronization
        print("Moving to CUDA...")
        device = torch.device('cuda:0')
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Move model to device
        backbone = backbone.to(device)
        torch.cuda.synchronize()
        
        # Test on CUDA
        print("Testing on CUDA...")
        x_cuda = torch.zeros(2, 3, 640, 640, dtype=torch.float32).to(device)
        features_cuda = backbone(x_cuda)
        print(f"CUDA features: {[f.shape for f in features_cuda]}")
        
        print("✅ Backbone CUDA fix successful!")
        return True
        
    except Exception as e:
        print(f"❌ Backbone fix failed: {e}")
        traceback.print_exc()
        return False

def fix_neck():
    """Fix neck CUDA issues."""
    print("\n" + "=" * 60)
    print("Fixing Neck CUDA Issues")
    print("=" * 60)
    
    try:
        from neck_fpem_ffm import FPEM_FFM
        
        # Create neck with safe initialization
        print("Creating neck...")
        neck = FPEM_FFM(in_channels=[24, 40, 96, 576], inner_channels=256)
        
        # Apply safe weight initialization
        print("Initializing weights safely...")
        neck.apply(safe_weight_init)
        
        # Test on CPU first
        print("Testing on CPU...")
        features_cpu = [
            torch.zeros(2, 24, 80, 80, dtype=torch.float32),
            torch.zeros(2, 40, 40, 40, dtype=torch.float32),
            torch.zeros(2, 96, 20, 20, dtype=torch.float32),
            torch.zeros(2, 576, 20, 20, dtype=torch.float32)
        ]
        output_cpu = neck(features_cpu)
        print(f"CPU output: {output_cpu.shape}")
        
        # Move to CUDA
        print("Moving to CUDA...")
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        neck = neck.to(device)
        torch.cuda.synchronize()
        
        # Test on CUDA
        print("Testing on CUDA...")
        features_cuda = [
            torch.zeros(2, 24, 80, 80, dtype=torch.float32).to(device),
            torch.zeros(2, 40, 40, 40, dtype=torch.float32).to(device),
            torch.zeros(2, 96, 20, 20, dtype=torch.float32).to(device),
            torch.zeros(2, 576, 20, 20, dtype=torch.float32).to(device)
        ]
        output_cuda = neck(features_cuda)
        print(f"CUDA output: {output_cuda.shape}")
        
        print("✅ Neck CUDA fix successful!")
        return True
        
    except Exception as e:
        print(f"❌ Neck fix failed: {e}")
        traceback.print_exc()
        return False

def fix_head():
    """Fix head CUDA issues."""
    print("\n" + "=" * 60)
    print("Fixing Head CUDA Issues")
    print("=" * 60)
    
    try:
        from head_DBHead import DBHead
        
        # Create head with safe initialization
        print("Creating head...")
        head = DBHead(in_channels=1024, out_channels=2, k=50)
        
        # Apply safe weight initialization
        print("Initializing weights safely...")
        head.apply(safe_weight_init)
        
        # Test on CPU first
        print("Testing on CPU...")
        x_cpu = torch.zeros(2, 1024, 160, 160, dtype=torch.float32)
        
        head.train()
        y_train_cpu = head(x_cpu)
        print(f"CPU training output: {y_train_cpu.shape}")
        
        head.eval()
        y_eval_cpu = head(x_cpu)
        print(f"CPU inference output: {y_eval_cpu.shape}")
        
        # Move to CUDA
        print("Moving to CUDA...")
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        head = head.to(device)
        torch.cuda.synchronize()
        
        # Test on CUDA
        print("Testing on CUDA...")
        x_cuda = torch.zeros(2, 1024, 160, 160, dtype=torch.float32).to(device)
        
        head.train()
        y_train_cuda = head(x_cuda)
        print(f"CUDA training output: {y_train_cuda.shape}")
        
        head.eval()
        y_eval_cuda = head(x_cuda)
        print(f"CUDA inference output: {y_eval_cuda.shape}")
        
        print("✅ Head CUDA fix successful!")
        return True
        
    except Exception as e:
        print(f"❌ Head fix failed: {e}")
        traceback.print_exc()
        return False

def fix_complete_model():
    """Fix complete model CUDA issues."""
    print("\n" + "=" * 60)
    print("Fixing Complete Model CUDA Issues")
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
        
        # Create model with safe initialization
        print("Creating model...")
        model = Model(model_config)
        
        # Apply safe weight initialization
        print("Initializing weights safely...")
        model.apply(safe_weight_init)
        
        # Test on CPU first
        print("Testing on CPU...")
        x_cpu = torch.zeros(2, 3, 640, 640, dtype=torch.float32)
        
        model.train()
        y_train_cpu = model(x_cpu)
        print(f"CPU training output: {y_train_cpu.shape}")
        
        model.eval()
        y_eval_cpu = model(x_cpu)
        print(f"CPU inference output: {y_eval_cpu.shape}")
        
        # Move to CUDA
        print("Moving to CUDA...")
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        model = model.to(device)
        torch.cuda.synchronize()
        
        # Test on CUDA
        print("Testing on CUDA...")
        x_cuda = torch.zeros(2, 3, 640, 640, dtype=torch.float32).to(device)
        
        model.train()
        y_train_cuda = model(x_cuda)
        print(f"CUDA training output: {y_train_cuda.shape}")
        
        model.eval()
        y_eval_cuda = model(x_cuda)
        print(f"CUDA inference output: {y_eval_cuda.shape}")
        
        print("✅ Complete model CUDA fix successful!")
        return True
        
    except Exception as e:
        print(f"❌ Complete model fix failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all CUDA fixes."""
    print("Starting Comprehensive CUDA Fixes")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Run fixes
    fixes = [
        ("Backbone", fix_backbone),
        ("Neck", fix_neck),
        ("Head", fix_head),
        ("Complete Model", fix_complete_model)
    ]
    
    results = []
    for name, fix_func in fixes:
        try:
            result = fix_func()
            results.append((name, result))
            if result:
                print(f"✅ {name}: FIXED")
            else:
                print(f"❌ {name}: FAILED")
        except Exception as e:
            print(f"❌ {name}: ERROR - {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Fix Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "FIXED" if result else "FAILED"
        print(f"{name}: {status}")
    
    print(f"\nOverall: {passed}/{total} components fixed")
    
    if passed == total:
        print("\n🎉 All CUDA issues fixed! Model is ready for use.")
    else:
        print("\n⚠️  Some issues remain. Check the error messages above.")

if __name__ == "__main__":
    main() 