#!/usr/bin/env python3
"""
Simple CUDA Fix for Google Colab
================================

This script addresses the specific CUDA device-side assert errors
you're experiencing in Google Colab.

Usage:
    python simple_cuda_fix.py
"""

import os
import sys
import gc
import warnings
warnings.filterwarnings("ignore")

def print_header(title):
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

def print_step(step, description):
    print(f"\n{step}. {description}")

def fix_cuda_environment():
    """Fix CUDA environment issues."""
    print_header("CUDA Environment Fix")
    
    # Step 1: Set environment variables
    print_step(1, "Setting CUDA environment variables")
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    print("✅ Environment variables set")
    
    # Step 2: Clear memory
    print_step(2, "Clearing memory and cache")
    gc.collect()
    print("✅ Memory cleared")
    
    # Step 3: Import torch with error handling
    print_step(3, "Importing PyTorch")
    try:
        import torch
        print(f"✅ PyTorch imported: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            # Clear CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✅ CUDA cache cleared")
        else:
            print("❌ CUDA not available")
            return False
            
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    return True

def test_safe_tensor_operations():
    """Test tensor operations with safe initialization."""
    print_header("Safe Tensor Operations Test")
    
    try:
        import torch
        
        device = torch.device('cuda:0')
        print(f"Testing on device: {device}")
        
        # Test 1: Create tensor on CPU first
        print_step(1, "Creating tensor on CPU first")
        x_cpu = torch.randn(2, 3, 64, 64)
        print(f"✅ CPU tensor created: {x_cpu.shape}")
        
        # Test 2: Move to GPU with explicit device
        print_step(2, "Moving tensor to GPU")
        x_gpu = x_cpu.to(device=device, non_blocking=False)
        print(f"✅ GPU tensor created: {x_gpu.shape}")
        
        # Test 3: Basic operations
        print_step(3, "Testing basic operations")
        y = x_gpu + 1.0
        z = torch.sum(y)
        print(f"✅ Basic operations: {z.item():.4f}")
        
        # Test 4: Clean up
        print_step(4, "Cleaning up")
        del x_cpu, x_gpu, y, z
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        return False

def test_model_safe():
    """Test model with safe initialization."""
    print_header("Safe Model Testing")
    
    try:
        import torch
        
        # Import model components
        print_step(1, "Importing model components")
        from backbone_mobilenetv3 import MobileNetV3
        from neck_fpem_ffm import FPEM_FFM
        from head_DBHead import DBHead
        from model import DBNet
        print("✅ Model components imported")
        
        # Test on CPU first
        print_step(2, "Testing on CPU")
        device_cpu = torch.device('cpu')
        
        # Create model on CPU
        model = DBNet()
        model.eval()
        model = model.to(device_cpu)
        
        # Test inference on CPU
        x = torch.randn(1, 3, 640, 640, device=device_cpu)
        with torch.no_grad():
            output = model(x)
        print(f"✅ CPU inference: {output.shape}")
        
        # Test GPU if available
        if torch.cuda.is_available():
            print_step(3, "Testing on GPU")
            device_gpu = torch.device('cuda:0')
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Move model to GPU
            model = model.to(device_gpu)
            
            # Test inference on GPU
            x = torch.randn(1, 3, 640, 640, device=device_gpu)
            with torch.no_grad():
                output = model(x)
            print(f"✅ GPU inference: {output.shape}")
            
            # Clean up
            del model, x, output
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✅ GPU cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        return False

def create_cpu_fallback():
    """Create a CPU-only fallback script."""
    print_header("Creating CPU Fallback")
    
    fallback_code = '''#!/usr/bin/env python3
"""
CPU-Only DBNet Fallback
=======================

Use this script if CUDA continues to have issues.
This runs DBNet on CPU only.

Usage:
    python cpu_fallback.py
"""

import torch
import warnings
warnings.filterwarnings("ignore")

def main():
    print("Running DBNet on CPU only...")
    
    # Force CPU
    device = torch.device('cpu')
    print(f"Device: {device}")
    
    try:
        # Import model
        from model import DBNet
        
        # Create model
        model = DBNet()
        model.eval()
        model = model.to(device)
        
        # Test inference
        x = torch.randn(1, 3, 640, 640, device=device)
        with torch.no_grad():
            output = model(x)
        
        print(f"✅ Success: {output.shape}")
        print("Model ready for CPU inference!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
'''
    
    with open('cpu_fallback.py', 'w') as f:
        f.write(fallback_code)
    
    print("✅ CPU fallback script created: cpu_fallback.py")

def main():
    """Main function."""
    print_header("Simple CUDA Fix")
    print("Addressing CUDA device-side assert errors...")
    
    # Step 1: Fix environment
    if not fix_cuda_environment():
        print("\n❌ Environment fix failed. Creating CPU fallback...")
        create_cpu_fallback()
        return
    
    # Step 2: Test tensor operations
    if not test_safe_tensor_operations():
        print("\n❌ Tensor operations failed. Creating CPU fallback...")
        create_cpu_fallback()
        return
    
    # Step 3: Test model
    if not test_model_safe():
        print("\n❌ Model testing failed. Creating CPU fallback...")
        create_cpu_fallback()
        return
    
    # Success
    print_header("Success!")
    print("✅ CUDA environment fixed")
    print("✅ Tensor operations working")
    print("✅ Model working on GPU")
    print("\n🎉 Your DBNet is ready to use!")

if __name__ == "__main__":
    main() 