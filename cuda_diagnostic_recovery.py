#!/usr/bin/env python3
"""
CUDA Diagnostic and Recovery Script for Google Colab
====================================================

This script performs comprehensive diagnostics and recovery steps for CUDA issues
in Google Colab environment. It includes:

1. Environment diagnostics
2. PyTorch reinstallation with specific versions
3. CUDA cache clearing and memory management
4. Safe tensor operations testing
5. Model component testing with fallbacks

Usage:
    python cuda_diagnostic_recovery.py
"""

import os
import sys
import subprocess
import time
import gc
import warnings
from typing import Optional, Dict, Any

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'-' * 40}")
    print(f" {title}")
    print(f"{'-' * 40}")

def run_command(command: str, description: str = "") -> bool:
    """Run a shell command and return success status."""
    if description:
        print(f"Running: {description}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def check_environment():
    """Check the current environment setup."""
    print_header("Environment Diagnostics")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check CUDA environment variables
    cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH']
    for var in cuda_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")
    
    # Check GPU info
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
        else:
            print("CUDA not available")
    except Exception as e:
        print(f"Error checking CUDA: {e}")

def reinstall_pytorch():
    """Reinstall PyTorch with specific versions for better CUDA compatibility."""
    print_header("PyTorch Reinstallation")
    
    # Uninstall current PyTorch
    print("Uninstalling current PyTorch...")
    run_command("pip uninstall torch torchvision torchaudio -y", "Uninstall PyTorch")
    
    # Install specific PyTorch version for better CUDA compatibility
    print("Installing PyTorch with CUDA 11.8...")
    success = run_command(
        "pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118",
        "Install PyTorch 2.0.1 with CUDA 11.8"
    )
    
    if not success:
        print("Trying alternative PyTorch version...")
        run_command(
            "pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu116",
            "Install PyTorch 1.13.1 with CUDA 11.6"
        )
    
    # Verify installation
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
    except Exception as e:
        print(f"Error verifying PyTorch: {e}")

def clear_cuda_cache():
    """Clear CUDA cache and memory."""
    print_header("CUDA Cache Clearing")
    
    try:
        import torch
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            print("Clearing PyTorch CUDA cache...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✅ PyTorch cache cleared")
        
        # Force garbage collection
        print("Running garbage collection...")
        gc.collect()
        print("✅ Garbage collection completed")
        
        # Clear system cache (if possible)
        run_command("sync", "Sync filesystem")
        
    except Exception as e:
        print(f"Error clearing cache: {e}")

def test_basic_cuda():
    """Test basic CUDA operations with error handling."""
    print_header("Basic CUDA Testing")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False
        
        device = torch.device('cuda:0')
        print(f"Using device: {device}")
        
        # Test 1: Simple tensor creation
        print("Test 1: Creating simple tensor...")
        try:
            x = torch.randn(2, 3, device=device)
            print(f"✅ Simple tensor created: {x.shape}")
        except Exception as e:
            print(f"❌ Simple tensor failed: {e}")
            return False
        
        # Test 2: Basic operations
        print("Test 2: Basic operations...")
        try:
            y = x + 1.0
            z = torch.sum(y)
            print(f"✅ Basic operations successful: {z.item()}")
        except Exception as e:
            print(f"❌ Basic operations failed: {e}")
            return False
        
        # Test 3: Memory operations
        print("Test 3: Memory operations...")
        try:
            del x, y, z
            torch.cuda.empty_cache()
            print("✅ Memory operations successful")
        except Exception as e:
            print(f"❌ Memory operations failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

def test_model_components():
    """Test model components with safe initialization."""
    print_header("Model Component Testing")
    
    try:
        # Import model components
        from backbone_mobilenetv3 import MobileNetV3
        from neck_fpem_ffm import FPEM_FFM
        from head_DBHead import DBHead
        from model import DBNet
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Testing on device: {device}")
        
        # Test Backbone
        print_section("Testing Backbone")
        try:
            backbone = MobileNetV3()
            backbone.eval()
            
            # Test on CPU first
            x = torch.randn(1, 3, 640, 640)
            with torch.no_grad():
                features = backbone(x)
            print(f"✅ Backbone CPU test: {len(features)} features")
            
            # Move to GPU if available
            if device.type == 'cuda':
                backbone = backbone.to(device)
                x = x.to(device)
                with torch.no_grad():
                    features = backbone(x)
                print(f"✅ Backbone GPU test: {len(features)} features")
            
        except Exception as e:
            print(f"❌ Backbone test failed: {e}")
        
        # Test Neck
        print_section("Testing Neck")
        try:
            neck = FPEM_FFM(in_channels=24, out_channels=128)
            neck.eval()
            
            # Test on CPU first
            x = torch.randn(1, 24, 80, 80)
            with torch.no_grad():
                output = neck(x)
            print(f"✅ Neck CPU test: {output.shape}")
            
            # Move to GPU if available
            if device.type == 'cuda':
                neck = neck.to(device)
                x = x.to(device)
                with torch.no_grad():
                    output = neck(x)
                print(f"✅ Neck GPU test: {output.shape}")
            
        except Exception as e:
            print(f"❌ Neck test failed: {e}")
        
        # Test Head
        print_section("Testing Head")
        try:
            head = DBHead(in_channels=128, out_channels=2)
            head.eval()
            
            # Test on CPU first
            x = torch.randn(1, 128, 80, 80)
            with torch.no_grad():
                output = head(x)
            print(f"✅ Head CPU test: {output.shape}")
            
            # Move to GPU if available
            if device.type == 'cuda':
                head = head.to(device)
                x = x.to(device)
                with torch.no_grad():
                    output = head(x)
                print(f"✅ Head GPU test: {output.shape}")
            
        except Exception as e:
            print(f"❌ Head test failed: {e}")
        
        # Test Complete Model
        print_section("Testing Complete Model")
        try:
            model = DBNet()
            model.eval()
            
            # Test on CPU first
            x = torch.randn(1, 3, 640, 640)
            with torch.no_grad():
                output = model(x)
            print(f"✅ Complete model CPU test: {output.shape}")
            
            # Move to GPU if available
            if device.type == 'cuda':
                model = model.to(device)
                x = x.to(device)
                with torch.no_grad():
                    output = model(x)
                print(f"✅ Complete model GPU test: {output.shape}")
            
        except Exception as e:
            print(f"❌ Complete model test failed: {e}")
        
    except Exception as e:
        print(f"❌ Model component testing failed: {e}")

def create_fallback_script():
    """Create a fallback script for CPU-only operation."""
    print_header("Creating Fallback Script")
    
    fallback_script = '''#!/usr/bin/env python3
"""
Fallback CPU-Only DBNet Script
==============================

This script runs DBNet on CPU only, bypassing CUDA issues.
Use this if CUDA continues to have problems.

Usage:
    python fallback_cpu_only.py
"""

import torch
import warnings
warnings.filterwarnings("ignore")

def run_cpu_only():
    """Run DBNet on CPU only."""
    print("Running DBNet on CPU only...")
    
    # Force CPU device
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    try:
        # Import and test model components
        from backbone_mobilenetv3 import MobileNetV3
        from neck_fpem_ffm import FPEM_FFM
        from head_DBHead import DBHead
        from model import DBNet
        
        # Create model
        model = DBNet()
        model.eval()
        model = model.to(device)
        
        # Test inference
        x = torch.randn(1, 3, 640, 640, device=device)
        with torch.no_grad():
            output = model(x)
        
        print(f"✅ CPU inference successful: {output.shape}")
        print("Model is ready for CPU-only operation")
        
        return True
        
    except Exception as e:
        print(f"❌ CPU-only test failed: {e}")
        return False

if __name__ == "__main__":
    run_cpu_only()
'''
    
    with open('fallback_cpu_only.py', 'w') as f:
        f.write(fallback_script)
    
    print("✅ Fallback script created: fallback_cpu_only.py")

def main():
    """Main diagnostic and recovery function."""
    print_header("CUDA Diagnostic and Recovery")
    print("This script will diagnose and attempt to fix CUDA issues in Google Colab.")
    
    # Step 1: Environment check
    check_environment()
    
    # Step 2: Clear cache
    clear_cuda_cache()
    
    # Step 3: Test basic CUDA
    if not test_basic_cuda():
        print("\n⚠️  Basic CUDA test failed. Attempting PyTorch reinstallation...")
        reinstall_pytorch()
        
        # Test again after reinstallation
        print("\nTesting CUDA after reinstallation...")
        if not test_basic_cuda():
            print("\n❌ CUDA issues persist. Creating fallback script...")
            create_fallback_script()
            print("\n💡 Recommendation: Use the fallback script for CPU-only operation")
            return
    
    # Step 4: Test model components
    print("\nTesting model components...")
    test_model_components()
    
    # Step 5: Final summary
    print_header("Diagnostic Summary")
    print("✅ Environment diagnostics completed")
    print("✅ CUDA cache cleared")
    print("✅ Basic CUDA operations tested")
    print("✅ Model components tested")
    print("\n🎉 Diagnostic and recovery completed!")

if __name__ == "__main__":
    main() 