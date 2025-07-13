#!/usr/bin/env python3
"""
Demo script to showcase the beautiful logging system
"""

import time
import json
from utils import logger, print_device_info, print_config_summary, print_training_progress

def demo_basic_logging():
    """Demo basic logging functions"""
    logger.header("Beautiful Logging Demo")
    
    logger.info("This is a basic info message")
    logger.success("This is a success message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    logger.subheader("Different Types of Information")
    
    logger.model_info("Model-related information")
    logger.data_info("Data-related information")
    logger.config_info("Configuration information")
    logger.training_info("Training-related information")
    logger.validation_info("Validation-related information")
    logger.testing_info("Testing-related information")
    logger.inference_info("Inference-related information")
    logger.checkpoint_info("Checkpoint-related information")
    logger.gpu_info("GPU-related information")
    logger.memory_info("Memory-related information")
    logger.time_info("Time-related information")

def demo_hierarchy():
    """Demo hierarchical logging"""
    logger.header("Hierarchical Logging Demo")
    
    logger.section("Main Section")
    logger.info("This is inside the main section")
    
    logger.section("Sub Section")
    logger.info("This is inside a sub section")
    logger.substep("This is a substep")
    logger.substep("Another substep")
    logger.end_section()
    
    logger.info("Back to main section")
    logger.end_section()

def demo_tables_and_metrics():
    """Demo table and metrics formatting"""
    logger.header("Tables and Metrics Demo")
    
    # Configuration table
    config_data = {
        "Learning Rate": 0.001,
        "Batch Size": 8,
        "Epochs": 100,
        "Image Size": 640,
        "Model Type": "MobileNetV3"
    }
    logger.table(config_data, "Training Configuration")
    
    # Metrics
    metrics_data = {
        "Training Loss": 0.8234,
        "Validation Loss": 0.7890,
        "Accuracy": 0.9456,
        "Precision": 0.9234,
        "Recall": 0.9567
    }
    logger.metrics(metrics_data, "Model Performance")
    
    # Summary
    summary_data = {
        "Total Parameters": "2,552,482",
        "Model Size": "9.7 MB",
        "Training Time": "2h 15m",
        "Best Epoch": 45
    }
    logger.summary(summary_data, "Training Summary")

def demo_progress():
    """Demo progress tracking"""
    logger.header("Progress Tracking Demo")
    
    # Simulate training progress
    for epoch in range(1, 6):
        train_loss = 1.0 - (epoch * 0.15)
        val_loss = 1.0 - (epoch * 0.12)
        
        print_training_progress(
            epoch=epoch,
            total_epochs=10,
            train_loss=train_loss,
            val_loss=val_loss,
            lr=0.001 * (0.9 ** epoch)
        )
        time.sleep(0.5)

def demo_device_info():
    """Demo device information"""
    logger.header("Device Information Demo")
    print_device_info()

def demo_config_summary():
    """Demo configuration summary"""
    logger.header("Configuration Summary Demo")
    
    # Sample configuration
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
        "training": {
            "epochs": 100,
            "batch_size": 8,
            "learning_rate": 0.001,
            "weight_decay": 0.0001
        },
        "data": {
            "train_images": "data/train/images",
            "train_labels": "data/train/annotations",
            "val_images": "data/val/images",
            "val_labels": "data/val/annotations"
        },
        "loss": {
            "alpha": 1.0,
            "beta": 10.0,
            "ohem_ratio": 3.0
        }
    }
    
    print_config_summary(config)

def main():
    """Main demo function"""
    try:
        # Run all demos
        demo_basic_logging()
        time.sleep(1)
        
        demo_hierarchy()
        time.sleep(1)
        
        demo_tables_and_metrics()
        time.sleep(1)
        
        demo_progress()
        time.sleep(1)
        
        demo_device_info()
        time.sleep(1)
        
        demo_config_summary()
        
        logger.success("All demos completed successfully!")
        logger.elapsed_time()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == '__main__':
    main() 