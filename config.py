# config.py
from addict import Dict

def get_default_config():
    """Get default configuration for DBNet."""
    config = Dict()
    
    # Model configuration
    config.model = {
        'backbone': {
            'type': 'MobileNetV3',
            'pretrained': True,
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
    
    # Training configuration
    config.train = {
        'batch_size': 16,
        'learning_rate': 0.007,
        'max_epochs': 1200,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'lr_power': 0.9,
        'save_interval': 10
    }
    
    # Loss configuration
    config.loss = {
        'alpha': 1.0,
        'beta': 10,
        'ohem_ratio': 3,
        'reduction': 'mean'
    }
    
    # Data configuration
    config.data = {
        'shrink_ratio': 0.4,
        'thresh_min': 0.3,
        'thresh_max': 0.7,
        'min_size': 3,
        'box_thresh': 0.7,
        'unclip_ratio': 1.5
    }
    
    # Augmentation configuration
    config.augmentation = {
        'train': {
            'RandomRotate90': {'p': 0.5},
            'RandomBrightnessContrast': {'p': 0.5},
            'HueSaturationValue': {'p': 0.5}
        }
    }
    
    return config

def get_test_config():
    """Get configuration for testing."""
    config = get_default_config()
    config.test = {
        'batch_size': 16,
        'score_threshold': 0.3,
        'min_size': 3,
        'box_thresh': 0.7,
        'unclip_ratio': 1.5
    }
    return config 