# DBNet Text Detection

A PyTorch implementation of DBNet (Differentiable Binarization Network) for real-time scene text detection, featuring MobileNetV3-Small backbone, FPEM-FFM neck architecture, and DBHead with comprehensive training and evaluation pipelines.

## Overview

This implementation follows the architecture described in "Real-time Scene Text Detection with Differentiable Binarization" (Liao et al., 2020). The model combines efficient feature extraction with differentiable binarization for accurate text detection in natural scenes.

**Architecture Components:**
- **Backbone**: MobileNetV3-Small with ImageNet pretraining
- **Neck**: FPEM-FFM (Feature Pyramid Enhancement Module with Feature Fusion Module)
- **Head**: DBHead with differentiable binarization

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd DBNET
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
DBNET/
├── backbone_mobilenetv3.py  # MobileNetV3 backbone implementation
├── neck_fpem_ffm.py         # FPEM-FFM neck architecture
├── neck_fpn.py              # Alternative FPN neck module
├── head_DBHead.py           # DBHead with differentiable binarization
├── losses.py                # Loss function implementations
├── model.py                 # Main model architecture
├── build.py                 # Model building utilities
├── dataset.py               # ICDAR 2015 dataset loader
├── train.py                 # Training script with anomaly detection
├── inference.py             # Inference and visualization script
├── test.py                  # Evaluation and metrics script
├── utils.py                 # Utility functions and logging
├── config.json              # Configuration parameters
├── requirements.txt         # Python dependencies
└── README.md               # Documentation
```

## Data Preparation

### Dataset Organization

Organize your dataset in the following structure:

```
data/icdar2015/
├── train/
│   ├── images/
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   └── ...
│   └── labels/
│       ├── img_1.txt
│       ├── img_2.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Annotation Format

Each annotation file should contain one text region per line in ICDAR 2015 format:
```
x1,y1,x2,y2,x3,y3,x4,y4,text
```

Where coordinates represent the four corners of the text bounding box in clockwise order.

## Configuration

The `config.json` file contains all model and training parameters:

**Model Configuration:**
- Backbone type and pretraining settings
- Neck architecture parameters (inner channels, FPEM repeats)
- Head configuration (output channels, steepness parameter k)

**Training Configuration:**
- Learning rate, batch size, and number of epochs
- Learning rate scheduling parameters
- Data augmentation settings

**Loss Configuration:**
- Alpha and beta weights for loss components
- OHEM ratio for balanced cross-entropy
- Epsilon values for numerical stability

## Usage

### Training

1. Prepare your dataset in ICDAR 2015 format
2. Update data paths in `config.json`
3. Start training:

```bash
python train.py --config config.json
```

**Training Features:**
- Automatic checkpoint saving and resuming
- Learning rate scheduling with step decay
- Comprehensive progress tracking
- GPU memory optimization
- Validation during training (optional)

### Inference

Detect text in individual images:

```bash
python inference.py \
    --model checkpoints/dbnet_checkpoint_best.pth \
    --config config.json \
    --image path/to/image.jpg \
    --output result.jpg
```

### Evaluation

Evaluate model performance on test datasets:

```bash
python test.py \
    --model checkpoints/dbnet_checkpoint_best.pth \
    --config config.json \
    --test_images data/icdar2015/test/images \
    --test_labels data/icdar2015/test/labels \
    --iou_threshold 0.5
```

## Model Architecture Details

### Backbone: MobileNetV3-Small

- Efficient depthwise separable convolutions
- Squeeze-and-Excitation blocks
- Outputs 4 feature maps: [24, 40, 96, 576] channels
- Pre-trained on ImageNet for better convergence

### Neck: FPEM-FFM

**Feature Pyramid Enhancement Module (FPEM):**
- Top-down and bottom-up feature enhancement
- Separable convolutions for efficiency
- Configurable repeat count (default: 2)

**Feature Fusion Module (FFM):**
- Multi-scale feature aggregation
- Bilinear upsampling for spatial alignment
- Channel concatenation for final representation

### Head: DBHead

**Binarization Module:**
- Convolutional layers with batch normalization
- Transposed convolutions for upsampling
- Sigmoid activation for probability maps

**Threshold Module:**
- Similar architecture to binarization module
- Generates adaptive threshold maps

**Step Function:**
- Differentiable approximation: `sigmoid(k * (x - y))`
- Steepness parameter k = 50 (configurable)

## Loss Function

The model employs a multi-component loss function:

**Balanced Cross-Entropy Loss:**
- Handles class imbalance in text detection
- Online Hard Example Mining (OHEM) with ratio 3.0
- Applied to shrink maps

**Masked L1 Loss:**
- Regression loss for threshold maps
- Masked to focus on text regions only

**Dice Loss:**
- Segmentation loss for binary maps
- Applied only during training
- Improves boundary accuracy

**Total Loss:**
```
L = α * L_shrink + β * L_threshold + L_binary
```
Where α = 1.0 and β = 10.0

## Training Guidelines

### Hyperparameter Recommendations

**Learning Rate Schedule:**
- Initial learning rate: 0.001
- Step size: 30 epochs
- Gamma: 0.1 (10x reduction)

**Data Augmentation:**
- Random horizontal flipping
- Color jittering
- Image resizing to 640x640

**Batch Size:**
- Recommended: 8-16 (GPU memory dependent)
- Adjust based on available hardware

### Optimization Tips

1. **Data Quality**: Ensure high-quality annotations and diverse training data
2. **Learning Rate**: Start with 0.001 and monitor validation loss
3. **Checkpointing**: Save best model based on validation performance
4. **Regularization**: Weight decay of 0.0001 helps prevent overfitting
5. **Image Size**: 640x640 provides good balance of speed and accuracy

### GPU Configuration

The model automatically detects and utilizes available GPUs. To force CPU usage:
```bash
CUDA_VISIBLE_DEVICES="" python train.py --config config.json
```

## Technical Implementation Notes

### Autograd Compatibility

The implementation includes comprehensive fixes for PyTorch autograd compatibility:
- All ReLU operations use `inplace=False`
- Non-in-place tensor operations in FPEM accumulation
- Proper tensor slicing in loss functions

### Memory Optimization

- Efficient data loading with proper worker configuration
- Gradient checkpointing for large models
- Optimized batch processing

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@inproceedings{liao2020real,
  title={Real-time scene text detection with differentiable binarization},
  author={Liao, Minghui and Wan, Zhaoyi and Yao, Cong and Chen, Kai and Bai, Xiang},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={34},
  number={07},
  pages={11474--11481},
  year={2020}
}
```