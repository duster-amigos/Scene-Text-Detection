# DBNet Text Detection

A PyTorch implementation of DBNet (Differentiable Binarization Network) for text detection using MobileNetV3-Small backbone, FPEM-FFM neck, and DBHead.

## Overview

This project implements the DBNet architecture as described in the paper "Real-time Scene Text Detection with Differentiable Binarization" (https://arxiv.org/pdf/1911.08947). The implementation uses:

- **Backbone**: MobileNetV3-Small (from torchvision)
- **Neck**: FPEM-FFM (Feature Pyramid Enhancement Module with Feature Fusion Module)
- **Head**: DBHead (Differentiable Binarization Head)

## Features

- ✅ Modern PyTorch implementation (2.0+)
- ✅ GPU/CPU support with automatic device detection
- ✅ Batch processing for training, inference, and testing
- ✅ Comprehensive logging and progress tracking
- ✅ Checkpoint saving and resuming
- ✅ Data augmentation for training
- ✅ Evaluation metrics (Precision, Recall, F1-Score)
- ✅ Visualization of detection results
- ✅ ICDAR 2015 dataset support
- ✅ Fine-tuning support

## Installation

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
├── backbone_mobilenetv3.py  # MobileNetV3 backbone
├── neck_fpem_ffm.py         # FPEM-FFM neck module
├── neck_fpn.py              # FPN neck module (alternative)
├── head_DBHead.py           # DBHead module
├── losses.py                # Loss functions
├── model.py                 # Main model architecture
├── build.py                 # Model building utilities
├── dataset.py               # ICDAR 2015 dataset loader
├── train.py                 # Training script
├── inference.py             # Inference script
├── test.py                  # Evaluation script
├── config.json              # Configuration file
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Data Preparation

### ICDAR 2015 Format

The dataset should be organized as follows:

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

Each `.txt` file should contain one line per text region in the format:
```
x1,y1,x2,y2,x3,y3,x4,y4,text
```

Where `(x1,y1)`, `(x2,y2)`, `(x3,y3)`, `(x4,y4)` are the four corner coordinates of the text region.

## Configuration

Edit `config.json` to customize:

- **Model architecture**: Backbone, neck, and head parameters
- **Training parameters**: Learning rate, batch size, epochs, etc.
- **Data paths**: Training, validation, and test data directories
- **Inference parameters**: Thresholds and post-processing settings

## Usage

### Training

1. Prepare your dataset in ICDAR 2015 format
2. Update the data paths in `config.json`
3. Start training:

```bash
python train.py --config config.json
```

Training features:
- Automatic checkpoint saving
- Learning rate scheduling
- Progress tracking with tqdm
- GPU/CPU support
- Resume from checkpoint

### Inference

Detect text in a single image:

```bash
python inference.py \
    --model checkpoints/dbnet_checkpoint_best.pth \
    --config config.json \
    --image path/to/image.jpg \
    --output result.jpg
```

### Evaluation

Evaluate model performance on test dataset:

```bash
python test.py \
    --model checkpoints/dbnet_checkpoint_best.pth \
    --config config.json \
    --test_images data/icdar2015/test/images \
    --test_labels data/icdar2015/test/labels \
    --iou_threshold 0.5
```

## Model Architecture

### Backbone: MobileNetV3-Small
- Lightweight and efficient backbone
- Pre-trained on ImageNet
- Outputs 4 feature maps with channels [24, 40, 96, 576]

### Neck: FPEM-FFM
- Feature Pyramid Enhancement Module (FPEM)
- Feature Fusion Module (FFM)
- Enhances multi-scale feature representation
- Configurable inner channels and repeat count

### Head: DBHead
- Differentiable Binarization head
- Generates shrink maps and threshold maps
- Produces binary maps during training
- Configurable steepness parameter (k)

## Loss Function

The model uses a combination of losses:
- **Balanced Cross-Entropy Loss**: For shrink maps
- **Masked L1 Loss**: For threshold maps  
- **Dice Loss**: For binary maps (training only)

## Performance

Expected performance on ICDAR 2015:
- Precision: ~85-90%
- Recall: ~80-85%
- F1-Score: ~82-87%

*Note: Actual performance depends on training data quality and hyperparameter tuning.*

## Training Tips

1. **Data Augmentation**: The training script includes random horizontal flipping and color jittering
2. **Learning Rate**: Start with 0.001 and reduce by 0.1 every 30 epochs
3. **Batch Size**: Adjust based on your GPU memory (8-16 recommended)
4. **Image Size**: 640x640 is recommended for good performance
5. **Checkpointing**: Save best model based on validation loss

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image size
2. **Import Errors**: Ensure all dependencies are installed
3. **Data Loading Issues**: Check data paths and format in config.json
4. **Poor Performance**: Try adjusting learning rate or data augmentation

### GPU Usage

The model automatically detects and uses GPU if available. To force CPU usage:
```bash
CUDA_VISIBLE_DEVICES="" python train.py --config config.json
```

## Citation

If you use this implementation, please cite the original paper:

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

## License

This project is for research purposes. Please refer to the original paper and repository for licensing information.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests. 