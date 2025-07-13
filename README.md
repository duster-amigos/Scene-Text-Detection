# DBNet Text Detection

A PyTorch implementation of [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947).

## Features
- Modern PyTorch (2.2+) implementation
- MobileNetV3 backbone for efficient inference
- FPEM-FFM feature fusion
- Batch processing support
- CPU and GPU support
- ICDAR2015 dataset support

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Download ICDAR2015 dataset
2. Organize the dataset as follows:
```
data/
    icdar2015/
        train/
            images/
            labels/
        test/
            images/
            labels/
```

## Fine-tuning

This project supports fine-tuning in two ways:

### 1. Dedicated Fine-tuning Script (`finetune.py`)

Use the dedicated fine-tuning script for advanced fine-tuning configurations:

```bash
# Basic fine-tuning with pre-trained weights
python finetune.py \
    --data_path /path/to/icdar2015 \
    --pretrained_weights checkpoints/best_model.pth \
    --learning_rate 0.001 \
    --max_epochs 100

# Fine-tune only the head (freeze backbone and neck)
python finetune.py \
    --data_path /path/to/icdar2015 \
    --pretrained_weights checkpoints/best_model.pth \
    --finetune_layers head_only \
    --learning_rate 0.001 \
    --max_epochs 50

# Fine-tune with different optimizer and scheduler
python finetune.py \
    --data_path /path/to/icdar2015 \
    --pretrained_weights checkpoints/best_model.pth \
    --optimizer adam \
    --scheduler cosine \
    --learning_rate 0.0005 \
    --max_epochs 100

# Fine-tune with frozen backbone
python finetune.py \
    --data_path /path/to/icdar2015 \
    --pretrained_weights checkpoints/best_model.pth \
    --freeze_backbone \
    --learning_rate 0.001 \
    --max_epochs 100
```

#### Fine-tuning Options:

- `--finetune_layers`: Choose which layers to fine-tune
  - `all`: Fine-tune all layers (default)
  - `head_only`: Fine-tune only the detection head
  - `neck_head`: Fine-tune neck and head (freeze backbone)
  - `backbone_head`: Fine-tune backbone and head (freeze neck)

- `--freeze_backbone`: Freeze backbone layers
- `--freeze_neck`: Freeze neck layers
- `--optimizer`: Choose optimizer (`sgd` or `adam`)
- `--scheduler`: Choose learning rate scheduler (`poly`, `step`, or `cosine`)

### 2. Resume Training with Fine-tuning (`train.py`)

Use the main training script with resume and fine-tuning options:

```bash
# Resume training from checkpoint
python train.py \
    --data_path /path/to/icdar2015 \
    --resume checkpoints/epoch_500.pth \
    --max_epochs 1200

# Resume with fine-tuning (reduced learning rate)
python train.py \
    --data_path /path/to/icdar2015 \
    --resume checkpoints/best_model.pth \
    --finetune \
    --learning_rate 0.0007 \
    --max_epochs 1200
```

### Fine-tuning Best Practices:

1. **Learning Rate**: Use 1/10th of the original learning rate for fine-tuning
2. **Epochs**: Fine-tune for fewer epochs (50-200) compared to full training
3. **Layer Selection**: 
   - For domain adaptation: Fine-tune all layers
   - For specific tasks: Fine-tune head only
   - For limited data: Freeze backbone, fine-tune neck and head
4. **Optimizer**: Adam often works better for fine-tuning than SGD
5. **Scheduler**: Cosine annealing is recommended for fine-tuning

### Example Fine-tuning Workflow:

```bash
# 1. Train the model from scratch
python train.py --data_path /path/to/icdar2015 --max_epochs 1200

# 2. Fine-tune on a specific domain
python finetune.py \
    --data_path /path/to/new_domain_data \
    --pretrained_weights checkpoints/best_model.pth \
    --finetune_layers all \
    --learning_rate 0.001 \
    --max_epochs 100

# 3. Further fine-tune only the head for a specific task
python finetune.py \
    --data_path /path/to/task_specific_data \
    --pretrained_weights finetune_checkpoints/best_finetuned_model.pth \
    --finetune_layers head_only \
    --learning_rate 0.0005 \
    --max_epochs 50
```

## Training

```bash
python train.py --data_path data/icdar2015 \
                --batch_size 16 \
                --learning_rate 0.007 \
                --max_epochs 1200 \
                --device cuda  # or cpu
```

## Testing

```bash
python test.py --data_path data/icdar2015/test \
               --weights path/to/weights.pth \
               --batch_size 16 \
               --device cuda  # or cpu
```

## Inference

```bash
python infer.py --image_path path/to/image.jpg \
                --weights path/to/weights.pth \
                --device cuda  # or cpu
```

## Model Architecture

- Backbone: MobileNetV3
- Neck: FPEM-FFM (Feature Pyramid Enhancement Module)
- Head: Differentiable Binarization Head

## Training Details

- Learning rate: 0.007
- Optimizer: SGD with momentum 0.9
- Weight decay: 1e-4
- Learning rate schedule: Poly decay
- Data augmentation: Random crop, rotate, color jittering

## References

- [DBNet Paper](https://arxiv.org/abs/1911.08947)
- [Official Implementation](https://github.com/WenmuZhou/DBNet.pytorch) 