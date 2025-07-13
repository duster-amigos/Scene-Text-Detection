import argparse
import os
import torch
from anyconfig import load
from utils import parse_config
import copy

def init_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='DBNet.pytorch')
    parser.add_argument('--config_file', default='config/open_dataset_resnet18_FPN_DBhead_polyLR.yaml', type=str)
    parser.add_argument('--local_rank', default=0, type=int, help='Use distributed training')
    return parser.parse_args()

def main(config):
    """Main function to set up and start training."""
    # from models import build_model, build_loss
    from data_loader import get_dataloader
    from trainer import Trainer
    from post_processing import get_post_processing
    from utils import get_metric

    # Set up distributed training if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=torch.cuda.device_count(), rank=args.local_rank)
        config['distributed'] = True
    else:
        config['distributed'] = False
    config['local_rank'] = args.local_rank

    # Build dataloaders
    train_loader = get_dataloader(config['dataset']['train'], config['distributed'])
    assert train_loader is not None, "Training dataloader could not be built."
    validate_loader = get_dataloader(config['dataset'].get('validate'), False) if 'validate' in config['dataset'] else None

    # Build loss function
    criterion = build_loss(config['loss']).cuda()

    # Adjust input channels based on image mode
    config['arch']['backbone']['in_channels'] = 3 if config['dataset']['train']['dataset']['args']['img_mode'] != 'GRAY' else 1

    # Build model
    model = build_model(config['arch'])

    # Build post-processing and metric
    post_p = get_post_processing(config['post_processing'])
    metric = get_metric(config['metric'])

    # Initialize and start trainer
    trainer = Trainer(config=config,
                      model=model,
                      criterion=criterion,
                      train_loader=train_loader,
                      post_process=post_p,
                      metric_cls=metric,
                      validate_loader=validate_loader)
    trainer.train()

if __name__ == '__main__':
    args = init_args()
    assert os.path.exists(args.config_file), "Config file does not exist."
    config = load(open(args.config_file, 'rb'))
    if 'base' in config:
        config = parse_config(config)
    main(config)