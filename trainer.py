import os
import pathlib
import shutil
import time
import torch
from tqdm import tqdm
from anyconfig import dump
from utils import runningScore, cal_text_score, WarmupPolyLR
from pprint import pformat

class Trainer:
    def __init__(self, config, model, criterion, train_loader, validate_loader=None, metric_cls=None, post_process=None):
        """Initialize the Trainer with configuration, model, and data loaders."""
        # Set up directories
        config['trainer']['output_dir'] = os.path.join(str(pathlib.Path(os.path.abspath(__name__)).parent), config['trainer']['output_dir'])
        config['name'] = config['name'] + '_' + model.name
        self.save_dir = os.path.join(config['trainer']['output_dir'], config['name'])
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        
        if config['trainer']['resume_checkpoint'] == '' and config['trainer']['finetune_checkpoint'] == '':
            shutil.rmtree(self.save_dir, ignore_errors=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Core attributes
        self.global_step = 0
        self.start_epoch = 0
        self.config = config
        self.model = model
        self.criterion = criterion
        self.epochs = config['trainer']['epochs']
        self.log_iter = config['trainer']['log_iter']

        # Print configuration if on main process
        if config['local_rank'] == 0:
            dump(config, os.path.join(self.save_dir, 'config.yaml'))
            print(f"Configuration:\n{pformat(config)}")

        # Device setup
        torch.manual_seed(config['trainer']['seed'])
        self.with_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
        self.device = torch.device("cuda" if self.with_cuda else "cpu")
        if self.with_cuda:
            torch.backends.cudnn.benchmark = True
            torch.cuda.manual_seed(config['trainer']['seed'])
            torch.cuda.manual_seed_all(config['trainer']['seed'])
        if config['local_rank'] == 0:
            print(f"Training with device {self.device} and PyTorch {torch.__version__}")

        # Training components
        self.metrics = {'recall': 0, 'precision': 0, 'hmean': 0, 'train_loss': float('inf'), 'best_model_epoch': 0}
        self.optimizer = self._initialize('optimizer', torch.optim, model.parameters())
        self.model.to(self.device)

        # Checkpoint handling
        if config['trainer']['resume_checkpoint']:
            self._load_checkpoint(config['trainer']['resume_checkpoint'], resume=True)
        elif config['trainer']['finetune_checkpoint']:
            self._load_checkpoint(config['trainer']['finetune_checkpoint'], resume=False)

        # Scheduler setup
        self.train_loader = train_loader
        self.train_loader_len = len(train_loader)
        if config['lr_scheduler']['type'] == 'WarmupPolyLR':
            warmup_iters = config['lr_scheduler']['args']['warmup_epoch'] * self.train_loader_len
            if self.start_epoch > 1:
                config['lr_scheduler']['args']['last_epoch'] = (self.start_epoch - 1) * self.train_loader_len
            self.scheduler = WarmupPolyLR(self.optimizer, max_iters=self.epochs * self.train_loader_len, warmup_iters=warmup_iters, **config['lr_scheduler']['args'])
        else:
            self.scheduler = self._initialize('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)

        # Distributed training
        if self.with_cuda and torch.cuda.device_count() > 1:
            local_rank = config['local_rank']
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)

        # Validation components
        self.validate_loader = validate_loader
        self.post_process = post_process
        self.metric_cls = metric_cls
        if config['local_rank'] == 0:
            print(f"Train dataset: {len(train_loader.dataset)} samples, {self.train_loader_len} in dataloader" + 
                  (f", Validate dataset: {len(validate_loader.dataset)} samples, {len(validate_loader)} in dataloader" if validate_loader else ""))

    def train(self):
        """Execute the full training process over all epochs."""
        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            if self.config['distributed']:
                self.train_loader.sampler.set_epoch(epoch)
            self.epoch_result = self._train_epoch(epoch)
            if self.config['lr_scheduler']['type'] != 'WarmupPolyLR':
                self.scheduler.step()
            self._on_epoch_finish()
        self._on_train_finish()

    def _train_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        epoch_start = time.time()
        train_loss = 0.0
        running_metric_text = runningScore(2)

        for i, batch in enumerate(self.train_loader):
            if i >= self.train_loader_len:
                break
            self.global_step += 1

            # Move data to device
            for key, value in batch.items():
                if value is not None and isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)

            # Forward and backward pass
            preds = self.model(batch['img'])
            loss_dict = self.criterion(preds, batch)
            self.optimizer.zero_grad()
            loss_dict['loss'].backward()
            self.optimizer.step()
            if self.config['lr_scheduler']['type'] == 'WarmupPolyLR':
                self.scheduler.step()

            # Compute metrics
            score_shrink_map = cal_text_score(preds[:, 0, :, :], batch['shrink_map'], batch['shrink_mask'], running_metric_text, thred=self.config['post_processing']['args']['thresh'])
            train_loss += loss_dict['loss'].item()
            acc = score_shrink_map['Mean Acc']
            iou_shrink_map = score_shrink_map['Mean IoU']

            # Log progress
            if self.global_step % self.log_iter == 0 and self.config['local_rank'] == 0:
                loss_str = f"loss: {loss_dict['loss'].item():.4f}"
                for key, value in loss_dict.items():
                    if key != 'loss':
                        loss_str += f", {key}: {value.item():.4f}"
                print(f"[{epoch}/{self.epochs}], [{i+1}/{self.train_loader_len}], step: {self.global_step}, acc: {acc:.4f}, iou: {iou_shrink_map:.4f}, {loss_str}")

        epoch_time = time.time() - epoch_start
        lr = self.optimizer.param_groups[0]['lr']
        return {'train_loss': train_loss / self.train_loader_len, 'lr': lr, 'time': epoch_time, 'epoch': epoch}

    def _eval(self, epoch):
        """Evaluate the model on the validation set."""
        self.model.eval()
        raw_metrics = []
        total_frame = 0.0
        total_time = 0.0

        for batch in tqdm(self.validate_loader, total=len(self.validate_loader), desc='Evaluating'):
            with torch.no_grad():
                for key, value in batch.items():
                    if value is not None and isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                start = time.time()
                preds = self.model(batch['img'])
                boxes, scores = self.post_process(batch, preds, is_output_polygon=self.metric_cls.is_output_polygon)
                total_frame += batch['img'].size()[0]
                total_time += time.time() - start
                raw_metrics.append(self.metric_cls.validate_measure(batch, (boxes, scores)))

        metrics = self.metric_cls.gather_measure(raw_metrics)
        if self.config['local_rank'] == 0:
            print(f"FPS: {total_frame / total_time:.2f}")
        return metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg

    def _on_epoch_finish(self):
        """Handle tasks at the end of each epoch."""
        if self.config['local_rank'] == 0:
            print(f"[{self.epoch_result['epoch']}/{self.epochs}], train_loss: {self.epoch_result['train_loss']:.4f}, time: {self.epoch_result['time']:.2f}, lr: {self.epoch_result['lr']:.6f}")
        net_save_path = os.path.join(self.checkpoint_dir, 'model_latest.pth')
        net_save_path_best = os.path.join(self.checkpoint_dir, 'model_best.pth')

        if self.config['local_rank'] == 0:
            self._save_checkpoint(self.epoch_result['epoch'], net_save_path)
            save_best = False

            if self.validate_loader and self.metric_cls:
                recall, precision, hmean = self._eval(self.epoch_result['epoch'])
                print(f"Test: recall: {recall:.6f}, precision: {precision:.6f}, hmean: {hmean:.6f}")
                if hmean >= self.metrics['hmean']:
                    save_best = True
                    self.metrics.update({'train_loss': self.epoch_result['train_loss'], 'hmean': hmean, 'precision': precision, 'recall': recall, 'best_model_epoch': self.epoch_result['epoch']})
            else:
                if self.epoch_result['train_loss'] <= self.metrics['train_loss']:
                    save_best = True
                    self.metrics.update({'train_loss': self.epoch_result['train_loss'], 'best_model_epoch': self.epoch_result['epoch']})

            best_str = "Current best: " + ", ".join(f"{k}: {v:.6f}" for k, v in self.metrics.items())
            print(best_str)
            if save_best:
                shutil.copy(net_save_path, net_save_path_best)
                print(f"Saving current best: {net_save_path_best}")
            else:
                print(f"Saving checkpoint: {net_save_path}")

    def _on_train_finish(self):
        """Finalize the training process."""
        if self.config['local_rank'] == 0:
            print("Training finished:")
            for k, v in self.metrics.items():
                print(f"{k}: {v}")
            print("Done.")

    def _save_checkpoint(self, epoch, file_name):
        """Save the model checkpoint."""
        state_dict = self.model.module.state_dict() if self.config['distributed'] else self.model.state_dict()
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': self.metrics
        }
        torch.save(state, os.path.join(self.checkpoint_dir, file_name))

    def _load_checkpoint(self, checkpoint_path, resume):
        """Load a model checkpoint."""
        if self.config['local_rank'] == 0:
            print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'], strict=resume)
        if resume:
            self.global_step = checkpoint['global_step']
            self.start_epoch = checkpoint['epoch']
            self.config['lr_scheduler']['args']['last_epoch'] = self.start_epoch
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'metrics' in checkpoint:
                self.metrics = checkpoint['metrics']
            if self.with_cuda:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            if self.config['local_rank'] == 0:
                print(f"Resumed from checkpoint {checkpoint_path} (epoch {self.start_epoch})")
        else:
            if self.config['local_rank'] == 0:
                print(f"Finetuning from checkpoint {checkpoint_path}")

    def _initialize(self, name, module, *args, **kwargs):
        """Initialize a module (optimizer or scheduler) from config."""
        module_name = self.config[name]['type']
        module_args = self.config[name]['args']
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)