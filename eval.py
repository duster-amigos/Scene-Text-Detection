import argparse
import time
import torch
from tqdm import tqdm

class Evaluator:
    """Class to evaluate the DBNet model on a validation dataset."""
    def __init__(self, model_path, gpu_id=0):
        from models import build_model
        from data_loader import get_dataloader
        from post_processing import get_post_processing
        from utils import get_metric

        # Set device
        self.device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None and torch.cuda.is_available() else "cpu")
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        # Load checkpoint and config
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint['config']
        config['arch']['backbone']['pretrained'] = False

        # Build validation dataloader
        self.validate_loader = get_dataloader(config['dataset']['validate'], distributed=False)

        # Build model and load weights
        self.model = build_model(config['arch']).to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        # Build post-processing and metric
        self.post_process = get_post_processing(config['post_processing'])
        self.metric_cls = get_metric(config['metric'])

    def eval(self):
        """Evaluate the model and compute metrics."""
        raw_metrics = []
        total_frame = 0.0
        total_time = 0.0

        for batch in tqdm(self.validate_loader, total=len(self.validate_loader), desc='Evaluating model'):
            with torch.no_grad():
                # Move batch to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)

                start = time.time()
                preds = self.model(batch['img'])
                boxes, scores = self.post_process(batch, preds, is_output_polygon=self.metric_cls.is_output_polygon)
                total_frame += batch['img'].size(0)
                total_time += time.time() - start

                # Compute metrics
                raw_metric = self.metric_cls.validate_measure(batch, (boxes, scores))
                raw_metrics.append(raw_metric)

        # Gather and print metrics
        metrics = self.metric_cls.gather_measure(raw_metrics)
        print(f"FPS: {total_frame / total_time:.2f}")
        return metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg

def init_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='DBNet.pytorch')
    parser.add_argument('--model_path', default='output/DBNet_resnet18_FPN_DBHead/checkpoint/1.pth', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = init_args()
    evaluator = Evaluator(args.model_path)
    result = evaluator.eval()
    print(result)