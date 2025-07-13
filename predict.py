import os
import time
import cv2
import torch
from data_loader import get_transforms
from models import build_model
from post_processing import get_post_processing
import argparse

def resize_image(img, short_size):
    """Resize image while maintaining aspect ratio."""
    height, width, _ = img.shape
    if height < width:
        new_height = short_size
        new_width = int(new_height / height * width)
    else:
        new_width = short_size
        new_height = int(new_width / width * height)
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    return cv2.resize(img, (new_width, new_height))

class Predictor:
    """Wrapper for making predictions with the DBNet model."""
    def __init__(self, model_path, post_p_thre=0.7, gpu_id=None):
        # Set device
        self.device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load checkpoint and config
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']
        config['arch']['backbone']['pretrained'] = False

        # Build model
        self.model = build_model(config['arch']).to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        # Set up post-processing
        self.post_process = get_post_processing(config['post_processing'])
        self.post_process.box_thresh = post_p_thre

        # Image mode and transforms
        self.img_mode = config['dataset']['train']['dataset']['args']['img_mode']
        transforms = [t for t in config['dataset']['train']['dataset']['args']['transforms'] if t['type'] in ['ToTensor', 'Normalize']]
        self.transform = get_transforms(transforms)

    def predict(self, img_path, is_output_polygon=False, short_size=1024):
        """Predict on a single image."""
        assert os.path.exists(img_path), 'Image file does not exist.'
        img = cv2.imread(img_path, 1 if self.img_mode != 'GRAY' else 0)
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img = resize_image(img, short_size)

        # Apply transforms and prepare batch
        tensor = self.transform(img).unsqueeze_(0).to(self.device)
        batch = {'shape': [(h, w)]}

        with torch.no_grad():
            if 'cuda' in str(self.device):
                torch.cuda.synchronize(self.device)
            start = time.time()
            preds = self.model(tensor)
            if 'cuda' in str(self.device):
                torch.cuda.synchronize(self.device)
            box_list, score_list = self.post_process(batch, preds, is_output_polygon=is_output_polygon)[0]
            t = time.time() - start

        # Filter out invalid boxes
        if len(box_list) > 0:
            idx = [x.sum() > 0 for x in box_list] if is_output_polygon else box_list.reshape(-1, 8).sum(axis=1) > 0
            box_list, score_list = [box_list[i] for i in range(len(idx)) if idx[i]], [score_list[i] for i in range(len(idx)) if idx[i]]
        else:
            box_list, score_list = [], []

        return preds[0, 0, :, :].cpu().numpy(), box_list, score_list, t

def init_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='DBNet.pytorch')
    parser.add_argument('--model_path', default='model_best.pth', type=str)
    parser.add_argument('--input_folder', default='./test/input', type=str, help='Folder with input images')
    parser.add_argument('--output_folder', default='./test/output', type=str, help='Folder for output results')
    parser.add_argument('--thre', default=0.3, type=float, help='Threshold for post-processing')
    parser.add_argument('--polygon', action='store_true', help='Output polygons instead of boxes')
    parser.add_argument('--show', action='store_true', help='Show results')
    parser.add_argument('--save_result', action='store_true', help='Save boxes and scores to txt file')
    return parser.parse_args()

if __name__ == '__main__':
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from utils.util import show_img, draw_bbox, save_result, get_file_list

    args = init_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Initialize predictor
    predictor = Predictor(args.model_path, post_p_thre=args.thre, gpu_id=0)

    # Process each image in the input folder
    for img_path in tqdm(get_file_list(args.input_folder, p_postfix=['.jpg'])):
        preds, boxes_list, score_list, t = predictor.predict(img_path, is_output_polygon=args.polygon)
        img = draw_bbox(cv2.imread(img_path)[:, :, ::-1], boxes_list)

        if args.show:
            show_img(preds)
            show_img(img, title=os.path.basename(img_path))
            plt.show()

        # Save results
        os.makedirs(args.output_folder, exist_ok=True)
        img_path = pathlib.Path(img_path)
        output_path = os.path.join(args.output_folder, img_path.stem + '_result.jpg')
        pred_path = os.path.join(args.output_folder, img_path.stem + '_pred.jpg')
        cv2.imwrite(output_path, img[:, :, ::-1])
        cv2.imwrite(pred_path, preds * 255)
        if args.save_result:
            save_result(output_path.replace('_result.jpg', '.txt'), boxes_list, score_list, args.polygon)