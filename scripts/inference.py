import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import argparse
import time
from src.models.model import Model
import json
import torchvision.transforms as transforms

class TextDetector:
    def __init__(self, model_path, config_path, device=None):
        """
        Initialize text detector
        
        Args:
            model_path (str): Path to trained model checkpoint
            config_path (str): Path to model configuration file
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            # Set device
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
            print(f"Using device: {self.device}")
            
            # Load model
            self.model = Model(self.config['model']).to(self.device)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Model loaded from {model_path}")
            
            # Set inference parameters
            self.min_area = self.config.get('inference', {}).get('min_area', 100)
            self.thresh = self.config.get('inference', {}).get('thresh', 0.3)
            self.box_thresh = self.config.get('inference', {}).get('box_thresh', 0.5)
            self.max_candidates = self.config.get('inference', {}).get('max_candidates', 1000)
            
        except Exception as e:
            print(f"Error initializing text detector: {e}")
            raise
    
    def preprocess_image(self, image):
        """
        Preprocess image for inference
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Apply transforms
            transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            return image_tensor
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            raise
    
    def postprocess(self, pred, original_shape):
        """
        Postprocess model predictions to get text boxes
        
        Args:
            pred (torch.Tensor): Model predictions
            original_shape (tuple): Original image shape (H, W)
            
        Returns:
            list: List of detected text boxes
        """
        try:
            # Extract predictions
            shrink_maps = pred[:, 0, :, :].cpu().numpy()
            threshold_maps = pred[:, 1, :, :].cpu().numpy()
            
            # Resize to original image size
            h, w = original_shape[:2]
            shrink_maps = cv2.resize(shrink_maps[0], (w, h))
            threshold_maps = cv2.resize(threshold_maps[0], (w, h))
            
            # Binarize shrink map
            binary_maps = (shrink_maps > self.thresh).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(binary_maps, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and process contours
            boxes = []
            for contour in contours:
                # Calculate area
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence using threshold map
                mask = np.zeros_like(binary_maps)
                cv2.fillPoly(mask, [contour], 1)
                confidence = np.mean(threshold_maps[mask == 1])
                
                if confidence > self.box_thresh:
                    boxes.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': float(confidence),
                        'area': float(area)
                    })
            
            # Sort by confidence and limit candidates
            boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
            boxes = boxes[:self.max_candidates]
            
            return boxes
            
        except Exception as e:
            print(f"Error in postprocessing: {e}")
            return []
    
    def detect_text(self, image):
        """
        Detect text in image
        
        Args:
            image: Input image (numpy array, PIL Image, or file path)
            
        Returns:
            list: List of detected text boxes with confidence scores
        """
        try:
            # Load image if path is provided
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image file not found: {image}")
                image = cv2.imread(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            original_shape = image.shape
            
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            # Postprocess predictions
            boxes = self.postprocess(predictions, original_shape)
            
            return boxes
            
        except Exception as e:
            print(f"Error in text detection: {e}")
            return []
    
    def visualize_results(self, image, boxes, output_path=None):
        """
        Visualize detection results
        
        Args:
            image: Input image
            boxes (list): Detected text boxes
            output_path (str): Path to save visualization
            
        Returns:
            numpy.ndarray: Image with visualized results
        """
        try:
            # Convert to numpy array if needed
            if isinstance(image, str):
                image = cv2.imread(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image, Image.Image):
                image = np.array(image)
            
            # Draw boxes
            result_image = image.copy()
            for box in boxes:
                x1, y1, x2, y2 = box['bbox']
                confidence = box['confidence']
                
                # Draw rectangle
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw confidence score
                text = f"{confidence:.2f}"
                cv2.putText(result_image, text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Save if output path provided
            if output_path:
                result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, result_image_bgr)
                print(f"Visualization saved to {output_path}")
            
            return result_image
            
        except Exception as e:
            print(f"Error in visualization: {e}")
            return image

def main():
    parser = argparse.ArgumentParser(description='Text detection inference')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default=None, help='Path to save output visualization')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    try:
        # Initialize detector
        detector = TextDetector(args.model, args.config, args.device)
        
        # Load image
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image file not found: {args.image}")
        
        # Detect text
        print(f"Processing image: {args.image}")
        start_time = time.time()
        boxes = detector.detect_text(args.image)
        inference_time = time.time() - start_time
        
        print(f"Detection completed in {inference_time:.3f} seconds")
        print(f"Found {len(boxes)} text regions")
        
        # Print results
        for i, box in enumerate(boxes):
            print(f"Box {i+1}: {box['bbox']}, Confidence: {box['confidence']:.3f}")
        
        # Visualize results
        if args.output:
            detector.visualize_results(args.image, boxes, args.output)
        
    except Exception as e:
        print(f"Error in main: {e}")
        raise

if __name__ == '__main__':
    main() 