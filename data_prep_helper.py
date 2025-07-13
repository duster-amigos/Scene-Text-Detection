#!/usr/bin/env python3
"""
Data Preparation Helper for DBNet
Converts various annotation formats to ICDAR 2015 format
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse

def convert_coco_to_icdar(coco_json_path, output_dir):
    """
    Convert COCO format annotations to ICDAR 2015 format
    """
    try:
        print(f"Converting COCO annotations from: {coco_json_path}")
        
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Group annotations by image
        image_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)
        
        # Create image id to filename mapping
        img_id_to_filename = {}
        for img in coco_data['images']:
            img_id_to_filename[img['id']] = img['file_name']
        
        # Convert each image's annotations
        for img_id, annotations in image_annotations.items():
            filename = img_id_to_filename[img_id]
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(output_dir, txt_filename)
            
            print(f"Processing {filename} with {len(annotations)} annotations")
            
            with open(txt_path, 'w') as f:
                for ann in annotations:
                    # COCO format: [x1, y1, x2, y2, x3, y3, x4, y4]
                    segmentation = ann['segmentation'][0]  # Assuming polygon format
                    
                    # Convert to ICDAR format: x1,y1,x2,y2,x3,y3,x4,y4,text
                    coords = [str(int(x)) for x in segmentation[:8]]
                    text = ann.get('text', '')  # Get text if available
                    
                    line = ','.join(coords + [text])
                    f.write(line + '\n')
            
            print(f"Created {txt_path}")
        
        print(f"Conversion completed. Output directory: {output_dir}")
        
    except Exception as e:
        print(f"Error converting COCO format: {e}")
        raise

def convert_yolo_to_icdar(yolo_dir, output_dir, img_dir):
    """
    Convert YOLO format annotations to ICDAR 2015 format
    Note: YOLO uses rectangular boxes, so we'll create rectangular polygons
    """
    try:
        print(f"Converting YOLO annotations from: {yolo_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all txt files in yolo directory
        txt_files = [f for f in os.listdir(yolo_dir) if f.endswith('.txt')]
        
        for txt_file in txt_files:
            img_filename = txt_file.replace('.txt', '.jpg')  # Assuming jpg images
            img_path = os.path.join(img_dir, img_filename)
            
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
            
            # Get image dimensions
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image: {img_path}")
                continue
            
            h, w = img.shape[:2]
            
            # Read YOLO annotations
            yolo_path = os.path.join(yolo_dir, txt_file)
            icdar_path = os.path.join(output_dir, txt_file)
            
            print(f"Converting {txt_file} (image size: {w}x{h})")
            
            with open(yolo_path, 'r') as yolo_f, open(icdar_path, 'w') as icdar_f:
                for line in yolo_f:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # class, x_center, y_center, width, height
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * w
                        y_center = float(parts[2]) * h
                        width = float(parts[3]) * w
                        height = float(parts[4]) * h
                        
                        # Convert to corner coordinates
                        x1 = x_center - width/2
                        y1 = y_center - height/2
                        x2 = x_center + width/2
                        y2 = y_center - height/2
                        x3 = x_center + width/2
                        y3 = y_center + height/2
                        x4 = x_center - width/2
                        y4 = y_center + height/2
                        
                        # Write in ICDAR format
                        coords = [str(int(x)) for x in [x1, y1, x2, y2, x3, y3, x4, y4]]
                        line = ','.join(coords + [''])  # Empty text field
                        icdar_f.write(line + '\n')
            
            print(f"Created {icdar_path}")
        
        print(f"Conversion completed. Output directory: {output_dir}")
        
    except Exception as e:
        print(f"Error converting YOLO format: {e}")
        raise

def create_sample_data(output_dir, num_images=10):
    """
    Create sample data for testing
    """
    try:
        print(f"Creating sample data in: {output_dir}")
        
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
        
        for i in range(num_images):
            # Create a simple image
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img_path = os.path.join(output_dir, 'images', f'sample_{i:03d}.jpg')
            cv2.imwrite(img_path, img)
            
            # Create annotation file with multiple text regions
            ann_path = os.path.join(output_dir, 'annotations', f'sample_{i:03d}.txt')
            
            with open(ann_path, 'w') as f:
                # Add 2-4 random text regions per image
                num_regions = np.random.randint(2, 5)
                for j in range(num_regions):
                    # Random coordinates for text region
                    x1 = np.random.randint(50, 500)
                    y1 = np.random.randint(50, 500)
                    width = np.random.randint(50, 150)
                    height = np.random.randint(20, 50)
                    
                    x2 = x1 + width
                    y2 = y1
                    x3 = x1 + width
                    y3 = y1 + height
                    x4 = x1
                    y4 = y1 + height
                    
                    coords = [str(x) for x in [x1, y1, x2, y2, x3, y3, x4, y4]]
                    line = ','.join(coords + ['text'])
                    f.write(line + '\n')
            
            print(f"Created sample_{i:03d}.jpg with {num_regions} text regions")
        
        print(f"Sample data created successfully in {output_dir}")
        
    except Exception as e:
        print(f"Error creating sample data: {e}")
        raise

def validate_icdar_format(data_dir):
    """
    Validate ICDAR 2015 format data
    """
    try:
        print(f"Validating ICDAR format data in: {data_dir}")
        
        images_dir = os.path.join(data_dir, 'images')
        annotations_dir = os.path.join(data_dir, 'annotations')
        
        if not os.path.exists(images_dir):
            print(f"Error: Images directory not found: {images_dir}")
            return False
        
        if not os.path.exists(annotations_dir):
            print(f"Error: Annotations directory not found: {annotations_dir}")
            return False
        
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
        
        print(f"Found {len(image_files)} images and {len(annotation_files)} annotation files")
        
        # Check if each image has corresponding annotation
        missing_annotations = []
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            ann_file = base_name + '.txt'
            if ann_file not in annotation_files:
                missing_annotations.append(img_file)
        
        if missing_annotations:
            print(f"Warning: {len(missing_annotations)} images missing annotations:")
            for img in missing_annotations[:5]:  # Show first 5
                print(f"  - {img}")
            if len(missing_annotations) > 5:
                print(f"  ... and {len(missing_annotations) - 5} more")
        else:
            print("✓ All images have corresponding annotation files")
        
        # Validate annotation format
        invalid_annotations = []
        for ann_file in annotation_files[:10]:  # Check first 10 files
            ann_path = os.path.join(annotations_dir, ann_file)
            try:
                with open(ann_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            parts = line.split(',')
                            if len(parts) < 8:
                                invalid_annotations.append(f"{ann_file}:line{line_num}")
                                break
                            # Check if coordinates are numeric
                            try:
                                coords = [float(x) for x in parts[:8]]
                                if any(c < 0 for c in coords):
                                    invalid_annotations.append(f"{ann_file}:line{line_num}")
                                    break
                            except ValueError:
                                invalid_annotations.append(f"{ann_file}:line{line_num}")
                                break
            except Exception as e:
                print(f"Error reading {ann_file}: {e}")
        
        if invalid_annotations:
            print(f"Warning: {len(invalid_annotations)} invalid annotation lines found:")
            for ann in invalid_annotations[:5]:
                print(f"  - {ann}")
        else:
            print("✓ All checked annotation files have valid format")
        
        return len(missing_annotations) == 0 and len(invalid_annotations) == 0
        
    except Exception as e:
        print(f"Error validating data: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Data preparation helper for DBNet')
    parser.add_argument('--action', choices=['convert_coco', 'convert_yolo', 'create_sample', 'validate'], 
                       required=True, help='Action to perform')
    parser.add_argument('--input', help='Input file or directory')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--img_dir', help='Image directory (for YOLO conversion)')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of sample images to create')
    
    args = parser.parse_args()
    
    try:
        if args.action == 'convert_coco':
            if not args.input or not args.output:
                print("Error: --input and --output required for convert_coco")
                return
            convert_coco_to_icdar(args.input, args.output)
            
        elif args.action == 'convert_yolo':
            if not args.input or not args.output or not args.img_dir:
                print("Error: --input, --output, and --img_dir required for convert_yolo")
                return
            convert_yolo_to_icdar(args.input, args.output, args.img_dir)
            
        elif args.action == 'create_sample':
            if not args.output:
                print("Error: --output required for create_sample")
                return
            create_sample_data(args.output, args.num_samples)
            
        elif args.action == 'validate':
            if not args.input:
                print("Error: --input required for validate")
                return
            validate_icdar_format(args.input)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 