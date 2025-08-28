#!/usr/bin/env python3
"""
Convert COCO format annotations to YOLO segmentation format.
"""

import json
import os
import cv2
import numpy as np
from pathlib import Path
from pycocotools import mask as coco_mask
import argparse
from tqdm import tqdm

def coco_to_yolo_bbox(bbox, img_width, img_height):
    """Convert COCO bbox to YOLO format."""
    x_min, y_min, width, height = bbox
    
    # Convert to center coordinates and normalize
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    
    return x_center, y_center, norm_width, norm_height

def rle_to_polygon(segmentation, img_height, img_width):
    """Convert RLE segmentation to polygon format for YOLO."""
    if isinstance(segmentation, dict) and 'counts' in segmentation:
        # RLE format
        rle = segmentation
        if isinstance(rle['counts'], str):
            rle['counts'] = rle['counts'].encode('utf-8')
        
        # Decode RLE to binary mask
        binary_mask = coco_mask.decode(rle)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
            
        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Simplify the contour
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to normalized polygon format
        polygon = []
        for point in approx:
            x, y = point[0]
            polygon.extend([x / img_width, y / img_height])
            
        return polygon
    
    elif isinstance(segmentation, list):
        # Polygon format (already in the right format, just need to normalize)
        if len(segmentation) == 0:
            return None
            
        polygon = segmentation[0]  # Take first polygon
        normalized_polygon = []
        
        for i in range(0, len(polygon), 2):
            if i + 1 < len(polygon):
                x = polygon[i] / img_width
                y = polygon[i + 1] / img_height
                normalized_polygon.extend([x, y])
                
        return normalized_polygon
    
    return None

def convert_coco_to_yolo_seg(coco_dir, output_dir, class_names):
    """Convert COCO dataset to YOLO segmentation format."""
    
    # Create output directories
    output_dir = Path(output_dir)
    for split in ['train', 'valid', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        coco_split_dir = Path(coco_dir) / split
        if not coco_split_dir.exists():
            print(f"Skipping {split} - directory doesn't exist")
            continue
            
        print(f"Converting {split} split...")
        
        # Get all JSON files
        json_files = list(coco_split_dir.glob("*.json"))
        
        for json_file in tqdm(json_files, desc=f"Converting {split}"):
            # Load COCO annotation
            with open(json_file, 'r') as f:
                coco_data = json.load(f)
            
            image_info = coco_data['image']
            annotations = coco_data['annotations']
            
            img_filename = image_info['file_name']
            img_width = image_info['width']
            img_height = image_info['height']
            
            # Copy image to output directory
            src_img_path = coco_split_dir / img_filename
            dst_img_path = output_dir / split / 'images' / img_filename
            
            if src_img_path.exists():
                # Copy image file
                import shutil
                shutil.copy2(src_img_path, dst_img_path)
                
                # Create YOLO label file
                label_filename = img_filename.replace('.jpg', '.txt').replace('.png', '.txt')
                label_path = output_dir / split / 'labels' / label_filename
                
                with open(label_path, 'w') as label_file:
                    for ann in annotations:
                        # Map class name to class ID
                        # For now, assume all objects are class 4 (cucumber)
                        class_id = 4  # cucumber class
                        
                        # Convert segmentation to YOLO format
                        segmentation = ann.get('segmentation', {})
                        
                        if segmentation:
                            polygon = rle_to_polygon(segmentation, img_height, img_width)
                            
                            if polygon and len(polygon) >= 6:  # At least 3 points
                                # Write YOLO segmentation format: class_id x1 y1 x2 y2 ... xn yn
                                line = f"{class_id}"
                                for coord in polygon:
                                    line += f" {coord:.6f}"
                                label_file.write(line + "\n")
    
    # Create data.yaml file
    data_yaml_content = f"""train: train/images
val: valid/images
test: test/images

nc: {len(class_names)}
names: {class_names}
"""
    
    with open(output_dir / 'data.yaml', 'w') as f:
        f.write(data_yaml_content)
    
    print(f"Conversion completed! Output saved to {output_dir}")
    print(f"data.yaml created with {len(class_names)} classes")

def main():
    parser = argparse.ArgumentParser(description="Convert COCO to YOLO segmentation format")
    parser.add_argument('--input', required=True, help='Input COCO dataset directory')
    parser.add_argument('--output', required=True, help='Output YOLO dataset directory')
    parser.add_argument('--classes', nargs='+', default=['big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber', 'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice'], help='Class names')
    
    args = parser.parse_args()
    
    convert_coco_to_yolo_seg(args.input, args.output, args.classes)

if __name__ == "__main__":
    main()
