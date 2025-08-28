#!/usr/bin/env python3
"""
Convert SAM2 annotations (COCO JSON) to YOLO segmentation format
Filters out empty annotation files and creates proper YOLO dataset structure
"""

import json
import cv2
import numpy as np
from pathlib import Path
import shutil
import os

def convert_sam2_to_yolo():
    print("🔄 CONVERTING SAM2 TO YOLO FORMAT")
    print("=" * 60)
    
    # Source and destination paths
    sam2_dir = Path("data/sam2_annotations")
    output_dir = Path("data/sam2_yolo_dataset")
    
    # Create output directory structure
    output_dir.mkdir(exist_ok=True)
    (output_dir / "train").mkdir(exist_ok=True)
    (output_dir / "valid").mkdir(exist_ok=True)
    (output_dir / "test").mkdir(exist_ok=True)
    
    for split in ["train", "valid", "test"]:
        (output_dir / split / "images").mkdir(exist_ok=True)
        (output_dir / split / "labels").mkdir(exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Process each split
    total_converted = 0
    total_annotations = 0
    
    for split in ["train", "valid", "test"]:
        print(f"\n🔍 Processing {split} split...")
        
        split_dir = sam2_dir / split
        if not split_dir.exists():
            print(f"  ⚠️  {split} directory not found, skipping")
            continue
        
        # Get JSON files
        json_files = list(split_dir.glob("*.json"))
        print(f"  📸 Found {len(json_files)} JSON files")
        
        # Filter files with annotations
        files_with_annotations = []
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                annotations = data.get('annotations', [])
                if len(annotations) > 0:
                    files_with_annotations.append((json_file, data))
            except Exception as e:
                print(f"  ❌ Error reading {json_file.name}: {e}")
        
        print(f"  ✅ {len(files_with_annotations)} files have annotations")
        
        # Convert each file
        split_converted = 0
        split_annotations = 0
        
        for json_file, data in files_with_annotations:
            try:
                # Get image info
                image_info = data.get('image', {})
                image_name = image_info.get('file_name', 'unknown')
                image_height = image_info.get('height', 0)
                image_width = image_info.get('width', 0)
                
                # Check if image exists
                image_file = split_dir / image_name
                if not image_file.exists():
                    print(f"    ⚠️  Image not found: {image_name}")
                    continue
                
                # Copy image
                dest_image = output_dir / split / "images" / image_name
                shutil.copy2(image_file, dest_image)
                
                # Convert annotations to YOLO format
                yolo_annotations = convert_annotations_to_yolo(data, image_width, image_height)
                
                if yolo_annotations:
                    # Save YOLO labels
                    label_name = image_name.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.JPG', '.txt')
                    label_file = output_dir / split / "labels" / label_name
                    
                    with open(label_file, 'w') as f:
                        for annotation in yolo_annotations:
                            f.write(annotation + '\n')
                    
                    split_converted += 1
                    split_annotations += len(yolo_annotations)
                    print(f"    ✅ {image_name}: {len(yolo_annotations)} annotations")
                else:
                    print(f"    ⚠️  {image_name}: No valid annotations")
                
            except Exception as e:
                print(f"    ❌ Error processing {json_file.name}: {e}")
        
        print(f"  📊 {split} split: {split_converted} files, {split_annotations} annotations")
        total_converted += split_converted
        total_annotations += split_annotations
    
    # Create data.yaml
    create_data_yaml(output_dir, total_converted, total_annotations)
    
    print(f"\n🎯 CONVERSION COMPLETE!")
    print("=" * 60)
    print(f"📊 Total files converted: {total_converted}")
    print(f"📊 Total annotations: {total_annotations}")
    print(f"📁 Output directory: {output_dir}")
    print(f"\n💡 Next steps:")
    print(f"  1. Verify the converted dataset")
    print(f"  2. Update training configuration")
    print(f"  3. Start training with the new dataset")

def convert_annotations_to_yolo(data, img_width, img_height):
    """Convert COCO annotations to YOLO segmentation format"""
    annotations = data.get('annotations', [])
    yolo_annotations = []
    
    for ann in annotations:
        try:
            # Get bounding box
            bbox = ann.get('bbox', [])  # [x, y, width, height]
            if len(bbox) < 4:
                continue
            
            x, y, w, h = bbox
            
            # Convert to YOLO format (normalized center coordinates and size)
            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            width_norm = w / img_width
            height_norm = h / img_height
            
            # Get segmentation mask
            segmentation = ann.get('segmentation', {})
            if 'counts' not in segmentation:
                continue
            
            # For now, we'll use class 0 (cucumber) for all objects
            # You can modify this if you have multiple classes
            class_id = 0
            
            # Create YOLO annotation line
            # Format: class_id x_center y_center width height
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
            
            yolo_annotations.append(yolo_line)
            
        except Exception as e:
            print(f"      ⚠️  Error converting annotation: {e}")
            continue
    
    return yolo_annotations

def create_data_yaml(output_dir, total_files, total_annotations):
    """Create data.yaml file for YOLO training"""
    yaml_content = f"""# YOLO dataset configuration
# Converted from SAM2 annotations

# Dataset paths (relative to this file)
train: train/images
val: valid/images
test: test/images

# Number of classes
nc: 1

# Class names
names: ['cucumber']

# Dataset statistics
# Total files: {total_files}
# Total annotations: {total_annotations}

# Source: SAM2 annotations from Roboflow
# Conversion date: {Path().cwd().name}
"""
    
    yaml_file = output_dir / "data.yaml"
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"  📝 Created data.yaml: {yaml_file}")

if __name__ == "__main__":
    convert_sam2_to_yolo()
