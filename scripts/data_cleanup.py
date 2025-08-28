#!/usr/bin/env python3
"""
Data Cleanup Script
Fixes annotation issues and prepares clean data for production training
"""

import os
import cv2
import numpy as np
import shutil
from pathlib import Path
import argparse
import json
from collections import defaultdict, Counter

class DataCleanup:
    def __init__(self, data_dir):
        """Initialize data cleanup."""
        self.data_dir = Path(data_dir)
        self.class_names = [
            'big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber',
            'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice'
        ]
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'images_with_annotations': 0,
            'images_without_annotations': 0,
            'total_annotations': 0,
            'class_distribution': Counter(),
            'over_segmented_cucumbers': 0,
            'fixed_annotations': 0
        }
    
    def analyze_data(self):
        """Analyze current data quality."""
        print("🔍 Analyzing data quality...")
        print("=" * 60)
        
        # Check each split
        for split in ['train', 'valid', 'test']:
            images_dir = self.data_dir / split / 'images'
            labels_dir = self.data_dir / split / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                print(f"⚠️ {split} directory not found")
                continue
            
            print(f"\n📁 {split.upper()} Split:")
            print("-" * 40)
            
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))
            label_files = list(labels_dir.glob('*.txt'))
            
            print(f"  Images: {len(image_files)}")
            print(f"  Labels: {len(label_files)}")
            
            # Check for missing annotations
            missing_annotations = 0
            empty_annotations = 0
            
            for img_file in image_files:
                self.stats['total_images'] += 1
                label_file = labels_dir / f"{img_file.stem}.txt"
                
                if not label_file.exists():
                    missing_annotations += 1
                    self.stats['images_without_annotations'] += 1
                else:
                    # Check if annotation file is empty
                    if label_file.stat().st_size == 0:
                        empty_annotations += 1
                        self.stats['images_without_annotations'] += 1
                    else:
                        self.stats['images_with_annotations'] += 1
                        
                        # Count annotations and classes
                        with open(label_file, 'r') as f:
                            for line in f:
                                if line.strip():
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        class_id = int(parts[0])
                                        if 0 <= class_id < len(self.class_names):
                                            self.stats['class_distribution'][self.class_names[class_id]] += 1
                                            self.stats['total_annotations'] += 1
            
            print(f"  Missing annotations: {missing_annotations}")
            print(f"  Empty annotations: {empty_annotations}")
        
        # Print overall statistics
        print(f"\n📊 OVERALL STATISTICS:")
        print("=" * 60)
        print(f"Total images: {self.stats['total_images']}")
        print(f"Images with annotations: {self.stats['images_with_annotations']}")
        print(f"Images without annotations: {self.stats['images_without_annotations']}")
        print(f"Total annotations: {self.stats['total_annotations']}")
        
        print(f"\nClass distribution:")
        for class_name, count in self.stats['class_distribution'].most_common():
            print(f"  {class_name}: {count}")
    
    def fix_over_segmented_cucumbers(self, iou_threshold=0.3):
        """Fix over-segmented cucumber annotations by merging overlapping ones."""
        print(f"\n🔧 Fixing over-segmented cucumbers (IoU threshold: {iou_threshold})...")
        print("=" * 60)
        
        for split in ['train', 'valid', 'test']:
            labels_dir = self.data_dir / split / 'labels'
            if not labels_dir.exists():
                continue
            
            print(f"\nProcessing {split} split...")
            
            for label_file in labels_dir.glob('*.txt'):
                if label_file.stat().st_size == 0:
                    continue
                
                # Read annotations
                annotations = []
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                
                                annotations.append({
                                    'class_id': class_id,
                                    'class_name': self.class_names[class_id] if 0 <= class_id < len(self.class_names) else 'unknown',
                                    'x_center': x_center,
                                    'y_center': y_center,
                                    'width': width,
                                    'height': height,
                                    'bbox': [x_center - width/2, y_center - height/2, x_center + width/2, y_center + height/2]
                                })
                
                # Find cucumber annotations
                cucumber_anns = [ann for ann in annotations if ann['class_name'] == 'cucumber']
                other_anns = [ann for ann in annotations if ann['class_name'] != 'cucumber']
                
                if len(cucumber_anns) <= 1:
                    continue
                
                # Check for overlapping cucumbers
                merged_cucumbers = self._merge_overlapping_annotations(cucumber_anns, iou_threshold)
                
                if len(merged_cucumbers) < len(cucumber_anns):
                    self.stats['over_segmented_cucumbers'] += len(cucumber_anns) - len(merged_cucumbers)
                    self.stats['fixed_annotations'] += len(cucumber_anns) - len(merged_cucumbers)
                    
                    # Write fixed annotations
                    with open(label_file, 'w') as f:
                        # Write merged cucumbers
                        for ann in merged_cucumbers:
                            x_center = (ann['bbox'][0] + ann['bbox'][2]) / 2
                            y_center = (ann['bbox'][1] + ann['bbox'][3]) / 2
                            width = ann['bbox'][2] - ann['bbox'][0]
                            height = ann['bbox'][3] - ann['bbox'][1]
                            
                            f.write(f"{ann['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        
                        # Write other annotations
                        for ann in other_anns:
                            f.write(f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n")
                    
                    print(f"  Fixed {label_file.name}: {len(cucumber_anns)} → {len(merged_cucumbers)} cucumbers")
    
    def _merge_overlapping_annotations(self, annotations, iou_threshold):
        """Merge overlapping annotations."""
        if len(annotations) <= 1:
            return annotations
        
        # Sort by area (largest first)
        annotations.sort(key=lambda x: x['width'] * x['height'], reverse=True)
        
        merged = []
        used = set()
        
        for i, ann1 in enumerate(annotations):
            if i in used:
                continue
            
            current_group = [ann1]
            used.add(i)
            
            # Find overlapping annotations
            for j, ann2 in enumerate(annotations[i+1:], i+1):
                if j in used:
                    continue
                
                iou = self._calculate_iou(ann1['bbox'], ann2['bbox'])
                if iou > iou_threshold:
                    current_group.append(ann2)
                    used.add(j)
            
            # Merge the group
            if len(current_group) > 1:
                merged_ann = self._merge_annotation_group(current_group)
                merged.append(merged_ann)
            else:
                merged.append(ann1)
        
        return merged
    
    def _merge_annotation_group(self, group):
        """Merge a group of overlapping annotations."""
        # Use the largest annotation as base
        base = max(group, key=lambda x: x['width'] * x['height'])
        
        # Calculate merged bounding box
        x1 = min(ann['bbox'][0] for ann in group)
        y1 = min(ann['bbox'][1] for ann in group)
        x2 = max(ann['bbox'][2] for ann in group)
        y2 = max(ann['bbox'][3] for ann in group)
        
        merged = base.copy()
        merged['bbox'] = [x1, y1, x2, y2]
        merged['x_center'] = (x1 + x2) / 2
        merged['y_center'] = (y1 + y2) / 2
        merged['width'] = x2 - x1
        merged['height'] = y2 - y1
        
        return merged
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def create_clean_dataset(self, output_dir):
        """Create a clean dataset with only properly annotated images."""
        print(f"\n🧹 Creating clean dataset...")
        print("=" * 60)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create splits
        for split in ['train', 'valid', 'test']:
            (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Copy only good images
        good_images = 0
        for split in ['train', 'valid', 'test']:
            images_dir = self.data_dir / split / 'images'
            labels_dir = self.data_dir / split / 'labels'
            
            if not images_dir.exists():
                continue
            
            for img_file in images_dir.glob('*.jpg'):
                label_file = labels_dir / f"{img_file.stem}.txt"
                
                # Only copy images with valid annotations
                if label_file.exists() and label_file.stat().st_size > 0:
                    # Copy image
                    shutil.copy2(img_file, output_dir / split / 'images' / img_file.name)
                    # Copy label
                    shutil.copy2(label_file, output_dir / split / 'labels' / label_file.name)
                    good_images += 1
        
        print(f"✅ Clean dataset created: {good_images} images")
        print(f"📁 Output directory: {output_dir}")
        
        # Create data.yaml for the clean dataset
        self._create_data_yaml(output_dir)
    
    def _create_data_yaml(self, output_dir):
        """Create data.yaml file for the clean dataset."""
        yaml_content = f"""# Clean Cucumber Dataset Configuration
path: {output_dir.absolute()}
train: train/images
val: valid/images
test: test/images

nc: {len(self.class_names)}
names: {self.class_names}

# Dataset metadata
dataset_info:
  description: "Clean Cucumber High-Throughput Phenotyping Dataset"
  version: "3.0"
  classes: {len(self.class_names)}
  total_images: {self.stats['images_with_annotations']}
  cleaned_annotations: {self.stats['fixed_annotations']}
"""
        
        with open(output_dir / 'data.yaml', 'w') as f:
            f.write(yaml_content)
        
        print(f"✅ Created data.yaml: {output_dir / 'data.yaml'}")
    
    def generate_cleanup_report(self, output_file):
        """Generate a detailed cleanup report."""
        report = {
            'timestamp': str(Path.cwd()),
            'data_directory': str(self.data_dir),
            'statistics': self.stats,
            'recommendations': [
                'Remove images with no annotations',
                'Fix over-segmented cucumber annotations',
                'Balance class distribution',
                'Increase training epochs to 500+',
                'Use GPU for faster training',
                'Enable aggressive data augmentation'
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Cleanup report saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Clean up cucumber dataset annotations')
    parser.add_argument('--data-dir', required=True, help='Path to dataset directory')
    parser.add_argument('--output-dir', required=True, help='Output directory for clean dataset')
    parser.add_argument('--iou-threshold', type=float, default=0.3, help='IoU threshold for merging cucumbers')
    
    args = parser.parse_args()
    
    # Initialize cleanup
    cleanup = DataCleanup(args.data_dir)
    
    # Run cleanup process
    cleanup.analyze_data()
    cleanup.fix_over_segmented_cucumbers(args.iou_threshold)
    cleanup.create_clean_dataset(args.output_dir)
    
    # Generate report
    report_file = Path(args.output_dir) / 'cleanup_report.json'
    cleanup.generate_cleanup_report(report_file)
    
    print(f"\n🎉 Data cleanup completed!")
    print(f"📊 Check the report: {report_file}")

if __name__ == "__main__":
    main()
