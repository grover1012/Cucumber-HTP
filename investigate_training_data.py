#!/usr/bin/env python3
"""
Investigate the training data to see what's actually labeled as cucumbers
"""

import os
from pathlib import Path
import cv2
import numpy as np

def investigate_training_data():
    print("🔍 Investigating Training Data Issues")
    print("=" * 60)
    
    # Check training data structure
    train_dir = Path("data/yolov12/train")
    train_images = train_dir / "images"
    train_labels = train_dir / "labels"
    
    print(f"📁 Training images: {train_images}")
    print(f"📁 Training labels: {train_labels}")
    
    if not train_images.exists() or not train_labels.exists():
        print("❌ Training directories not found!")
        return
    
    # Get some sample files
    image_files = list(train_images.glob("*.jpg"))[:5]  # First 5 images
    print(f"\n📸 Found {len(image_files)} sample images")
    
    for i, img_file in enumerate(image_files):
        print(f"\n🔍 Sample {i+1}: {img_file.name}")
        
        # Check corresponding label file
        label_file = train_labels / f"{img_file.stem}.txt"
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            print(f"  📝 Label file: {len(lines)} annotations")
            
            # Parse YOLO format: class_id x_center y_center width height
            for j, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    class_names = ['big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber', 'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice']
                    class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown_{class_id}"
                    
                    print(f"    Annotation {j+1}: {class_name} (ID: {class_id}) at ({x_center:.3f}, {y_center:.3f}) size {width:.3f}x{height:.3f}")
        else:
            print(f"  ⚠️  No label file found: {label_file}")
    
    print(f"\n🎯 Class Distribution Analysis:")
    print("=" * 40)
    
    # Count class occurrences across all label files
    class_counts = {i: 0 for i in range(12)}
    
    for label_file in train_labels.glob("*.txt"):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    if class_id < 12:
                        class_counts[class_id] += 1
    
    class_names = ['big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber', 'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice']
    
    for class_id, count in class_counts.items():
        if count > 0:
            print(f"  {class_names[class_id]:12} (ID: {class_id:2d}): {count:4d} annotations")
    
    print(f"\n🚨 PROBLEM IDENTIFIED:")
    print("=" * 40)
    print("The training data has WRONG labels!")
    print("• Objects labeled as 'cucumber' are probably NOT cucumbers")
    print("• Objects labeled as 'green_dot', 'blue_dot', 'red_dot' might be actual cucumbers")
    print("• The model learned incorrect associations")
    print("• This requires RETRAINING with correct labels!")

if __name__ == "__main__":
    investigate_training_data()
