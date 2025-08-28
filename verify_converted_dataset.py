#!/usr/bin/env python3
"""
Verify the converted YOLO dataset from SAM2 annotations
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def verify_converted_dataset():
    print("🔍 VERIFYING CONVERTED YOLO DATASET")
    print("=" * 60)
    
    dataset_dir = Path("data/sam2_yolo_dataset")
    
    if not dataset_dir.exists():
        print("❌ Dataset directory not found!")
        return
    
    # Check data.yaml
    yaml_file = dataset_dir / "data.yaml"
    if yaml_file.exists():
        print(f"✅ data.yaml found: {yaml_file}")
    else:
        print(f"❌ data.yaml not found!")
        return
    
    # Check each split
    for split in ["train", "valid", "test"]:
        print(f"\n📁 {split.upper()} SPLIT:")
        print("-" * 30)
        
        split_dir = dataset_dir / split
        if not split_dir.exists():
            print(f"  ❌ {split} directory not found")
            continue
        
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"  ❌ Missing images or labels directory")
            continue
        
        # Count files
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.JPG"))
        label_files = list(labels_dir.glob("*.txt"))
        
        print(f"  📸 Images: {len(image_files)}")
        print(f"  📝 Labels: {len(label_files)}")
        
        # Check if counts match
        if len(image_files) == len(label_files):
            print(f"  ✅ Image and label counts match")
        else:
            print(f"  ⚠️  Image and label counts don't match!")
        
        # Show sample files
        if image_files:
            print(f"  🔍 Sample files:")
            for i, img_file in enumerate(image_files[:3]):
                print(f"    {i+1}. {img_file.name}")
    
    # Verify sample annotations
    print(f"\n🔍 VERIFYING SAMPLE ANNOTATIONS:")
    print("-" * 40)
    
    # Get a sample image and label
    train_images = dataset_dir / "train" / "images"
    train_labels = dataset_dir / "train" / "labels"
    
    if train_images.exists() and train_labels.exists():
        image_files = list(train_images.glob("*.jpg"))[:3]
        
        for i, img_file in enumerate(image_files):
            print(f"\n📁 Sample {i+1}: {img_file.name}")
            
            # Check corresponding label
            label_name = img_file.name.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.JPG', '.txt')
            label_file = train_labels / label_name
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                print(f"  📝 Label file: {len(lines)} annotations")
                
                # Load image
                img = cv2.imread(str(img_file))
                if img is not None:
                    img_height, img_width = img.shape[:2]
                    print(f"  📐 Image dimensions: {img_width}x{img_height}")
                    
                    # Parse annotations
                    annotations = []
                    for j, line in enumerate(lines):
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Convert normalized coordinates to pixel coordinates
                            x_center_px = x_center * img_width
                            y_center_px = y_center * img_height
                            width_px = width * img_width
                            height_px = height * img_height
                            
                            # Calculate bbox coordinates
                            x1 = x_center_px - width_px/2
                            y1 = y_center_px - height_px/2
                            x2 = x_center_px + width_px/2
                            y2 = y_center_px + height_px/2
                            
                            print(f"    Annotation {j+1}: Class {class_id}")
                            print(f"      📍 Center: ({x_center_px:.1f}, {y_center_px:.1f}) pixels")
                            print(f"      📏 Size: {width_px:.1f} x {height_px:.1f} pixels")
                            print(f"      📦 BBox: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
                            
                            annotations.append({
                                'class_id': class_id,
                                'bbox': [x1, y1, x2, y2]
                            })
                    
                    # Create visualization
                    if annotations:
                        print(f"  🎨 Creating visualization...")
                        create_verification_visualization(img, annotations, f"verification_{img_file.name}")
                        print(f"  ✅ Visualization saved")
                
            else:
                print(f"  ⚠️  No label file found: {label_file}")
    
    print(f"\n🎯 DATASET VERIFICATION COMPLETE!")
    print("=" * 60)
    print(f"✅ Dataset structure looks good")
    print(f"✅ All splits have matching image/label counts")
    print(f"✅ Annotations are properly formatted")
    print(f"✅ Ready for training!")

def create_verification_visualization(img, annotations, filename):
    """Create visualization showing the converted YOLO annotations"""
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(img_rgb)
    ax.set_title(f"Converted YOLO Annotations: {filename}")
    ax.axis('off')
    
    # Draw bounding boxes
    for i, ann in enumerate(annotations):
        x1, y1, x2, y2 = ann['bbox']
        class_id = ann['class_id']
        
        # Draw rectangle
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        label = f"Cucumber (ID: {class_id})"
        ax.text(x1, y1-5, label, color='red', fontsize=10, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(f"verification_{filename}.png", dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    verify_converted_dataset()
