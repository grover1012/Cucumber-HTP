#!/usr/bin/env python3
"""
Verify and compare labels from both new_annotations and sam2_annotations datasets
"""

import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def verify_both_datasets():
    print("🔍 VERIFYING BOTH DATASETS")
    print("=" * 60)
    
    # Check new_annotations (YOLO format)
    print("\n📁 NEW_ANNOTATIONS (YOLO format)")
    print("-" * 40)
    verify_new_annotations()
    
    # Check sam2_annotations (COCO JSON format)
    print("\n📁 SAM2_ANNOTATIONS (COCO JSON format)")
    print("-" * 40)
    verify_sam2_annotations()

def verify_new_annotations():
    """Verify new_annotations dataset (YOLO format)"""
    train_dir = Path("data/new_annotations/train")
    train_images = train_dir / "images"
    train_labels = train_dir / "labels"
    
    if not train_images.exists() or not train_labels.exists():
        print("❌ Training directories not found!")
        return
    
    # Class names
    class_names = ['big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber', 'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice']
    
    # Get sample files
    image_files = list(train_images.glob("*.jpg"))[:5]  # First 5 images
    
    print(f"📸 Found {len(image_files)} sample images")
    
    for i, img_file in enumerate(image_files):
        print(f"\n🔍 Sample {i+1}: {img_file.name}")
        
        # Check corresponding label file
        label_file = train_labels / f"{img_file.stem}.txt"
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            print(f"  📝 Label file: {len(lines)} annotations")
            
            # Load image
            img = cv2.imread(str(img_file))
            if img is None:
                print("  ❌ Failed to load image")
                continue
            
            img_height, img_width = img.shape[:2]
            print(f"  📐 Image dimensions: {img_width}x{img_height}")
            
            # Parse and display annotations
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
                    
                    class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown_{class_id}"
                    
                    print(f"    Annotation {j+1}: {class_name} (ID: {class_id})")
                    print(f"      📍 Center: ({x_center_px:.1f}, {y_center_px:.1f}) pixels")
                    print(f"      📏 Size: {width_px:.1f} x {height_px:.1f} pixels")
                    
                    annotations.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': 1.0
                    })
            
            # Create visualization
            if annotations:
                print(f"  🎨 Creating visualization...")
                create_annotation_visualization(img, annotations, f"new_annotations_{img_file.name}")
                print(f"  ✅ Visualization saved")
            
        else:
            print(f"  ⚠️  No label file found: {label_file}")

def verify_sam2_annotations():
    """Verify sam2_annotations dataset (COCO JSON format)"""
    train_dir = Path("data/sam2_annotations/train")
    
    if not train_dir.exists():
        print("❌ Training directory not found!")
        return
    
    # Get sample files
    json_files = list(train_dir.glob("*.json"))[:5]  # First 5 JSON files
    
    print(f"📸 Found {len(json_files)} sample JSON files")
    
    for i, json_file in enumerate(json_files):
        print(f"\n🔍 Sample {i+1}: {json_file.name}")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract image info
            image_info = data.get('image', {})
            image_name = image_info.get('file_name', 'unknown')
            image_height = image_info.get('height', 0)
            image_width = image_info.get('width', 0)
            
            print(f"  🖼️  Image: {image_name}")
            print(f"  📐 Image dimensions: {image_width}x{image_height}")
            
            # Extract annotations
            annotations = data.get('annotations', [])
            print(f"  📝 JSON file: {len(annotations)} annotations")
            
            # Check if corresponding image exists
            image_file = train_dir / image_name
            if image_file.exists():
                print(f"  ✅ Image file found")
                
                # Load image
                img = cv2.imread(str(image_file))
                if img is not None:
                    # Parse annotations
                    parsed_annotations = []
                    for j, ann in enumerate(annotations):
                        bbox = ann.get('bbox', [])  # [x, y, width, height]
                        area = ann.get('area', 0)
                        segmentation = ann.get('segmentation', {})
                        
                        if len(bbox) >= 4:
                            x, y, w, h = bbox
                            x1, y1, x2, y2 = x, y, x + w, y + h
                            
                            print(f"    Annotation {j+1}:")
                            print(f"      📦 BBox: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
                            print(f"      📏 Size: {w:.1f} x {h:.1f} pixels")
                            print(f"      📊 Area: {area:.1f} pixels²")
                            
                            # Check if segmentation exists
                            if 'counts' in segmentation:
                                print(f"      🎭 Has segmentation mask")
                            else:
                                print(f"      ❌ No segmentation mask")
                            
                            parsed_annotations.append({
                                'class_id': 0,  # Default class for SAM2
                                'class_name': 'object',
                                'bbox': [x1, y1, x2, y2],
                                'area': area,
                                'has_segmentation': 'counts' in segmentation
                            })
                    
                    # Create visualization
                    if parsed_annotations:
                        print(f"  🎨 Creating visualization...")
                        create_annotation_visualization(img, parsed_annotations, f"sam2_annotations_{image_name}")
                        print(f"  ✅ Visualization saved")
                else:
                    print(f"  ❌ Failed to load image")
            else:
                print(f"  ⚠️  Image file not found: {image_name}")
                
        except Exception as e:
            print(f"  ❌ Error reading JSON file: {e}")

def create_annotation_visualization(img, annotations, filename):
    """Create a visualization showing the image with bounding boxes and labels"""
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(img_rgb)
    ax.set_title(f"Dataset Labels: {filename}")
    ax.axis('off')
    
    # Colors for different classes
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan', 'magenta', 'lime']
    
    # Draw bounding boxes and labels
    for i, ann in enumerate(annotations):
        x1, y1, x2, y2 = ann['bbox']
        class_name = ann['class_name']
        class_id = ann['class_id']
        
        # Choose color
        color = colors[class_id % len(colors)]
        
        # Draw rectangle
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        label = f"{class_name} (ID: {class_id})"
        ax.text(x1, y1-5, label, color=color, fontsize=10, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(f"dataset_check_{filename}.png", dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    verify_both_datasets()
