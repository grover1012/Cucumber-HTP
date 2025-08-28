#!/usr/bin/env python3
"""
Verify training data labels by showing images and their annotations
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def verify_labels():
    print("🔍 VERIFYING TRAINING DATA LABELS")
    print("=" * 60)
    
    # Training data paths
    train_dir = Path("data/yolov12/train")
    train_images = train_dir / "images"
    train_labels = train_dir / "labels"
    
    print(f"📁 Training images: {train_images}")
    print(f"📁 Training labels: {train_labels}")
    
    if not train_images.exists() or not train_labels.exists():
        print("❌ Training directories not found!")
        return
    
    # Get sample files
    image_files = list(train_images.glob("*.jpg"))[:10]  # First 10 images
    print(f"\n📸 Found {len(image_files)} sample images")
    
    # Class names
    class_names = ['big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber', 'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice']
    
    for i, img_file in enumerate(image_files):
        print(f"\n🔍 Sample {i+1}: {img_file.name}")
        print("-" * 50)
        
        # Check corresponding label file
        label_file = train_labels / f"{img_file.stem}.txt"
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            print(f"📝 Label file: {len(lines)} annotations")
            
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
                    print(f"      📦 BBox: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
                    
                    annotations.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': 1.0
                    })
            
            # Create visualization
            if annotations:
                print(f"  🎨 Creating visualization...")
                create_annotation_visualization(img, annotations, img_file.name)
                print(f"  ✅ Visualization saved as: annotation_check_{i+1}.png")
            
        else:
            print(f"  ⚠️  No label file found: {label_file}")
    
    print(f"\n🎯 LABEL VERIFICATION COMPLETE")
    print("=" * 60)
    print("Check the generated images to see if:")
    print("• Bounding boxes are around the right objects")
    print("• Class labels match what you see")
    print("• Annotations make sense for the image content")

def create_annotation_visualization(img, annotations, filename):
    """Create a visualization showing the image with bounding boxes and labels"""
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(img_rgb)
    ax.set_title(f"Training Data Labels: {filename}")
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
    plt.savefig(f"annotation_check_{filename}.png", dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    verify_labels()
