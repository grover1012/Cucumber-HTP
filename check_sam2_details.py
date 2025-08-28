#!/usr/bin/env python3
"""
Detailed check of sam2_annotations dataset
"""

import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def check_sam2_details():
    print("🔍 DETAILED SAM2_ANNOTATIONS CHECK")
    print("=" * 60)
    
    train_dir = Path("data/sam2_annotations/train")
    
    if not train_dir.exists():
        print("❌ Training directory not found!")
        return
    
    # Get all JSON files
    json_files = list(train_dir.glob("*.json"))
    print(f"📸 Found {len(json_files)} JSON annotation files")
    
    # Count annotations per file
    annotation_counts = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            annotations = data.get('annotations', [])
            annotation_counts.append(len(annotations))
        except:
            annotation_counts.append(0)
    
    print(f"\n📊 ANNOTATION STATISTICS:")
    print(f"  • Files with annotations: {sum(1 for c in annotation_counts if c > 0)}")
    print(f"  • Files without annotations: {sum(1 for c in annotation_counts if c == 0)}")
    print(f"  • Total annotations: {sum(annotation_counts)}")
    print(f"  • Average annotations per file: {sum(annotation_counts) / len(annotation_counts):.1f}")
    
    # Show files with most annotations
    print(f"\n🏆 TOP 5 FILES WITH MOST ANNOTATIONS:")
    file_annotation_pairs = list(zip(json_files, annotation_counts))
    file_annotation_pairs.sort(key=lambda x: x[1], reverse=True)
    
    for i, (json_file, count) in enumerate(file_annotation_pairs[:5]):
        print(f"  {i+1}. {json_file.name}: {count} annotations")
    
    # Detailed analysis of top files
    print(f"\n🔍 DETAILED ANALYSIS OF TOP FILES:")
    print("-" * 50)
    
    for i, (json_file, count) in enumerate(file_annotation_pairs[:3]):
        if count > 0:
            print(f"\n📁 File {i+1}: {json_file.name}")
            print(f"  📝 Annotations: {count}")
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract image info
                image_info = data.get('image', {})
                image_name = image_info.get('file_name', 'unknown')
                image_height = image_info.get('height', 0)
                image_width = image_info.get('width', 0)
                
                print(f"  🖼️  Image: {image_name}")
                print(f"  📐 Dimensions: {image_width}x{image_height}")
                
                # Analyze annotations
                annotations = data.get('annotations', [])
                areas = []
                bbox_sizes = []
                
                for j, ann in enumerate(annotations[:5]):  # Show first 5
                    bbox = ann.get('bbox', [])
                    area = ann.get('area', 0)
                    segmentation = ann.get('segmentation', {})
                    
                    if len(bbox) >= 4:
                        x, y, w, h = bbox
                        areas.append(area)
                        bbox_sizes.append((w, h))
                        
                        print(f"    Annotation {j+1}:")
                        print(f"      📦 BBox: ({x:.0f}, {y:.0f}) to ({x+w:.0f}, {y+h:.0f})")
                        print(f"      📏 Size: {w:.0f} x {h:.0f} pixels")
                        print(f"      📊 Area: {area:.0f} pixels²")
                        print(f"      🎭 Segmentation: {'Yes' if 'counts' in segmentation else 'No'}")
                
                # Show statistics
                if areas:
                    print(f"  📊 Statistics:")
                    print(f"    • Min area: {min(areas):.0f} pixels²")
                    print(f"    • Max area: {max(areas):.0f} pixels²")
                    print(f"    • Avg area: {sum(areas)/len(areas):.0f} pixels²")
                    
                    # Check if sizes make sense for cucumbers
                    avg_w = sum(w for w, h in bbox_sizes) / len(bbox_sizes)
                    avg_h = sum(h for w, h in bbox_sizes) / len(bbox_sizes)
                    print(f"    • Avg bbox size: {avg_w:.0f} x {avg_h:.0f} pixels")
                    
                    # Calculate aspect ratio
                    aspect_ratios = [w/h for w, h in bbox_sizes if h > 0]
                    if aspect_ratios:
                        avg_aspect = sum(aspect_ratios) / len(aspect_ratios)
                        print(f"    • Avg aspect ratio: {avg_aspect:.2f}")
                        
                        # Check if aspect ratios make sense for cucumbers (should be > 1 for length > width)
                        if avg_aspect > 1.2:
                            print(f"    ✅ Aspect ratio looks good for cucumbers (length > width)")
                        else:
                            print(f"    ⚠️  Aspect ratio seems low for cucumbers")
                
                # Create visualization
                image_file = train_dir / image_name
                if image_file.exists():
                    print(f"  🎨 Creating visualization...")
                    create_sam2_visualization(data, image_file, f"sam2_detailed_{image_name}")
                    print(f"  ✅ Visualization saved")
                
            except Exception as e:
                print(f"  ❌ Error processing file: {e}")
    
    print(f"\n🎯 SAM2 DATASET ASSESSMENT:")
    print("=" * 50)
    
    # Overall assessment
    total_annotations = sum(annotation_counts)
    files_with_annotations = sum(1 for c in annotation_counts if c > 0)
    
    if total_annotations > 100 and files_with_annotations > 10:
        print("✅ EXCELLENT: This dataset looks very promising!")
        print("   • High annotation count")
        print("   • Multiple files with annotations")
        print("   • Should provide good training data")
    elif total_annotations > 50 and files_with_annotations > 5:
        print("✅ GOOD: This dataset should work well")
        print("   • Decent annotation count")
        print("   • Multiple files with annotations")
    else:
        print("⚠️  CAUTION: Dataset might be too small")
        print("   • Low annotation count")
        print("   • Few files with annotations")

def create_sam2_visualization(data, image_file, filename):
    """Create detailed visualization of SAM2 annotations"""
    # Load image
    img = cv2.imread(str(image_file))
    if img is None:
        return
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left: Original image with bounding boxes
    ax1.imshow(img_rgb)
    ax1.set_title(f"Original Image with Bounding Boxes\n{filename}")
    ax1.axis('off')
    
    # Right: Original image with segmentation overlay
    ax2.imshow(img_rgb)
    ax2.set_title(f"Original Image with Segmentation\n{filename}")
    ax2.axis('off')
    
    # Get annotations
    annotations = data.get('annotations', [])
    
    # Colors for different annotations
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan', 'magenta', 'lime']
    
    for i, ann in enumerate(annotations):
        bbox = ann.get('bbox', [])
        segmentation = ann.get('segmentation', {})
        
        if len(bbox) >= 4:
            x, y, w, h = bbox
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # Choose color
            color = colors[i % len(colors)]
            
            # Draw bounding box on left image
            rect = patches.Rectangle((x1, y1), w, h, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax1.add_patch(rect)
            
            # Add annotation number
            ax1.text(x1, y1-5, f"#{i+1}", color=color, fontsize=12, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Draw bounding box on right image
            rect2 = patches.Rectangle((x1, y1), w, h, 
                                    linewidth=2, edgecolor=color, facecolor='none')
            ax2.add_patch(rect2)
            
            # Add annotation number
            ax2.text(x1, y1-5, f"#{i+1}", color=color, fontsize=12, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(f"sam2_detailed_{filename}.png", dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    check_sam2_details()
